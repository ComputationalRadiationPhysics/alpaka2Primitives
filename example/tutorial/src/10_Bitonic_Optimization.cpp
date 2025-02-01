#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <vector>

using namespace alpaka;

//-------------------------------------
// Helper types and constant definitions
//-------------------------------------
using UInt = uint32_t; // Alias for unsigned 32-bit integer
using Vec1D = alpaka::Vec<std::uint32_t, 1u>; // 1D vector type using Alpaka
static constexpr UInt INF = std::numeric_limits<UInt>::max(); // Constant representing infinity

//-------------------------------------
// Calculate the smallest power of 2 >= n
//-------------------------------------
UInt nextPowerOfTwo(UInt n)
{
    if(n == 0u)
        return 1u; // Return 1 for input 0
    double power = std::ceil(std::log2(static_cast<double>(n))); // Calculate log2(n) and round up
    return static_cast<UInt>(std::pow(2.0, power)); // Return 2^power
}

//-------------------------------------
// Print the elements of an array
//-------------------------------------
void printArray(std::vector<UInt> const& arr, UInt n)
{
    for(UInt i = 0u; i < n; i++)
    {
        if(arr[i] != INF) // Skip elements with value INF
        {
            std::cout << arr[i] << " "; // Print the array element
        }
    }
    std::cout << std::endl; // Print newline after the array
}

//-------------------------------------
// Comparison and Swap Kernel
//-------------------------------------
struct CompareSwapKernel
{
    // Kernel operator to perform comparison and swapping
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, // Alpaka accelerator object
        auto inOut, // Raw pointer to input/output data array
        auto length, // Length of the segment
        auto dist, // Distance between elements to compare
        uint32_t paddedSize) const // Sorting direction (true for ascending)
    {
        // x is the number of frames required to iterate over length elements
        // y is the number of segments in older versions called Middle loop on the host side
        auto numFrames = acc[alpaka::frame::count];
        auto frameExtents = acc[alpaka::frame::extent];
        auto frameDataExtents = numFrames * frameExtents;

        auto traverseInFrame
            = alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{frameExtents});

        auto traverseOverFrames = onAcc::makeIdxMap(
            acc,
            onAcc::worker::blocksInGrid,
            IdxRange{alpaka::CVec<uint32_t, 0u, 0u>{}, frameDataExtents, frameExtents});


        // 2D for loop
        for(auto frameIdx : traverseOverFrames)
        {
            auto segment = frameIdx.y();
            auto start = segment * length;
            bool ascending = ((start / length) % 2u) == 0u;

            // iterate only over the X-dimension
            // 1D for loop
            for(auto elemIdx : traverseInFrame[alpaka::CVec<uint32_t, 1u>{}])
            {
                // 1D for loop
                for(auto [i] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::WorkerGroup{Vec1D{elemIdx.x() + frameIdx.x()}, Vec1D{frameDataExtents.x()}},
                        alpaka::IdxRange{Vec1D{start}, Vec1D{start + length - dist}}))
                {
                    auto ll = inOut[i];
                    auto rr = inOut[i + dist];
                    if((ascending && ll > rr) || (!ascending && ll < rr))
                    {
                        inOut[i] = rr;
                        inOut[i + dist] = ll;
                    }
                }
            }
        }
    }
};

//-------------------------------------
// Bitonic Sort Using Alpaka
//-------------------------------------
int bitonicSortWithAlpaka(
    alpaka::onHost::concepts::Device auto host, // Host device object
    alpaka::onHost::concepts::Device auto device, // Accelerator device object
    auto computeExec, // Alpaka execution object
    std::vector<UInt>& arr, // Array to sort
    UInt n) // Size of the array
{
    //-----------------------------------------
    // Create a queue for task execution
    //-----------------------------------------
    alpaka::onHost::Queue queue = device.makeQueue();

    //-----------------------------------------
    // Allocate memory
    //-----------------------------------------
    auto hostBuf = alpaka::onHost::alloc<UInt>(host, Vec1D{n}); // Allocate host memory
    auto deviceBuf = alpaka::onHost::allocMirror(device, hostBuf); // Allocate device memory

    // Initialize host buffer with input data
    for(UInt i = 0; i < n; ++i)
    {
        hostBuf[i] = arr[i];
    }

    // Copy data from host to device
    alpaka::onHost::memcpy(queue, deviceBuf, hostBuf);
    alpaka::onHost::wait(queue);

    //-----------------------------------------
    // Execute kernel on device memory
    //-----------------------------------------
    //-----------------------------------------
    // Perform bitonic sort
    // The bitonic sort process is controlled by three nested loops.
    // The outer loop manages the length of the segments being sorted.
    // The middle loop iterates over each segment in the array.
    // The innermost loop reduces the comparison distance (dist) within each segment.
    //-----------------------------------------

    for(UInt length = 2u; length <= n; length *= 2u) // Outer loop: Gradually increase the segment size to sort.
    {
        /*
         * Purpose:
         * - The `length` variable represents the size of the segments being processed in this stage.
         * - Initially, `length = 2`, meaning we compare pairs of elements.
         * - For the next stage, `length = 4`, we process 4-element segments.
         * - The segment size doubles on each iteration until it equals the total size of the array (`n`).
         */

        // for(UInt start = 0u; start < n; start += length) // Middle loop: Iterate over each segment in the array.
        {
            /*
             * Purpose:
             * - Divide the array into segments of size `length`.
             * - `start` is the beginning index of the current segment.
             * - The loop increments by `length` so that each iteration processes the next segment.
             */


            /*
             * Purpose:
             * - Determine the sorting order (ascending or descending) for the current segment.
             * - If `(start / length) % 2 == 0`, the segment is sorted in ascending order.
             * - Otherwise, the segment is sorted in descending order.
             * - This alternation ensures that bitonic sequences are formed, which are necessary for the bitonic merge.
             */

            for(UInt dist = (length / 2u); dist > 0u;
                dist /= 2u) // Inner loop: Perform comparisons at decreasing distances.
            {
                /*
                 * Purpose:
                 * - The `dist` variable represents the distance between elements being compared in the segment.
                 * - Initially, `dist = length / 2`, meaning elements halfway apart in the segment are compared.
                 * - After each iteration, the distance is halved until `dist = 1`, meaning adjacent elements are
                 * compared.
                 */

                auto numSegments = alpaka::core::divCeil(n, length);
                constexpr auto frameExtent = 8u;
                auto numFrames = alpaka::core::divCeil(length, frameExtent);
                auto frameSpec = alpaka::onHost::FrameSpec{
                    Vec2D{numSegments, numFrames},
                    alpaka::CVec<uint32_t, 1u, frameExtent>{}};

                queue.enqueue(
                    computeExec, // The execution policy for the computation (e.g., GPU execution).
                    frameSpec, // The frame specification that defines thread and block layout.
                    CompareSwapKernel{}, // The kernel function to perform the comparison and swapping.
                    deviceBuf.getMdSpan(), // A pointer to the device buffer containing the data to be sorted.
                    length, // The size of the current segment.
                    dist, // The distance between elements being compared in this iteration.
                    n // The sorting order for this segment (true = ascending, false = descending).
                );
                /*
                 * Purpose:
                 * - Enqueue the `CompareSwapKernel` on the GPU to perform the comparison and swapping operation.
                 * - The kernel is called with parameters defining the segment of the array being processed,
                 *   the distance between compared elements, and the sorting order.
                 */

                alpaka::onHost::wait(queue); // Synchronize the host with the GPU.
            }
        }
    }


    //-----------------------------------------
    // Copy device data back to host for modification
    //-----------------------------------------
    alpaka::onHost::memcpy(queue, hostBuf, deviceBuf);
    alpaka::onHost::wait(queue);

    // Update the input array with sorted data
    for(UInt i = 0; i < n; ++i)
    {
        arr[i] = hostBuf[i];
    }

    return EXIT_SUCCESS; // Indicate success
}

//-------------------------------------
// Example function using cfg
//-------------------------------------
int example(auto const cfg)
{
    // Retrieve the device API and execution policy from the configuration
    auto deviceApi = cfg[alpaka::object::api];
    auto computeExec = cfg[alpaka::object::exec];

    // Initialize the accelerator platform for the selected device API
    alpaka::onHost::Platform platform = alpaka::onHost::makePlatform(deviceApi);

    // Check if at least one device is available
    std::size_t n = alpaka::onHost::getDeviceCount(platform);
    if(n == 0)
    {
        return EXIT_FAILURE; // Exit if no devices are found
    }

    // Create a host platform and device
    alpaka::onHost::Platform host_platform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = host_platform.makeDevice(0);
    std::cout << "Host:   " << alpaka::onHost::getName(host) << "\n\n";

    // Select the first device from the accelerator platform
    alpaka::onHost::Device device = platform.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    // Prepare the data for sorting
    UInt size = 111; // Original size of the array
    UInt paddedSize = nextPowerOfTwo(size); // Adjust to the next power of two
    std::vector<UInt> data(paddedSize, INF); // Initialize with INF for padding
    std::srand(1234); // Seed for reproducibility
    for(UInt i = 0; i < size; ++i)
    {
        data[i] = static_cast<UInt>(std::rand() % 1000); // Generate random numbers
    }

    std::cout << "Unsorted array:\n";
    printArray(data, size); // Print the unsorted array

    // Perform Bitonic Sort using Alpaka
    if(bitonicSortWithAlpaka(host, device, computeExec, data, paddedSize) == EXIT_SUCCESS)
    {
        std::cout << "Sorted array:\n";
        printArray(data, size); // Print the sorted array
    }

    return EXIT_SUCCESS; // Indicate successful execution
}

//-------------------------------------
// Main function
//-------------------------------------
int main()
{
    // Test the example function with all enabled APIs and executors
    return alpaka::executeForEach(
        [=](auto const& tag) { return example(tag); },
        alpaka::onHost::allExecutorsAndApis(alpaka::onHost::enabledApis));
}
