#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <vector>
#include <chrono>

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
    UInt n,
    double& totalKernelTime) 
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

    for(UInt length = 2u; length <= n; length *= 2u) // Outer loop: Gradually increase the segment size to sort.
    {
        {

            for(UInt dist = (length / 2u); dist > 0u;dist /= 2u) // Inner loop: Perform comparisons at decreasing distances.
            {

                auto numSegments = alpaka::core::divCeil(n, length);
                constexpr auto frameExtent = 512u;
                auto numFrames = alpaka::core::divCeil(length, frameExtent);
                auto frameSpec = alpaka::onHost::FrameSpec{
                    Vec2D{numSegments, numFrames},
                    alpaka::CVec<uint32_t, 1u, frameExtent>{}};
                auto startTime = std::chrono::high_resolution_clock::now();
                queue.enqueue(
                    computeExec, // The execution policy for the computation (e.g., GPU execution).
                    frameSpec, // The frame specification that defines thread and block layout.
                    CompareSwapKernel{}, // The kernel function to perform the comparison and swapping.
                    deviceBuf.getMdSpan(), // A pointer to the device buffer containing the data to be sorted.
                    length, // The size of the current segment.
                    dist, // The distance between elements being compared in this iteration.
                    n // The sorting order for this segment (true = ascending, false = descending).
                );
                alpaka::onHost::wait(queue); // Synchronize the host with the GPU.
                auto endTime = std::chrono::high_resolution_clock::now();
                totalKernelTime += std::chrono::duration<double>(endTime - startTime).count();



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
int example(auto const cfg, auto in_size)
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
    //std::cout << "Host:   " << alpaka::onHost::getName(host) << "\n\n";

    // Select the first device from the accelerator platform
    alpaka::onHost::Device device = platform.makeDevice(0);
    //std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    // Prepare the data for sorting
    UInt size = in_size; // Original size of the array
    UInt paddedSize = nextPowerOfTwo(size); // Adjust to the next power of two
    std::vector<UInt> data(paddedSize, INF); // Initialize with INF for padding
    std::srand(1234); // Seed for reproducibility
    for(UInt i = 0; i < size; ++i)
    {
        data[i] = static_cast<UInt>(std::rand() % 1000); // Generate random numbers
    }

    //std::cout << "Unsorted array:\n";
    //printArray(data, size); // Print the unsorted array

    double totalKernelTime = 0.0; 


    // Perform Bitonic Sort using Alpaka
    //if(bitonicSortWithAlpaka(host, device, computeExec, data, paddedSize, totalKernelTime) == EXIT_SUCCESS)
    //{
    //    //std::cout << "Sorted array:\n";
    //    //printArray(data, size); // Print the sorted array
    //}


      int result = bitonicSortWithAlpaka(host, device, computeExec, data, paddedSize, totalKernelTime); // 


    std::cout << alpaka::onHost::getName(device) << ", " << paddedSize << ", " << totalKernelTime << ", "
              << (result == EXIT_SUCCESS ? "success" : "failure") << std::endl;

    return EXIT_SUCCESS; // Indicate successful execution
}

//-------------------------------------
// Main function
//-------------------------------------
int main()
{
    std::cout << "Device, Problem Size, T Kernel Exec (s), Results" << std::endl << std::flush;
    // Test the example function with all enabled APIs and executors
    UInt maxsize = 1024 * 1024 * 1024 / 4;
    for(UInt n = 1024; n <= maxsize; n = n * 2)
    {
        alpaka::executeForEach(
            [=](auto const& tag) { return example(tag, n); },
            alpaka::onHost::allExecutorsAndApis(alpaka::onHost::enabledApis));
    }
    return 0;
}
