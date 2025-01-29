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
        auto start, // Starting index of the segment
        auto length, // Length of the segment
        auto dist, // Distance between elements to compare
        bool ascending) const // Sorting direction (true for ascending)
    {
        Vec1D linearNumFrames = acc[alpaka::frame::count].product();
        auto frameExtent = acc[alpaka::frame::extent];
        auto frameDataExtent = linearNumFrames * frameExtent;
        Vec1D linearFrameExtent = frameExtent.product();
        auto traverseInFrame
            = alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{frameExtent});
        auto traverseOverFrames = alpaka::onAcc::makeIdxMap(
            acc,
            alpaka::onAcc::worker::blocksInGrid,
            alpaka::IdxRange{alpaka::CVec<uint32_t, 0u>{}, frameDataExtent, linearFrameExtent});


        for(auto frameIdx : traverseOverFrames)
        {
            for(auto elemIdx : traverseInFrame)
            {
                for(auto [i] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::WorkerGroup{frameIdx + elemIdx, frameDataExtent},
                        alpaka::IdxRange{Vec1D{start}, Vec1D{start + length - dist}}))
                {
                    // if (i >= start && i < start + length - dist && i + dist < start + length)
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
    double& totalKernelTime) // Size of the array
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
    UInt threadsPerBlock = 256u; // Number of threads per block
    UInt blocks = (n + threadsPerBlock - 1u) / threadsPerBlock; // Number of blocks
    auto threadSpec = alpaka::onHost::ThreadSpec{Vec1D{blocks}, Vec1D{threadsPerBlock}};

    constexpr auto frameExtent = 256u;
    auto numFrames = Vec1D{alpaka::core::divCeil(n, frameExtent * 4)};
    auto frameSpec = alpaka::onHost::FrameSpec{numFrames, alpaka::CVec<uint32_t, frameExtent>{}};

    //-----------------------------------------
    // Perform bitonic sort
    // The bitonic sort process is controlled by three nested loops.
    // The outer loop manages the length of the segments being sorted.
    // The middle loop iterates over each segment in the array.
    // The innermost loop reduces the comparison distance (dist) within each segment.
    //-----------------------------------------

    for(UInt length = 2u; length <= n; length <<= 1u) // Outer loop: Gradually increase the segment size to sort.
    {
        /*
         * Purpose:
         * - The `length` variable represents the size of the segments being processed in this stage.
         * - Initially, `length = 2`, meaning we compare pairs of elements.
         * - For the next stage, `length = 4`, we process 4-element segments.
         * - The segment size doubles on each iteration until it equals the total size of the array (`n`).
         */

        for(UInt start = 0u; start < n; start += length) // Middle loop: Iterate over each segment in the array.
        {
            /*
             * Purpose:
             * - Divide the array into segments of size `length`.
             * - `start` is the beginning index of the current segment.
             * - The loop increments by `length` so that each iteration processes the next segment.
             */

            bool ascending = ((start / length) % 2u) == 0u;
            /*
             * Purpose:
             * - Determine the sorting order (ascending or descending) for the current segment.
             * - If `(start / length) % 2 == 0`, the segment is sorted in ascending order.
             * - Otherwise, the segment is sorted in descending order.
             * - This alternation ensures that bitonic sequences are formed, which are necessary for the bitonic merge.
             */

            for(UInt dist = (length >> 1u); dist > 0u;
                dist >>= 1u) // Inner loop: Perform comparisons at decreasing distances.
            {
                /*
                 * Purpose:
                 * - The `dist` variable represents the distance between elements being compared in the segment.
                 * - Initially, `dist = length / 2`, meaning elements halfway apart in the segment are compared.
                 * - After each iteration, the distance is halved until `dist = 1`, meaning adjacent elements are
                 * compared.
                 */
                auto startTime = std::chrono::high_resolution_clock::now();
                queue.enqueue(
                    computeExec, // The execution policy for the computation (e.g., GPU execution).
                    frameSpec, // The frame specification that defines thread and block layout.
                    CompareSwapKernel{}, // The kernel function to perform the comparison and swapping.
                    deviceBuf.getMdSpan(), // A pointer to the device buffer containing the data to be sorted.
                    start, // The starting index of the current segment.
                    length, // The size of the current segment.
                    dist, // The distance between elements being compared in this iteration.
                    ascending // The sorting order for this segment (true = ascending, false = descending).
                );
                /*
                 * Purpose:
                 * - Enqueue the `CompareSwapKernel` on the GPU to perform the comparison and swapping operation.
                 * - The kernel is called with parameters defining the segment of the array being processed,
                 *   the distance between compared elements, and the sorting order.
                 */

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

    //// Modify host data (e.g., increment even numbers)
    //for(UInt i = 0; i < n; ++i)
    //{
    //    if(hostBuf[i] % 2 == 0) // Increment even numbers
    //    {
    //        hostBuf[i] += 1;
    //    }
    //}

    //// Copy modified data back to device
    //alpaka::onHost::memcpy(queue, deviceBuf, hostBuf);
    //alpaka::onHost::wait(queue);

    ////-----------------------------------------
    //// Copy final sorted data from device to host
    ////-----------------------------------------
    //alpaka::onHost::memcpy(queue, hostBuf, deviceBuf);
    //alpaka::onHost::wait(queue);

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

    double totalKernelTime = 0.0; // ������ʱ����
    // Perform Bitonic Sort using Alpaka
    if(bitonicSortWithAlpaka(host, device, computeExec, data, paddedSize,totalKernelTime) == EXIT_SUCCESS)
    {
        //std::cout << "Sorted array:\n";
        //printArray(data, size); // Print the sorted array
    }

        
    int result = bitonicSortWithAlpaka(host, device, computeExec, data, paddedSize, totalKernelTime); // ���ݼ�ʱ����

    


            // ��ָ����ʽ������
    std::cout << alpaka::onHost::getName(device) << ", " << paddedSize << ", " << totalKernelTime << ", "
              << (result == EXIT_SUCCESS ? "success" : "failure") << std::endl << std::flush;




    return EXIT_SUCCESS; // Indicate successful execution



}

//-------------------------------------
// Main function
//-------------------------------------
int main()
{
    std::cout << "Device, Problem Size, T Kernel Exec (s), Results" << std::endl  << std::flush;
    // Test the example function with all enabled APIs and executors
    UInt size = 1024*1024;
    for(auto i = 0; i < 11; i++){
        size *= pow(2,i);
        alpaka::executeForEach(
        [=](auto const& tag) { return example(tag, size); },
        alpaka::onHost::allExecutorsAndApis(alpaka::onHost::enabledApis));
    }
    return 0;
}
