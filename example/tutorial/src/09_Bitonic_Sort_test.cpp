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
        UInt* inOut, // Raw pointer to input/output data array
        UInt start, // Starting index of the segment
        UInt length, // Length of the segment
        UInt dist, // Distance between elements to compare
        bool ascending) const // Sorting direction (true for ascending)
    {
        // Retrieve thread and block information
        auto [threadIndex] = acc[alpaka::layer::thread].idx();
        auto [blockDimension] = acc[alpaka::layer::thread].count();
        auto [blockIndex] = acc[alpaka::layer::block].idx();
        auto [gridDimension] = acc[alpaka::layer::block].count();

        // Calculate linear thread index and total grid size
        auto linearGridThreadIndex = blockDimension * blockIndex + threadIndex;
        auto linearGridSize = gridDimension * blockDimension;

        // Grid-strided loop to handle all elements assigned to the thread
        for(UInt globalIdx = linearGridThreadIndex; globalIdx < length; globalIdx += linearGridSize)
        {
            // Ensure indices are within the valid range
            if(globalIdx >= start && globalIdx < (start + length - dist) && (globalIdx + dist) < (start + length))
            {
                // Perform comparison and swap if necessary
                UInt lhs = inOut[globalIdx];
                UInt rhs = inOut[globalIdx + dist];
                if((ascending && lhs > rhs) || (!ascending && lhs < rhs))
                {
                    inOut[globalIdx] = rhs;
                    inOut[globalIdx + dist] = lhs;
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
    UInt threadsPerBlock = 256u; // Number of threads per block
    UInt blocks = (n + threadsPerBlock - 1u) / threadsPerBlock; // Number of blocks
    auto threadSpec = alpaka::onHost::ThreadSpec{Vec1D{blocks}, Vec1D{threadsPerBlock}};

    // Perform bitonic sort
    for(UInt length = 2u; length <= n; length <<= 1u)
    {
        for(UInt start = 0u; start < n; start += length)
        {
            bool ascending = ((start / length) % 2u) == 0u;
            for(UInt dist = (length >> 1u); dist > 0u; dist >>= 1u)
            {
                queue.enqueue(
                    computeExec,
                    threadSpec,
                    CompareSwapKernel{}, // Kernel function
                    deviceBuf.data(), // Raw pointer to device buffer
                    start,
                    length,
                    dist,
                    ascending);
                alpaka::onHost::wait(queue);
            }
        }
    }

    //-----------------------------------------
    // Copy device data back to host for modification
    //-----------------------------------------
    alpaka::onHost::memcpy(queue, hostBuf, deviceBuf);
    alpaka::onHost::wait(queue);

    // Modify host data (e.g., increment even numbers)
    for(UInt i = 0; i < n; ++i)
    {
        if(hostBuf[i] % 2 == 0) // Increment even numbers
        {
            hostBuf[i] += 1;
        }
    }

    // Copy modified data back to device
    alpaka::onHost::memcpy(queue, deviceBuf, hostBuf);
    alpaka::onHost::wait(queue);

    //-----------------------------------------
    // Copy final sorted data from device to host
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
    UInt size = 1024; // Original size of the array
    UInt paddedSize = nextPowerOfTwo(size); // Adjust to the next power of two
    std::vector<UInt> data(paddedSize, INF); // Initialize with INF for padding
    std::srand(static_cast<unsigned>(std::time(nullptr))); // Seed for random number generation
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
