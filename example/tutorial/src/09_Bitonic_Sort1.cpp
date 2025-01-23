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
        return 1u;
    double power = std::ceil(std::log2(static_cast<double>(n)));
    return static_cast<UInt>(std::pow(2.0, power));
}

//-------------------------------------
// Print the elements of an array
//-------------------------------------
void printArray(std::vector<UInt> const& arr, UInt n)
{
    for(UInt i = 0u; i < n; i++)
    {
        if(arr[i] != INF)
        {
            std::cout << arr[i] << " ";
        }
    }
    std::cout << std::endl;
}

//-------------------------------------
// Bitonic Compare-and-Swap Kernel (经典的双调排序实现)
//-------------------------------------
struct BitonicCompareSwapKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        auto inOut, // 数据数组
        UInt n, // 数组大小
        UInt k, // 当前子序列总长度
        UInt j // 当前比较步长
    ) const
    {
        // 这里采用典型的 grid-stride loop 或类似方式来并行遍历所有 i
        auto threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto numThreads = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        // 并行遍历 [0..n-1]
        for(UInt i = threadIdx; i < n; i += numThreads)
        {
            // 计算 partner 索引
            UInt partner = i ^ j; // 与经典算法一致

            // 确保 partner 在范围内
            if(partner < n)
            {
                // 判断这一段是要升序还是降序： (i & k) == 0 则升序，否则降序
                bool ascending = ((i & k) == 0);

                // 取出要比较的元素
                UInt lhs = inOut[i];
                UInt rhs = inOut[partner];

                // 按照 ascending 决定是否交换
                if((ascending && lhs > rhs) || (!ascending && lhs < rhs))
                {
                    inOut[i] = rhs;
                    inOut[partner] = lhs;
                }
            }
        }
    }
};

//-------------------------------------
// Bitonic Sort Using Alpaka
//-------------------------------------
int bitonicSortWithAlpaka(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec,
    std::vector<UInt>& arr,
    UInt n)
{
    //-----------------------------------------
    // Create a queue for task execution
    //-----------------------------------------
    alpaka::onHost::Queue queue = device.makeQueue();

    //-----------------------------------------
    // Allocate memory
    //-----------------------------------------
    auto hostBuf = alpaka::onHost::alloc<UInt>(host, Vec1D{n});
    auto deviceBuf = alpaka::onHost::allocMirror(device, hostBuf);

    // Initialize host buffer with input data
    for(UInt i = 0; i < n; ++i)
    {
        hostBuf[i] = arr[i];
    }

    // Copy data from host to device
    alpaka::onHost::memcpy(queue, deviceBuf, hostBuf);
    alpaka::onHost::wait(queue);

    //-----------------------------------------
    // Execute Bitonic Sort (经典两层循环)
    //-----------------------------------------
    // 准备网格大小
    UInt threadsPerBlock = 256u;
    UInt blocks = (n + threadsPerBlock - 1u) / threadsPerBlock;
    auto threadSpec = alpaka::onHost::ThreadSpec{Vec1D{blocks}, Vec1D{threadsPerBlock}};

    // 双调排序核心：外层 k（子序列大小），内层 j（步长）
    // k 从 2 翻倍到 n
    for(UInt k = 2u; k <= n; k <<= 1u)
    {
        // j 从 k/2 一路减到 1
        for(UInt j = (k >> 1u); j > 0u; j >>= 1u)
        {
            // 向队列中提交 Kernel，用 (k, j) 指定当前阶段
            queue.enqueue(computeExec, threadSpec, BitonicCompareSwapKernel{}, deviceBuf.getMdSpan(), n, k, j);
            alpaka::onHost::wait(queue);
        }
    }

    //-----------------------------------------
    // 拷回结果，并做演示性修改
    //-----------------------------------------
    // 1. 从 Device 拷贝回 Host
    alpaka::onHost::memcpy(queue, hostBuf, deviceBuf);
    alpaka::onHost::wait(queue);

    // 2. 修改偶数为偶数+1
    for(UInt i = 0; i < n; ++i)
    {
        if(hostBuf[i] % 2 == 0)
        {
            hostBuf[i] += 1;
        }
    }

    // 3. 再次拷回 Device
    alpaka::onHost::memcpy(queue, deviceBuf, hostBuf);
    alpaka::onHost::wait(queue);

    // 4. 最终结果再一次拷回 Host
    alpaka::onHost::memcpy(queue, hostBuf, deviceBuf);
    alpaka::onHost::wait(queue);

    // 把最终结果写回 vector
    for(UInt i = 0; i < n; ++i)
    {
        arr[i] = hostBuf[i];
    }

    return EXIT_SUCCESS;
}

//-------------------------------------
// Example function using cfg
//-------------------------------------
int example(auto const cfg)
{
    auto deviceApi = cfg[alpaka::object::api];
    auto computeExec = cfg[alpaka::object::exec];

    // Initialize the accelerator platform for the selected device API
    alpaka::onHost::Platform platform = alpaka::onHost::makePlatform(deviceApi);

    // Check if at least one device is available
    std::size_t nDev = alpaka::onHost::getDeviceCount(platform);
    if(nDev == 0)
    {
        return EXIT_FAILURE;
    }

    // Create a host platform and device
    alpaka::onHost::Platform host_platform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = host_platform.makeDevice(0);
    std::cout << "Host:   " << alpaka::onHost::getName(host) << "\n\n";

    // Select the first device from the accelerator platform
    alpaka::onHost::Device device = platform.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    // Prepare the data for sorting
    UInt size = 2048u; // 原始大小
    UInt paddedSize = nextPowerOfTwo(size); // 向上补到 2 的幂
    std::vector<UInt> data(paddedSize, INF);

    // 随机填充前面 size 个元素
    std::srand(1234); // 固定随机种子
    for(UInt i = 0; i < size; ++i)
    {
        data[i] = static_cast<UInt>(std::rand() % 1000);
    }

    std::cout << "Unsorted array:\n";
    printArray(data, size);

    // Perform Bitonic Sort using Alpaka
    if(bitonicSortWithAlpaka(host, device, computeExec, data, paddedSize) == EXIT_SUCCESS)
    {
        std::cout << "Sorted array:\n";
        printArray(data, size);
    }

    return EXIT_SUCCESS;
}

//-------------------------------------
// Main function
//-------------------------------------
int main()
{
    // 测试 example 函数 (对所有已启用的 Alpaka API/Executor)
    return alpaka::executeForEach(
        [=](auto const& tag) { return example(tag); },
        alpaka::onHost::allExecutorsAndApis(alpaka::onHost::enabledApis));
}
