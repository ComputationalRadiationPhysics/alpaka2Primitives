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
// 辅助类型与常量定义
//-------------------------------------
using UInt = uint32_t;
using Vec1D = alpaka::Vec<std::uint32_t, 1u>;
static constexpr UInt INF = std::numeric_limits<UInt>::max();

//-------------------------------------
// 计算 >= n 的最小2次幂
//-------------------------------------
UInt nextPowerOfTwo(UInt n)
{
    if(n == 0u)
        return 1u;
    double power = std::ceil(std::log2(static_cast<double>(n)));
    return static_cast<UInt>(std::pow(2.0, power));
}

//-------------------------------------
// 打印数组
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
// 比较交换内核
//-------------------------------------
struct CompareSwapKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        UInt* inOut, // 使用原始指针替代 mdspan
        UInt start,
        UInt length,
        UInt dist,
        bool ascending) const
    {
        auto [threadIndex] = acc[alpaka::layer::thread].idx();
        auto [blockDimension] = acc[alpaka::layer::thread].count();
        auto [blockIndex] = acc[alpaka::layer::block].idx();
        auto [gridDimension] = acc[alpaka::layer::block].count();

        auto linearGridThreadIndex = blockDimension * blockIndex + threadIndex;
        auto linearGridSize = gridDimension * blockDimension;

        for(UInt globalIdx = linearGridThreadIndex; globalIdx < length; globalIdx += linearGridSize)
        {
            if(globalIdx >= start && globalIdx < (start + length - dist) && (globalIdx + dist) < (start + length))
            {
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
// 使用 Alpaka 实现的 Bitonic Sort
//-------------------------------------
int bitonicSortWithAlpaka(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec,
    std::vector<UInt>& arr,
    UInt n)
{
    //-----------------------------------------
    // 创建队列
    //-----------------------------------------
    alpaka::onHost::Queue queue = device.makeQueue();
    //-----------------------------------------
    // 分配内存
    //-----------------------------------------
    auto hostBuf = alpaka::onHost::alloc<UInt>(host, Vec1D{n});
    auto deviceBuf = alpaka::onHost::allocMirror(device, hostBuf);

    // 初始化主机缓冲区
    for(UInt i = 0; i < n; ++i)
    {
        hostBuf[i] = arr[i];
    }

    // 主机到设备拷贝
    alpaka::onHost::memcpy(queue, deviceBuf, hostBuf);
    alpaka::onHost::wait(queue);

    //-----------------------------------------
    // 在设备内存执行核函数
    //-----------------------------------------
    UInt threadsPerBlock = 256u;
    UInt blocks = (n + threadsPerBlock - 1u) / threadsPerBlock;
    auto threadSpec = alpaka::onHost::ThreadSpec{Vec1D{blocks}, Vec1D{threadsPerBlock}};

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
                    CompareSwapKernel{},
                    deviceBuf.data(), // 传递原始指针
                    start,
                    length,
                    dist,
                    ascending);
                alpaka::onHost::wait(queue);
            }
        }
    }

    //-----------------------------------------
    // 从设备内存拷贝到主机内存以修改
    //-----------------------------------------
    alpaka::onHost::memcpy(queue, hostBuf, deviceBuf);
    alpaka::onHost::wait(queue);

    // 修改主机内存
    for(UInt i = 0; i < n; ++i)
    {
        if(hostBuf[i] % 2 == 0) // 示例修改：将偶数加 1
        {
            hostBuf[i] += 1;
        }
    }

    // 将修改后的数据拷贝回设备内存
    alpaka::onHost::memcpy(queue, deviceBuf, hostBuf);
    alpaka::onHost::wait(queue);

    //-----------------------------------------
    // 设备到主机拷贝最终结果
    //-----------------------------------------
    alpaka::onHost::memcpy(queue, hostBuf, deviceBuf);
    alpaka::onHost::wait(queue);

    for(UInt i = 0; i < n; ++i)
    {
        arr[i] = hostBuf[i];
    }

    return EXIT_SUCCESS;
}

//-------------------------------------
// 使用 cfg 的 example 函数
//-------------------------------------
int example(auto const cfg)
{
    auto deviceApi = cfg[alpaka::object::api];
    auto computeExec = cfg[alpaka::object::exec];

    // initialise the accelerator platform
    alpaka::onHost::Platform platform = alpaka::onHost::makePlatform(deviceApi);

    // require at least one device
    std::size_t n = alpaka::onHost::getDeviceCount(platform);

    if(n == 0)
    {
        return EXIT_FAILURE;
    }

    // use the single host device
    alpaka::onHost::Platform host_platform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = host_platform.makeDevice(0);
    std::cout << "Host:   " << alpaka::onHost::getName(host) << "\n\n";

    // use the first device
    alpaka::onHost::Device device = platform.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    // 准备排序数据
    UInt size = 1024;
    UInt paddedSize = nextPowerOfTwo(size);
    std::vector<UInt> data(paddedSize, INF);
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for(UInt i = 0; i < size; ++i)
    {
        data[i] = static_cast<UInt>(std::rand() % 1000);
    }

    std::cout << "Unsorted array:\n";
    printArray(data, size);

    // 调用 Bitonic Sort
    if(bitonicSortWithAlpaka(host, device, computeExec, data, paddedSize) == EXIT_SUCCESS)
    {
        std::cout << "Sorted array:\n";
        printArray(data, size);
    }

    return EXIT_SUCCESS;
}

//-------------------------------------
// 主函数
//-------------------------------------
int main()
{
    // 对所有启用的 API 和执行器进行测试
    return alpaka::executeForEach(
        [=](auto const& tag) { return example(tag); },
        alpaka::onHost::allExecutorsAndApis(alpaka::onHost::enabledApis));
}
