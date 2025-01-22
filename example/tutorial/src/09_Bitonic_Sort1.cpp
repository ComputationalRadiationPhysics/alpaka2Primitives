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
// ���������볣������
//-------------------------------------
using UInt = uint32_t;
using Vec1D = alpaka::Vec<std::uint32_t, 1u>;
static constexpr UInt INF = std::numeric_limits<UInt>::max();

//-------------------------------------
// ���� >= n ����С2����
//-------------------------------------
UInt nextPowerOfTwo(UInt n)
{
    if(n == 0u)
        return 1u;
    double power = std::ceil(std::log2(static_cast<double>(n)));
    return static_cast<UInt>(std::pow(2.0, power));
}

//-------------------------------------
// ��ӡ����
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
// �ȽϽ����ں�
//-------------------------------------
struct CompareSwapKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        auto const inOut,
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

        // Grid-strided loop
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
// ʹ�� Alpaka ʵ�ֵ� Bitonic Sort
//-------------------------------------
int bitonicSortWithAlpaka(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec,
    std::vector<UInt>& arr,
    UInt n)
{
    //-----------------------------------------
    // ��������
    //-----------------------------------------
    alpaka::onHost::Queue queue = device.makeQueue();
    //-----------------------------------------
    // �����ڴ�
    //-----------------------------------------
    auto hostBuf = alpaka::onHost::alloc<UInt>(host, Vec1D{n});
    auto deviceBuf = alpaka::onHost::allocMirror(device, hostBuf);

    // ��ʼ������������
    for(UInt i = 0; i < n; ++i)
    {
        hostBuf[i] = arr[i];
    }

    // �������豸����
    alpaka::onHost::memcpy(queue, deviceBuf, hostBuf);
    alpaka::onHost::wait(queue);

    //-----------------------------------------
    // ���ú˺����̷ֲ߳�
    //-----------------------------------------
    UInt threadsPerBlock = 256u;
    UInt blocks = (n + threadsPerBlock - 1u) / threadsPerBlock;

    auto threadSpec = alpaka::onHost::ThreadSpec{Vec1D{blocks}, Vec1D{threadsPerBlock}};

    //-----------------------------------------
    // Bitonic Sort �����߼�
    //-----------------------------------------
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
                    deviceBuf.getMdSpan(),
                    start,
                    length,
                    dist,
                    ascending);
                alpaka::onHost::wait(queue);
            }
        }
    }

    //-----------------------------------------
    // �豸�������������
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
// ʹ�� cfg �� example ����
//-------------------------------------
int example(auto const cfg)
{
    auto deviceApi = cfg[alpaka::object::api];
    auto computeExec = cfg[alpaka::object::exec];

    // ��ʼ��ƽ̨���豸
    alpaka::onHost::Platform platform = alpaka::onHost::makePlatform(deviceApi);
    if(alpaka::onHost::getDeviceCount(platform) == 0)
    {
        std::cerr << "No devices available for the selected API.\n";
        return EXIT_FAILURE;
    }

    alpaka::onHost::Platform hostPlatform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = hostPlatform.makeDevice(0);
    alpaka::onHost::Device device = platform.makeDevice(0);

    std::cout << "Host:   " << alpaka::onHost::getName(host) << "\n";
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n";

    // ׼����������
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

    // ���� Bitonic Sort
    if(bitonicSortWithAlpaka(host, device, computeExec, data, paddedSize) == EXIT_SUCCESS)
    {
        std::cout << "Sorted array:\n";
        printArray(data, size);
    }

    return EXIT_SUCCESS;
}

//-------------------------------------
// ������
//-------------------------------------
auto main() -> int
{
    // ���������õ� API ��ִ�������в���
    return alpaka::executeForEach(
        [=](auto const& tag) { return example(tag); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
}
