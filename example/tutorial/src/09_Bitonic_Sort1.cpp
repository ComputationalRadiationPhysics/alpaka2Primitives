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
// Bitonic Compare-and-Swap Kernel (�����˫������ʵ��)
//-------------------------------------
struct BitonicCompareSwapKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        auto inOut, // ��������
        UInt n, // �����С
        UInt k, // ��ǰ�������ܳ���
        UInt j // ��ǰ�Ƚϲ���
    ) const
    {
        // ������õ��͵� grid-stride loop �����Ʒ�ʽ�����б������� i
        auto threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto numThreads = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        // ���б��� [0..n-1]
        for(UInt i = threadIdx; i < n; i += numThreads)
        {
            // ���� partner ����
            UInt partner = i ^ j; // �뾭���㷨һ��

            // ȷ�� partner �ڷ�Χ��
            if(partner < n)
            {
                // �ж���һ����Ҫ�����ǽ��� (i & k) == 0 �����򣬷�����
                bool ascending = ((i & k) == 0);

                // ȡ��Ҫ�Ƚϵ�Ԫ��
                UInt lhs = inOut[i];
                UInt rhs = inOut[partner];

                // ���� ascending �����Ƿ񽻻�
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
    // Execute Bitonic Sort (��������ѭ��)
    //-----------------------------------------
    // ׼�������С
    UInt threadsPerBlock = 256u;
    UInt blocks = (n + threadsPerBlock - 1u) / threadsPerBlock;
    auto threadSpec = alpaka::onHost::ThreadSpec{Vec1D{blocks}, Vec1D{threadsPerBlock}};

    // ˫��������ģ���� k�������д�С�����ڲ� j��������
    // k �� 2 ������ n
    for(UInt k = 2u; k <= n; k <<= 1u)
    {
        // j �� k/2 һ·���� 1
        for(UInt j = (k >> 1u); j > 0u; j >>= 1u)
        {
            // ��������ύ Kernel���� (k, j) ָ����ǰ�׶�
            queue.enqueue(computeExec, threadSpec, BitonicCompareSwapKernel{}, deviceBuf.getMdSpan(), n, k, j);
            alpaka::onHost::wait(queue);
        }
    }

    //-----------------------------------------
    // ���ؽ����������ʾ���޸�
    //-----------------------------------------
    // 1. �� Device ������ Host
    alpaka::onHost::memcpy(queue, hostBuf, deviceBuf);
    alpaka::onHost::wait(queue);

    // 2. �޸�ż��Ϊż��+1
    for(UInt i = 0; i < n; ++i)
    {
        if(hostBuf[i] % 2 == 0)
        {
            hostBuf[i] += 1;
        }
    }

    // 3. �ٴο��� Device
    alpaka::onHost::memcpy(queue, deviceBuf, hostBuf);
    alpaka::onHost::wait(queue);

    // 4. ���ս����һ�ο��� Host
    alpaka::onHost::memcpy(queue, hostBuf, deviceBuf);
    alpaka::onHost::wait(queue);

    // �����ս��д�� vector
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
    UInt size = 2048u; // ԭʼ��С
    UInt paddedSize = nextPowerOfTwo(size); // ���ϲ��� 2 ����
    std::vector<UInt> data(paddedSize, INF);

    // ������ǰ�� size ��Ԫ��
    std::srand(1234); // �̶��������
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
    // ���� example ���� (�����������õ� Alpaka API/Executor)
    return alpaka::executeForEach(
        [=](auto const& tag) { return example(tag); },
        alpaka::onHost::allExecutorsAndApis(alpaka::onHost::enabledApis));
}
