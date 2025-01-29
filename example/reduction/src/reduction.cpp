/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cstdio>
#include <random>

#include <chrono>
#include <iostream>

#include <cstdlib>
#include <ctime>

/** @file
 *
 * In the previous example we showed how to handle thread indices by hand to iterate over 1 and 3-dimensional data.
 * There are very seldom cases where you need this explict control over threads and blocks. Very often handling thread
 * indices by hand will result in performance issues at least on CPU devices.
 *
 * This example will show how you can iterate with frames, which can be seen as data index chunks without explicit
 * calculate thread indices by hand. The code is is easy to write and read and will mostly be faster on CPU and GPU
 * devices.
 */

// sum += A[]*B[];

struct ReductionKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto out, auto arraySize) const
    {
        // product() returns a scalar therefore we need the explicit Vec1D type
        Vec1D linearNumFrames = acc[alpaka::frame::count].product();
        auto frameExtent = acc[alpaka::frame::extent];
        auto frameDataExtent = linearNumFrames*frameExtent;
        Vec1D linearFrameExtent = frameExtent.product();
        auto traverseInFrame = alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{frameExtent});
        auto traverseOverFrames = alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::blocksInGrid, alpaka::IdxRange{alpaka::CVec<uint32_t, 0u>{}, frameDataExtent, linearFrameExtent});
        // Shared Mempry
        auto sumOnSharedMem = alpaka::onAcc::declareSharedMdArray<double>(acc, frameExtent);

        // Values of these addresses will be used later. 
        // Sync() is not required.
        for(auto [elemIdxInFrame] : traverseInFrame){
            sumOnSharedMem[elemIdxInFrame] = 0;
        }

        // init completed
        ///////////////////////////////////////////////////////////
        // For each frame in frames and for each thread in frame do
        // get the sum of a small piece in in1[] on the thread to the sharedMem[].

        for(auto frameIdx : traverseOverFrames){
            for(auto elemIdx : traverseInFrame){
                for(auto [i]: alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::WorkerGroup{frameIdx + elemIdx, frameDataExtent}, alpaka::IdxRange{arraySize})){
                    sumOnSharedMem[elemIdx] += in1[i];
                }
            }
        }

        alpaka::onAcc::syncBlockThreads(acc);

        // Copying data to sharedMem completed.
        ///////////////////////////////////////////////////////////
        // For each thread on acc do
        // sum up.

        for(auto [elemIdx] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{acc[alpaka::layer::thread].count(), frameExtent})){
            sumOnSharedMem[acc[alpaka::layer::thread].idx()] += sumOnSharedMem[elemIdx];
        }

        // Suming up on each thread completed.
        ///////////////////////////////////////////////////////////
        // 

        auto const [local_i] = acc[alpaka::layer::thread].idx();
        auto const [blockSize] = acc[alpaka::layer::thread].count();
        for(auto stride = blockSize /2; stride > 0; stride /=2){
            alpaka::onAcc::syncBlockThreads(acc);
            if(local_i < stride){
                sumOnSharedMem[local_i] += sumOnSharedMem[local_i+stride];
                //printf("[kernel] %f\n", sumOnSharedMem[local_i]);
                //sumOnSharedMem[local_i+stride] = 0;
            }
        }

        if(local_i == 0){
            out[acc[alpaka::layer::block].idx().x()] = sumOnSharedMem[local_i];
        }

        // 
        ///////////////////////////////////////////////////////////
        // 


    }
};

void testReductionKernel(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec, auto size)
{

    //std::cout << "[Host] " << alpaka::onHost::getName(host) << ", ";
    //std::cout << "[Device] " << alpaka::onHost::getName(device) << ", ";
    std::cout << alpaka::onHost::getName(device) << ", " << std::flush;

    // random number generator with a gaussian distribution
    //std::random_device rd{};
    //std::default_random_engine rand{rd()};
    std::default_random_engine rand{};
    std::normal_distribution<double> dist{(double)0.01, (double)1.};

    // buffer size
    //std::cout << "[Problem Size] " << size << ", ";
    std::cout << size << ", " << std::flush;

    // tolerance
    constexpr double epsilon = (double)0.0001;
    //constexpr float epsilon = 0.000001f*size;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = alpaka::onHost::alloc<double>(host, Vec1D{size});
    auto out_h = alpaka::onHost::allocMirror(host, in1_h);

    // fill the input buffers with random data, and the output buffer with zeros
    for(auto i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        out_h[i] = 0.;
    }

    // run the test the given device
    alpaka::onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = alpaka::onHost::allocMirror(device, in1_h);
    auto out_d = alpaka::onHost::allocMirror(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::onHost::memcpy(queue, in1_d, in1_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    // launch the 1-dimensional kernel
    constexpr auto frameExtent = 256;
    //constexpr auto frameExtent = 1024;
    auto numFrames = Vec1D{size} / frameExtent /8;
    // The kernel assumes that the problem size is a multiple of the frame size.
    //assert((numFrames * frameExtent).x() *8 == size);

    auto frameSpec = alpaka::onHost::FrameSpec{numFrames, alpaka::CVec<uint32_t, frameExtent>{}};

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    //std::cout << "Grid of " << frameSpec << ", ";

    alpaka::onHost::wait(queue); 
    auto beginT = std::chrono::high_resolution_clock::now();
    queue
        .enqueue(computeExec, frameSpec, ReductionKernel{}, in1_d.getMdSpan(), out_d.getMdSpan(), Vec1D(size));
    alpaka::onHost::wait(queue);
    auto endT = std::chrono::high_resolution_clock::now();
    //std::cout << "[T Kernel Exec] " << std::chrono::duration<double>(endT - beginT).count() << 's' << ", ";
    std::cout << std::chrono::duration<double>(endT - beginT).count() << ", " << std::flush;


    alpaka::onHost::wait(queue); 
    beginT = std::chrono::high_resolution_clock::now();
    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);
    // wait for all the operations to complete
    alpaka::onHost::wait(queue);
    endT = std::chrono::high_resolution_clock::now();
    //std::cout << "[T HtoD Copy] " << std::chrono::duration<double>(endT - beginT).count() << 's' << ", ";

    alpaka::onHost::wait(queue); 
    beginT = std::chrono::high_resolution_clock::now();
    auto finalSum = std::accumulate(
        &out_h[0],
        &out_h[size-1],
        double(0));
    endT = std::chrono::high_resolution_clock::now();
    //std::cout << "[T Partial Sum Accumulation] " << std::chrono::duration<double>(endT - beginT).count() << 's' << ", ";

    double sum = 0;
    // check the results
    for(auto i = 0; i < size; ++i)
    {
        //if (i < 5) std::cout << "[num] " << in1_h[i] << std::endl;
        sum += in1_h[i];     
    }
    //std::cout << "acc output: " << finalSum << " host answer: " << sum << std::endl;
    //printf("[Device Output] %f [Host Output] %f, ",finalSum, sum);
    assert(pow(finalSum - sum,2) < pow(epsilon,2));
    //std::cout << "[Results] " << "success\n";
    std::cout << "success\n" << std::flush;
}

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

    // use the first device
    alpaka::onHost::Device device = platform.makeDevice(0);

    uint32_t size = 1024 * 1024;
    for(int fac = 10; fac < 11; fac++){
        size = 1024*1024*pow(2,fac);
        testReductionKernel(host, device, computeExec, size);
    }

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    //std::srand(std::time(0)); // set time as random seed; rand() after this line will automatically use the same seed
    std::srand(12345);
    std::cout << "Device, Problem Size, T Kernel Exec (s), Results" << std::endl;
    return executeForEach(
        [=](auto const& tag) { return example(tag); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
}
