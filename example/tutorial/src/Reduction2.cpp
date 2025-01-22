/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Luca Ferragina,
 *                Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <typeinfo>

using namespace alpaka;


constexpr size_t blockThreadExtentMain = 1024; // 每个块中的线程数


#include <type_traits>
#include <typeinfo>

template<typename T>
constexpr char const* getTypeName()
{
    return typeid(T).name();
}




// Reduction Kernel
struct SumKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param x Pointer for the vector to be reduced
    //! \param sum Pointer for result vector consisting sums of blocks
    //! \param arraySize the size of the array
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* x, T* sum, auto arraySize) const
    {
        // 与 DotKernel 相同的共享内存声明方式
        auto tbSum = onAcc::declareSharedMdArray<T>(acc, CVec<uint32_t, blockThreadExtentMain>{});
#if 1
        // 采用与 DotKernel 相同的 3-level 结构 (frames, threadsInBlock, ...)
        auto numFrames = Vec{acc[frame::count]};
        //auto frameExtent = Vec{acc[frame::extent]};
        auto frameExtent = alpaka::Vec<size_t, 1>{acc[frame::extent]};


        auto traverseInFrame = onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtent});
        // 初始化共享内存
        for(auto [elemIdxInFrame] : traverseInFrame)
        {
            tbSum[elemIdxInFrame] = T{0};
        }

        //auto const frameDataExtent = numFrames * frameExtent;
         auto const frameDataExtent = frameExtent * alpaka::Vec<size_t, 1>{numFrames};

        //auto frameDataExtent = static_cast<decltype(frameExtent)>(numFrames) * frameExtent;



         /////////////////////////////////////////////////////////////////////////////////////////////////////////

        //auto traverseOverFrames = onAcc::makeIdxMap(
        //    acc,
        //    onAcc::worker::blocksInGrid,
        //    IdxRange{
        //        alpaka::CVec<uint32_t, 1u>{0},
        //        alpaka::CVec<uint32_t, 1u>{frameDataExtent},
        //        alpaka::CVec<uint32_t, 1u>{frameExtent}});



        //




        //// 将 x[i] 加到共享内存 tbSum
        //for(auto frameIdx : traverseOverFrames)
        //{
        //    for(auto elemIdxInFrame : traverseInFrame)
        //    {
        //        for(auto [i] : onAcc::makeIdxMap(
        //                acc,
        //                onAcc::WorkerGroup{
        //                    alpaka::Vec<long long unsigned int, 1>{frameIdx} + elemIdxInFrame,//强制类型转换,避免match错误
        //                    frameDataExtent},
        //                IdxRange{arraySize}))
        //        {
        //            tbSum[elemIdxInFrame] += x[i];
        //        }

        //    }
        //}
        //// 与 DotKernel 一样，需要同步块内线程以确保所有写入完成
        //onAcc::syncBlockThreads(acc);


         //////////////////////////////////////////////////////////



         // 替换 IdxRange 的逻辑
         auto begin = alpaka::CVec<uint32_t, 0u>{};
         auto end = frameDataExtent; // 替代范围的结束
         auto step = frameExtent; // 步长

         // 创建 traverseOverFrames 数据结构
         std::vector<std::size_t> traverseOverFrames;

         // 模拟 IdxRange 的逻辑手动生成范围
         for(std::size_t frameIdx = 0; frameIdx < end[0]; frameIdx += step[0])
         {
             traverseOverFrames.push_back(frameIdx);
         }

         // 使用 traverseOverFrames 的结果
         for(auto frameIdx : traverseOverFrames)
         {
             for(auto elemIdxInFrame : traverseInFrame)
             {
                 for(auto [i] : onAcc::makeIdxMap(
                         acc,
                         onAcc::WorkerGroup{frameIdx + elemIdxInFrame, frameDataExtent},
                         IdxRange{arraySize}))
                 {
                     tbSum[elemIdxInFrame] += x[i];
                 }
             }
         }

         // 与 DotKernel 一样，需要同步块内线程以确保所有写入完成
         onAcc::syncBlockThreads(acc);















        // 将 tbSum[] 中所有元素归约到 tbSum[acc[layer::thread].idx()]
        for(auto [elemIdxInFrame] :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{acc[layer::thread].count(), frameExtent}))
        {
            tbSum[acc[layer::thread].idx()] += tbSum[elemIdxInFrame];
        }

#else
        // 这个 #else 分支与 DotKernel 中相同，只是改成单数组相加
        // 更短但可能在浮点精度上不如上面的版本
        auto threadSum = T{0};
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{arraySize}))
        {
            threadSum += x[i];
        }
        for(auto [local_i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::threadsInBlock))
        {
            tbSum[local_i] = threadSum;
        }
#endif

        // 块内最终折半归约（tree-based reduction）
        auto const [local_i] = acc[layer::thread].idx();
        auto const [blockSize] = acc[layer::thread].count();
        for(auto offset = blockSize / 2; offset > 0; offset /= 2)
        {
            onAcc::syncBlockThreads(acc);
            if(local_i < offset)
                tbSum[local_i] += tbSum[local_i + offset];
        }

        // 写出本块的归约结果
        if(local_i == 0)
            sum[acc[layer::block].idx().x()] = tbSum[local_i];
    }
};


//! A vector addition kernel.
class VectorAddKernel
{
public:
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    ALPAKA_FN_ACC auto operator()(auto const& acc, auto const A, auto const B, auto C, auto const& numElements) const
        -> void
    {
        using namespace alpaka;
        static_assert(ALPAKA_TYPEOF(numElements)::dim() == 1, "The VectorAddKernel expects 1-dimensional indices!");

        // The uniformElements range for loop takes care automatically of the blocks, threads and elements in the
        // kernel launch grid.
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numElements}))
        {
            C[i] = A[i] + B[i];
        }
    }
};



// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename T_Cfg>
auto example(T_Cfg const& cfg) -> int
{
    using IdxVec = Vec<std::size_t, 1u>;

    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for " << api.getName() << std::endl;

    // Select a device
    onHost::Platform platform = onHost::makePlatform(api);
    onHost::Device devAcc = platform.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Define the work division
    IdxVec const extent(123456);

    // Define the buffer element type
    using Data = std::uint32_t;

    // Get the host device for allocating memory on the host.
    onHost::Platform platformHost = onHost::makePlatform(api::cpu);
    onHost::Device devHost = platformHost.makeDevice(0);

    // Allocate 3 host memory buffers
    auto bufHostA = onHost::alloc<Data>(devHost, extent);
    auto bufHostB = onHost::allocMirror(devHost, bufHostA);
    auto bufHostC = onHost::allocMirror(devHost, bufHostA);

    // C++14 random generator for uniformly distributed numbers in {1,..,42}
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<Data> dist(1, 42);

    for(auto i(0); i < extent; ++i)
    {
        bufHostA.getMdSpan()[i] = dist(eng);
        bufHostB.getMdSpan()[i] = dist(eng);
        bufHostC.getMdSpan()[i] = 0;
    }

    // Allocate 3 buffers on the accelerator
    auto bufAccA = onHost::allocMirror(devAcc, bufHostA);
    auto bufAccB = onHost::allocMirror(devAcc, bufHostB);
    auto bufAccC = onHost::allocMirror(devAcc, bufHostC);

    // Copy Host -> Acc
    onHost::memcpy(queue, bufAccA, bufHostA);
    onHost::memcpy(queue, bufAccB, bufHostB);
    onHost::memcpy(queue, bufAccC, bufHostC);

    // Instantiate the kernel function object
    VectorAddKernel kernel;
    auto const taskKernel
        = KernelBundle{kernel, bufAccA.getMdSpan(), bufAccB.getMdSpan(), bufAccC.getMdSpan(), extent};

    Vec<size_t, 1u> chunkSize = 256u;
    auto dataBlocking = onHost::FrameSpec{core::divCeil(extent, chunkSize), chunkSize};

    // Enqueue the kernel execution task
    {
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        onHost::enqueue(queue, exec, dataBlocking, taskKernel);
        onHost::wait(queue); // wait in case we are using an asynchronous queue to time actual kernel runtime
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    //////////////////////////REDUCTION///////////////////////////////////////////////////////////////
    ////////////////////////REDUCTION///////////////////////////////////////////////////////////////
    // Enqueue the reduction kernel task (SumKernel)
    {
        // 配置 Reduction 工作划分
        //Vec<size_t, 1u> reductionChunkSize = blockThreadExtentMain; // 每块线程数
        Vec<size_t, 1u> reductionChunkSize = alpaka::Vec<size_t, 1u>{blockThreadExtentMain};

        auto reductionDataBlocking = onHost::FrameSpec{core::divCeil(extent, reductionChunkSize), reductionChunkSize};
        
        // 分配用于存储归约结果的缓冲区
        //using IdxVec = Vec<std::size_t, 1u>;
        //auto numFramesVec = IdxVec(reductionDataBlocking.m_numFrames); // 使用与 `extent` 相同的类型
        std::size_t numFrames = reductionDataBlocking.m_numFrames[0];
        auto numFramesVec = alpaka::Vec<std::size_t, 1u>{numFrames};
        auto bufReductionOut = onHost::alloc<Data>(devAcc, numFramesVec);

        // 实例化 SumKernel
        SumKernel sumKernel;
        //auto const taskReduction = KernelBundle{sumKernel, bufAccC.getMdSpan(), bufReductionOut.getMdSpan(), extent};
        auto const taskReduction = KernelBundle{
            sumKernel,
            bufAccC.getMdSpan().data(), // 输入数据指针
            bufReductionOut.getMdSpan().data(), // 输出数据指针
            extent // 数组大小
        };

        // 执行归约内核
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        onHost::enqueue(queue, exec, reductionDataBlocking, taskReduction); // 调用归约内核
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for reduction kernel: " << std::chrono::duration<double>(endT - beginT).count() << "s"
                  << std::endl;

        // 从设备拷贝部分和回主机并归约
        auto bufHostReductionOut = onHost::alloc<Data>(devHost, numFramesVec);
        onHost::memcpy(queue, bufHostReductionOut, bufReductionOut);
        onHost::wait(queue);

        // 在 Host 上完成最终归约
        // 获取 MdSpan 对象
        auto mdSpan = bufHostReductionOut.getMdSpan();

        // 获取第 0 维的大小
        auto extent0 = mdSpan.getExtents()[0]; // 使用下标访问 Vec 的第 0 维

        // 初始化总和
        Data finalSum = 0;

        // 遍历 MdSpan 的所有元素
        for(std::size_t i = 0; i < extent0; ++i)
        {
            finalSum += mdSpan[i]; // 使用 operator[] 进行一维索引访问
        }



        std::cout << "Final sum after reduction: " << finalSum << std::endl;




    }
    ////////////////////////REDUCTION////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////REDUCTION////////////////////////////////////////////////////////////////////////////////////




    // Copy back the result
    {
        auto beginT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHostC, bufAccC);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for HtoD copy: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;
    for(auto i(0u); i < extent; ++i)
    {
        Data const& val(bufHostC.getMdSpan()[i]);
        Data const correctResult(bufHostA.getMdSpan()[i] + bufHostB.getMdSpan()[i]);
        if(val != correctResult)
        {
            if(falseResults < MAX_PRINT_FALSE_RESULTS)
                std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            ++falseResults;
        }
    }

    if(falseResults == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << MAX_PRINT_FALSE_RESULTS
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach(
        [=](auto const& tag) { return example(tag); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
}
