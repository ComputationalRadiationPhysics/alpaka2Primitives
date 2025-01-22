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


constexpr uint32_t blockThreadExtentMain = 1024; // 每个块中的线程数


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
        // Declare shared memory in the same way as in DotKernel
        auto tbSum = onAcc::declareSharedMdArray<T>(acc, CVec<uint32_t, blockThreadExtentMain>{});
#if 1
        // Use the same 3-level structure as in DotKernel (frames, threadsInBlock, ...)
        auto numFrames = acc[frame::count];
        // auto frameExtent = Vec{acc[frame::extent]};
        auto frameExtent = acc[frame::extent];

        // Traverse within a frame using thread indices
        auto traverseInFrame = onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtent});

        // Initialize shared memory
        for(auto [elemIdxInFrame] : traverseInFrame)
        {
            tbSum[elemIdxInFrame] = T{0};
        }


        //auto const frameDataExtent = numFrames * frameExtent;
         auto const frameDataExtent = frameExtent * numFrames;

        //auto frameDataExtent = static_cast<decltype(frameExtent)>(numFrames) * frameExtent;



         /////////////////////////////////////////////////////////////////////////////////////////////////////////

        auto traverseOverFrames = onAcc::makeIdxMap(
            acc,
            onAcc::worker::blocksInGrid,
            IdxRange{
                alpaka::Vec<uint32_t, 1u>{0},
                alpaka::Vec<uint32_t, 1u>{frameDataExtent},
                alpaka::Vec<uint32_t, 1u>{frameExtent}});



        // Add x[i] to the shared memory tbSum
        for(auto frameIdx : traverseOverFrames)
        {
            for(auto elemIdxInFrame : traverseInFrame)
            {
                for(auto [i] : onAcc::makeIdxMap(
                        acc,
                        onAcc::WorkerGroup{
                            frameIdx + elemIdxInFrame,
                            frameDataExtent},
                        IdxRange{arraySize}))
                {
                    tbSum[elemIdxInFrame] += x[i];
                }

            }
        }
        // Similar to DotKernel, synchronize block threads to ensure all writes are completed
        onAcc::syncBlockThreads(acc);


         //////////////////////////////////////////////////////////



        // Reduce all elements in tbSum[] to tbSum[acc[layer::thread].idx()]
        for(auto [elemIdxInFrame] :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{acc[layer::thread].count(), frameExtent}))
        {
            tbSum[acc[layer::thread].idx()] += tbSum[elemIdxInFrame];
        }

#else
        // This #else branch is similar to DotKernel, but performs a single array summation.
        // It's shorter but may have less floating-point precision compared to the version above.
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

        // （tree-based reduction）
        auto const [local_i] = acc[layer::thread].idx();
        auto const [blockSize] = acc[layer::thread].count();
        for(auto offset = blockSize / 2; offset > 0; offset /= 2)
        {
            onAcc::syncBlockThreads(acc);
            if(local_i < offset)
                tbSum[local_i] += tbSum[local_i + offset];
        }

        // 
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
    using IdxVec = Vec<std::uint32_t, 1u>;

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

    Vec<uint32_t, 1u> chunkSize = 256u;
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
        // Configure the reduction work division
        // Vec<size_t, 1u> reductionChunkSize = blockThreadExtentMain; // Number of threads per block
        auto reductionChunkSize = alpaka::Vec<uint32_t, 1u>{blockThreadExtentMain};

        auto reductionDataBlocking = onHost::FrameSpec{core::divCeil(extent, reductionChunkSize), reductionChunkSize};

        // Allocate a buffer to store the reduction results
        // using IdxVec = Vec<std::size_t, 1u>;
        // auto numFramesVec = IdxVec(reductionDataBlocking.m_numFrames); // Use the same type as `extent`
        std::uint32_t numFrames = reductionDataBlocking.m_numFrames[0];
        auto numFramesVec = alpaka::Vec<std::uint32_t, 1u>{numFrames};
        auto bufReductionOut = onHost::alloc<Data>(devAcc, numFramesVec);

        // Instantiate the SumKernel
        SumKernel sumKernel;

        // auto const taskReduction = KernelBundle{sumKernel, bufAccC.getMdSpan(), bufReductionOut.getMdSpan(),
        // extent};
        auto const taskReduction = KernelBundle{
            sumKernel,
            bufAccC.getMdSpan().data(), // Input data pointer
            bufReductionOut.getMdSpan().data(), // Output data pointer
            extent // Array size
        };

        // Execute the reduction kernel
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        onHost::enqueue(queue, exec, reductionDataBlocking, taskReduction); // Invoke the reduction kernel
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for reduction kernel: " << std::chrono::duration<double>(endT - beginT).count() << "s"
                  << std::endl;

        // Copy partial sums from device back to host and finalize reduction
        auto bufHostReductionOut = onHost::alloc<Data>(devHost, numFramesVec);
        onHost::memcpy(queue, bufHostReductionOut, bufReductionOut);
        onHost::wait(queue);

        // Finalize reduction on the host
        // Get the MdSpan object
        auto mdSpan = bufHostReductionOut.getMdSpan();

        // Get the size of the first dimension
        auto extent0 = mdSpan.getExtents()[0]; // Access the first dimension using subscript operator

        // Initialize the total sum
        Data finalSum = 0;

        // Iterate over all elements of MdSpan
        for(std::size_t i = 0; i < extent0; ++i)
        {
            finalSum += mdSpan[i]; // Use operator[] for 1D indexing
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
