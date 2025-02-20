/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/api/cpu/exec/OmpBlocks.hpp"
#include "alpaka/api/cpu/exec/OmpThreads.hpp"
#include "alpaka/api/cpu/exec/Serial.hpp"
#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/internal.hpp"

#include <cstdint>
#include <cstring>
#include <future>
#include <sstream>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_Device>
        struct Queue
        {
        public:
            Queue(concepts::DeviceHandle auto device, uint32_t const idx) : m_device(std::move(device)), m_idx(idx)
            {
            }

            ~Queue()
            {
                internal::Wait::wait(*this);
            }

            Queue(Queue const&) = delete;
            Queue(Queue&&) = delete;

            bool operator==(Queue const& other) const
            {
                return m_idx == other.m_idx && m_device == other.m_device;
            }

            bool operator!=(Queue const& other) const
            {
                return !(*this == other);
            }

        private:
            void _()
            {
                static_assert(concepts::Queue<Queue>);
            }

            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            core::CallbackThread m_workerThread;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("cpu::Queue id=") + std::to_string(m_idx);
            }

            friend struct internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return 0;
            }

            friend struct internal::Enqueue;

            template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
            void enqueue(
                auto const executor,
                ThreadSpec<T_NumBlocks, T_NumThreads> const& threadBlocking,
                auto kernelBundle)
            {
                m_workerThread.submit(
                    [=, kernel = std::move(kernelBundle)]()
                    {
                        onAcc::Acc acc = makeAcc(executor, threadBlocking);
                        acc(std::move(kernel));
                    });
            }

            template<typename T_Mapping, alpaka::concepts::Vector T_NumFrames, alpaka::concepts::Vector T_FrameExtent>
            void enqueue(T_Mapping const executor, FrameSpec<T_NumFrames, T_FrameExtent> frameSpec, auto kernelBundle)
            {
                auto threadBlocking = internal::adjustThreadSpec(*m_device.get(), executor, frameSpec, kernelBundle);
                m_workerThread.submit(
                    [=, kernel = std::move(kernelBundle)]()
                    {
                        auto moreLayer = Dict{
                            DictEntry(frame::count, frameSpec.m_numFrames),
                            DictEntry(frame::extent, frameSpec.m_frameExtent),
                            DictEntry(object::api, api::cpu),
                            DictEntry(object::exec, executor)};
                        onAcc::Acc acc = makeAcc(executor, threadBlocking);
                        acc(std::move(kernel), moreLayer);
                    });
            }

            void enqueue(auto task)
            {
                m_workerThread.submit([task]() { task(); });
            }

            friend struct internal::Wait;
            friend struct internal::Memcpy;
            friend struct internal::Memset;
            friend struct alpaka::internal::GetApi;
        };
    } // namespace cpu

    namespace internal
    {
        template<typename T_Device>
        struct Wait::Op<cpu::Queue<T_Device>>
        {
            void operator()(cpu::Queue<T_Device>& queue) const
            {
                std::promise<void> p;
                auto f = p.get_future();
                internal::enqueue(queue, [&p]() { p.set_value(); });

                f.wait();
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<cpu::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(cpu::Queue<T_Device>& queue, T_Dest dest, T_Source const source, T_Extents const& extents)
                const
            {
                static_assert(std::is_same_v<ALPAKA_TYPEOF(dest), ALPAKA_TYPEOF(source)>);
                constexpr auto dim = dest.dim();
                if constexpr(dim == 1u)
                {
                    internal::enqueue(
                        queue,
                        [extents, l_dest = std::move(dest), l_source = std::move(source)]()
                        {
                            std::memcpy(
                                alpaka::onHost::data(l_dest),
                                alpaka::onHost::data(l_source),
                                extents.x() * sizeof(typename T_Dest::type));
                        });
                }

                else
                {
                    internal::enqueue(
                        queue,
                        [extents, l_dest = std::move(dest), l_source = std::move(source)]()
                        {
                            auto const dstExtentWithoutColumn = extents.eraseBack();
                            if(static_cast<std::size_t>(extents.product()) != 0u)
                            {
                                auto const destPitchBytesWithoutColumn = l_dest.getPitches().eraseBack();
                                auto* destPtr = data(l_dest);
                                auto const sourcePitchBytesWithoutColumn = l_source.getPitches().eraseBack();
                                auto* sourcePtr = data(l_source);

                                meta::ndLoopIncIdx(
                                    dstExtentWithoutColumn,
                                    [&](auto const& idx)
                                    {
                                        std::memcpy(
                                            reinterpret_cast<std::uint8_t*>(destPtr)
                                                + (idx * destPitchBytesWithoutColumn).sum(),
                                            reinterpret_cast<std::uint8_t*>(sourcePtr)
                                                + (idx * sourcePitchBytesWithoutColumn).sum(),
                                            static_cast<size_t>(extents.back()) * sizeof(typename T_Dest::type));
                                    });
                            }
                        });
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Extents>
        struct Memset::Op<cpu::Queue<T_Device>, T_Dest, T_Extents>
        {
            void operator()(cpu::Queue<T_Device>& queue, T_Dest dest, uint8_t byteValue, T_Extents const& extents)
                const
            {
                constexpr auto dim = dest.dim();
                if constexpr(dim == 1u)
                {
                    internal::enqueue(
                        queue,
                        [extents, l_dest = std::move(dest), byteValue]() {
                            std::memset(
                                alpaka::onHost::data(l_dest),
                                byteValue,
                                extents.x() * sizeof(typename T_Dest::type));
                        });
                }

                else
                {
                    internal::enqueue(
                        queue,
                        [extents, l_dest = std::move(dest), byteValue]()
                        {
                            auto const dstExtentWithoutColumn = extents.eraseBack();
                            if(static_cast<std::size_t>(extents.product()) != 0u)
                            {
                                auto const destPitchBytesWithoutColumn = l_dest.getPitches().eraseBack();
                                auto* destPtr = data(l_dest);

                                meta::ndLoopIncIdx(
                                    dstExtentWithoutColumn,
                                    [&](auto const& idx)
                                    {
                                        std::memset(
                                            reinterpret_cast<std::uint8_t*>(destPtr)
                                                + (idx * destPitchBytesWithoutColumn).sum(),
                                            byteValue,
                                            static_cast<size_t>(extents.back()) * sizeof(typename T_Dest::type));
                                    });
                            }
                        });
                }
            }
        };


    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::cpu::Queue<T_Device>>
    {
        decltype(auto) operator()(auto&& queue) const
        {
            return onHost::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal
