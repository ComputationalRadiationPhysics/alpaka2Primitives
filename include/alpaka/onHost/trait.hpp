/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/KernelBundle.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/tag.hpp"

#include <type_traits>

namespace alpaka::onHost
{
    namespace trait
    {
        struct IsPlatformAvailable
        {
            template<alpaka::concepts::Api T_Api>
            struct Op : std::false_type
            {
            };
        };

        struct IsMappingSupportedBy
        {
            template<typename T_Mapping, typename T_Device>
            struct Op : std::false_type
            {
            };
        };

        template<typename T_Mapping, concepts::DeviceHandle T_DeviceHandle>
        struct IsMappingSupportedBy::Op<T_Mapping, T_DeviceHandle>
            : IsMappingSupportedBy::Op<T_Mapping, typename T_DeviceHandle::element_type>
        {
        };

    } // namespace trait

    consteval bool isPlatformAvaiable(alpaka::concepts::Api auto api)
    {
        return trait::IsPlatformAvailable::Op<std::decay_t<decltype(api)>>::value;
    }

    constexpr auto supportedMappings(concepts::DeviceHandle auto deviceHandle)
    {
        return meta::filter(
            [&](auto executor) constexpr
            {
                return trait::IsMappingSupportedBy::
                    Op<std::decay_t<decltype(executor)>, std::decay_t<decltype(deviceHandle)>>::value;
            },
            exec::availableMappings);
    }
} // namespace alpaka::onHost
