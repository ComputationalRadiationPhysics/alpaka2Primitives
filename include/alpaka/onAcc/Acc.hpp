/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/tag.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::onAcc
{
    template<typename T_Storage>
    struct Acc : T_Storage
    {
        constexpr Acc(T_Storage const& storage) : T_Storage{storage}
        {
        }

        constexpr Acc(Acc const&) = delete;
        constexpr Acc(Acc const&&) = delete;
        constexpr Acc& operator=(Acc const&) = delete;

        template<typename T>
        constexpr decltype(auto) declareSharedVar() const
        {
            return (*this)[layer::shared].template allocVar<T>();
        }

        constexpr void syncBlockThreads() const
        {
            (*this)[action::sync]();
        }

        consteval bool hasKey(auto key) const
        {
            return hasTag(static_cast<T_Storage>(*this));
        }
    };

} // namespace alpaka::onAcc
