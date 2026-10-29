#pragma once

#include <concepts>

namespace cuda_render_math {

// 実数を表現する型には float / double のみを許す
template<typename T>
concept real_scalar = 
    std::same_as<T, float> || 
    std::same_as<T, double>;

// 整数を表現する型
template<typename T>
concept integer_scalar =
    std::same_as<T, std::int32_t> ||
    std::same_as<T, std::uint32_t>;

template <typename T>
concept supported_arithmetic = 
    real_scalar<T> || 
    integer_scalar<T>;

} // namespace cuda_render_math