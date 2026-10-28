#pragma once

#include <concepts>

namespace cuda_render_math {

// 型には float / double のみを許す
template<typename T>
concept real_scalar = 
    std::same_as<T, float> || 
    std::same_as<T, double>;

} // namespace cuda_render_math