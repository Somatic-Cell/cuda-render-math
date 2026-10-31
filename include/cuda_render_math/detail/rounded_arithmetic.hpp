#pragma once 

#include <cuda_render_math/config.hpp>
#include <cuda_render_math/scalar.hpp>

#include <cmath>

namespace cuda_render_math::detail {

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
add_rn(T a, T b) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) return ::__fadd_rn(a, b);
    else return ::__dadd_rn(a, b);
#else
    return a + b;
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
sub_rn(T a, T b) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) return ::__fsub_rn(a, b);
    else return ::__dsub_rn(a, b);
#else
    return a - b;
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
mul_rn(T a, T b) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) return ::__fmul_rn(a, b);
    else return ::__dmul_rn(a, b);
#else
    return a * b;
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
fma_rn(T a, T b, T c) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) return ::__fmaf_rn(a, b, c);
    else return ::__fma_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

} // cuda_render_math::detail