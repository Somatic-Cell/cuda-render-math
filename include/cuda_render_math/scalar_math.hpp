#pragma once

#include <cuda_render_math/config.hpp>
#include <cuda_render_math/scalar.hpp>

#include <cmath>

namespace cuda_render_math {

// CPU での計算では cmath, NVCC の device pass では CUDA Math API を使う
// NaN / Inf / 符号付きゼロの扱いは IEEE 754 を基本とする
// 入力のクランプや NaN を 0 にフォールバックしたりはしない

// 絶対値
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
abs(T x) noexcept {
#if defined (__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) {
        return ::fabsf(x);
    } else {
        return ::fabs(x);
    }
#else
    return std::fabs(x);
#endif
}

//
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
sqrt(T x) noexcept {
#if defined (__CUDA__ARCH__)
    if constexpr (std::same_as<T, float>) {
        return ::sqrtf(x);
    } else {
        return ::sqrt(x);
    }
#else
    return std::sqrt(x);
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
fma(T x, T y, T z) noexcept {
#if defined (__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) {
        return ::fmaf(x, y, z);
    } else {
        return ::fma(x, y, z);
    }
#else
    return std::fma(x, y, z);
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
copysign(T magnitude, T sign_source) noexcept {
#if defined (__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) {
        return ::copysignf(magnitude, sign_source);
    } else {
        return ::copysign(magnitude, sign_source);
    }
#else
    return std::copysign(magnitude, sign_source);;
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
isfinite(T x) noexcept {
#if defined (__CUDA_ARCH__)
    return ::isfinite(x) != 0;
#else
    return std::isfinite(x);
#endif
}

} // namespace cuda_render_math