#pragma once 

#include <cuda_render_math/scalar_math.hpp>

namespace cuda_render_math {

// 入力を [+0, Inf] に制限
// NaN は NaN で返す
// - Inf は　NaN で返す
// + Inf は + Inf で返す
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
sqrt_clamped(T value) noexcept {
    
    // NaN チェック (NaN で返す)
    if(value != value) return value;
    
    // sqrt の中身は非負である必要があるのでチェック
    // -Inf は qNaN として返す
    if (value < T(0)) {
        return cuda_render_math::isfinite(value) ? T(0) : detail::quiet_nan<T>();
    }

    return cuda_render_math::sqrt(value);
}

// 入力を [-1, 1] にクランプする
// Inf か NaN は NaN で返す
// radian で返す
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
asin_clamped(T value) noexcept {
    // NaN チェック (NaN で返す)
    if(value != value) return value;
    
    // +-Inf は qNaN として返す
    if (!cuda_render_math::isfinite(value)) {
        return  detail::quiet_nan<T>();
    }

    const T bounded = cuda_render_math::clamp(value, T(-1), T(1));

#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) return ::asinf(bounded);
    else return ::asin(bounded);
#else
    return std::asin(bounded);
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
acos_clamped(T value) noexcept {
    // NaN チェック (NaN で返す)
    if(value != value) return value;
    
    // +-Inf は qNaN として返す
    if (!cuda_render_math::isfinite(value)) {
        return  detail::quiet_nan<T>();
    }

    const T bounded = cuda_render_math::clamp(value, T(-1), T(1));

#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) return ::acosf(bounded);
    else return ::acos(bounded);
#else
    return std::acos(bounded);
#endif
}

// エラーフリー変換 (EFT)
// 外積の計算などに登場する a*b - c*d を EFT する
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
difference_of_products(T a, T b, T c, T d) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) {
        const T rounded_product = ::__fmul_rn(c, d);
        const T product_error = ::__fmaf_rn(-c, d, rounded_product);
        const T difference = ::__fmaf_rn(a, b, -rounded_product);
        return ::__fadd_rn(difference, product_error);
    } else {
        const T rounded_product = ::__dmul_rn(c, d);
        const T product_error = ::__fma_rn(-c, d, rounded_product);
        const T difference = ::__fma_rn(a, b, -rounded_product);
        return ::__dadd_rn(difference, product_error);
    }
#else
    const T rounded_product = c * d;
    const T product_error = std::fma(-c, d, rounded_product);
    const T difference = std::fma(a, b, -rounded_product);
    
    return difference + product_error;
#endif
}

}