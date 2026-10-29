#pragma once

#include <cuda_render_math/config.hpp>
#include <cuda_render_math/scalar.hpp>

#include <cmath>
#include <limits>

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
#if defined (__CUDA_ARCH__)
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

namespace detail {

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
quiet_nan() noexcept {
#if defined (__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>)
    {
        return ::nanf("");
    } 
    else
    {
        return nan("");
    }
#else
    return std::numeric_limits<T>::quiet_NaN();    
#endif
}

}// namespace detail

// std::min / max は NaN が含まれている場合それ以外の数を返すが，
// このライブラリは NaN を伝播させる (デバッグしやすくするため)
// NaN はデータの欠損ではなく，計算結果の破綻と捉えたい
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr T
(min)(T a, T b) noexcept {
    // NaN が含まれている場合は NaN を返す
    if (a != a)     // IEEE 754 より，NaN == NaN は false
    {
        return a;   
    }  
    return (b != b || b < a) ? b : a;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr T
(max)(T a, T b) noexcept {
    // NaN が含まれている場合は NaN を返す
    if (a != a)     // IEEE 754 より，NaN == NaN は false
    {
        return a;   
    }  
    return (b != b || a < b) ? b : a;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
clamp(T value, T lower, T upper) noexcept {
    // NaN を伝播させる
    if (value != value) return value;
    if (lower != lower) return lower;
    if (upper != upper) return upper;
    
    // 前提が破綻している場合は aNaN を返す
    if (upper < lower) return detail::quiet_nan<T>();
    return value < lower ? lower : (upper < value ? upper : value);
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
saturate(T value) noexcept {
    return cuda_render_math::clamp(value, T(0), T(1));
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
floor(T x) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) return ::floorf(x);
    else return ::floor(x);
#else
    return std::floor(x);
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
ceil(T x) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) return ::ceilf(x);
    else return ::ceil(x);
#else
    return std::ceil(x);
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
fmod(T x, T y) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) return ::fmodf(x, y);
    else return ::fmod(x, y);
#else
    return std::fmod(x, y);
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
fract(T x) noexcept {
    return x - cuda_render_math::floor(x);
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
lerp(T a, T b, T t) noexcept {
    // 無効な入力の場合， qNaN を返す
    if(!cuda_render_math::isfinite(a) || !cuda_render_math::isfinite(b) || !cuda_render_math::isfinite(t))
    {
        return detail::quiet_nan<T>();
    }
    // a, b が逆符号: 
    const T result = ((a < T(0)) != (b < T(0))) 
        ? (T(1) - t) * a + t * b 
        : a + t * (b - a);
    if(t > T(0) && t < T(1)) {
        return cuda_render_math::clamp(result, (cuda_render_math::min)(a, b), (cuda_render_math::max)(a, b));
    }
    return result;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
smoothstep(T lower, T upper, T x) noexcept {
    // 無効な入力の場合， qNaN を返す
    if(!cuda_render_math::isfinite(lower) || !cuda_render_math::isfinite(upper) || !(lower < upper))
    {
        return detail::quiet_nan<T>();
    }
    if(x != x) return x;

    // 範囲外の処理
    if(x <= lower) return T(0);
    if(x >= upper) return T(1);

    const T width = upper - lower;
    const T t = cuda_render_math::isfinite(width) 
        ? (x - lower) / width
        : (x * T(0.5) - lower * T(0.5)) / (upper * T(0.5) - lower * T(0.5));

    const T u = cuda_render_math::saturate(t); 
    return cuda_render_math::saturate(u * u * (T(3) - T(2) * u));

}


} // namespace cuda_render_math