#pragma once 

#include <cuda_render_math/scalar_numerics.hpp>
#include <cuda_render_math/vector.hpp>

namespace cuda_render_math {


// -------------------
// 2次元ベクトル
// -------------------

// ベクトルの要素が有効かどうかをチェック
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
is_all_finite(vec2<T> value) noexcept {
    return cuda_render_math::isfinite(value.x) &&
    cuda_render_math::isfinite(value.y);
}

// 要素の最大値を返す
// NaN は伝播させる
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
max_abs_component(vec2<T> value) noexcept {
    return (cuda_render_math::max)(
        cuda_render_math::abs(value.x),
        cuda_render_math::abs(value.y)
    );
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
length(vec2<T> value) noexcept {
    const T scale = cuda_render_math::max_abs_component(value);
    if(scale == T(0) || !cuda_render_math::isfinite(scale)) return scale;
    const vec2<T> scaled = value / scale;
    return scale * cuda_render_math::sqrt(cuda_render_math::dot(scaled, scaled));
}

// ベクトルの正規化
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
try_normalize(vec2<T> value, vec2<T>& output) noexcept {
    if (!cuda_render_math::is_all_finite(value)) return false;
    const T scale = cuda_render_math::max_abs_component(value);
    if(scale == T(0)) return false;
    const vec2<T> scaled = value / scale;
    const T scaled_length = cuda_render_math::sqrt(cuda_render_math::dot(scaled, scaled));
    const vec2<T> normalized = scaled / scaled_length;
    if(!cuda_render_math::is_all_finite(normalized)) return false;
    output = normalized;
    return true;
}

// 失敗時は全要素が NaN のベクトルを返す
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
normalize(vec2<T> value) noexcept {
    vec2<T> normalized;
    if(cuda_render_math::try_normalize(value, normalized)) return normalized;
    return vec2<T>(detail::quiet_nan<T>());
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
reflect(
    vec2<T> incident,   // サーフェスに向かう方向
    vec2<T> unit_normal // 単位法線ベクトル
) noexcept {
    // 入力が不正な場合，全要素に qNaN を埋めて返す
    if (!cuda_render_math::is_all_finite(incident) || ! cuda_render_math::is_all_finite(unit_normal))
    {
        return vec2<T>(detail::quiet_nan<T>());
    }

    // 法線が全て 0 の場合，反射させずにそのまま方向を返す
    if (cuda_render_math::max_abs_component(incident) == T(0)) return incident;
    
    const T projection = cuda_render_math::dot(incident, unit_normal);
    return {
        cuda_render_math::fma(-projection, detail::mul_rn(T(2), unit_normal.x), incident.x),
        cuda_render_math::fma(-projection, detail::mul_rn(T(2), unit_normal.y), incident.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
reflect_from_outward(vec2<T> outward, vec2<T> unit_normal) noexcept {
    return cuda_render_math::reflect(-outward, unit_normal);
}

// -------------------
// 3次元ベクトル
// -------------------

// ベクトルの要素が有効かどうかをチェック
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
is_all_finite(vec3<T> value) noexcept {
    return  cuda_render_math::isfinite(value.x) &&
            cuda_render_math::isfinite(value.y) &&
            cuda_render_math::isfinite(value.z);
}

// 要素の最大値を返す
// NaN は伝播させる
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
max_abs_component(vec3<T> value) noexcept {
    return (
        cuda_render_math::max)(
        (cuda_render_math::max)(
            cuda_render_math::abs(value.x),
            cuda_render_math::abs(value.y)
        ),
        cuda_render_math::abs(value.z)
    );
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
length(vec3<T> value) noexcept {
    const T scale = cuda_render_math::max_abs_component(value);
    if(scale == T(0) || !cuda_render_math::isfinite(scale)) return scale;
    const vec3<T> scaled = value / scale;
    return scale * cuda_render_math::sqrt(cuda_render_math::dot(scaled, scaled));
}

// ベクトルの正規化
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
try_normalize(vec3<T> value, vec3<T>& output) noexcept {
    if (!cuda_render_math::is_all_finite(value)) return false;
    const T scale = cuda_render_math::max_abs_component(value);
    if(scale == T(0)) return false;
    const vec3<T> scaled = value / scale;
    const T scaled_length = cuda_render_math::sqrt(cuda_render_math::dot(scaled, scaled));
    const vec3<T> normalized = scaled / scaled_length;
    if(!cuda_render_math::is_all_finite(normalized)) return false;
    output = normalized;
    return true;
}

// 失敗時は全要素が NaN のベクトルを返す
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
normalize(vec3<T> value) noexcept {
    vec3<T> normalized;
    if(cuda_render_math::try_normalize(value, normalized)) return normalized;
    return vec3<T>(detail::quiet_nan<T>());
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
reflect(
    vec3<T> incident,   // サーフェスに向かう方向
    vec3<T> unit_normal // 単位法線ベクトル
) noexcept {
    // 入力が不正な場合，全要素に qNaN を埋めて返す
    if (!cuda_render_math::is_all_finite(incident) || ! cuda_render_math::is_all_finite(unit_normal))
    {
        return vec3<T>(detail::quiet_nan<T>());
    }

    // 法線が全て 0 の場合，反射させずにそのまま方向を返す
    if (cuda_render_math::max_abs_component(incident) == T(0)) return incident;
    
    const T projection = cuda_render_math::dot(incident, unit_normal);
    return {
        cuda_render_math::fma(-projection, detail::mul_rn(T(2), unit_normal.x), incident.x),
        cuda_render_math::fma(-projection, detail::mul_rn(T(2), unit_normal.y), incident.y),
        cuda_render_math::fma(-projection, detail::mul_rn(T(2), unit_normal.z), incident.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
reflect_from_outward(vec3<T> outward, vec3<T> unit_normal) noexcept {
    return cuda_render_math::reflect(-outward, unit_normal);
}

// -------------------
// 4次元ベクトル
// -------------------

// ベクトルの要素が有効かどうかをチェック
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
is_all_finite(vec4<T> value) noexcept {
    return  cuda_render_math::isfinite(value.x) &&
            cuda_render_math::isfinite(value.y) &&
            cuda_render_math::isfinite(value.z) &&
            cuda_render_math::isfinite(value.w);
}

// 要素の最大値を返す
// NaN は伝播させる
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
max_abs_component(vec4<T> value) noexcept {
    return (
        cuda_render_math::max)(
            (cuda_render_math::max)(
                (cuda_render_math::max)(
                    cuda_render_math::abs(value.x),
                    cuda_render_math::abs(value.y)
                ),
            cuda_render_math::abs(value.z)
            ),
        cuda_render_math::abs(value.w)
    );
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
length(vec4<T> value) noexcept {
    const T scale = cuda_render_math::max_abs_component(value);
    if(scale == T(0) || !cuda_render_math::isfinite(scale)) return scale;
    const vec4<T> scaled = value / scale;
    return scale * cuda_render_math::sqrt(cuda_render_math::dot(scaled, scaled));
}

// ベクトルの正規化
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
try_normalize(vec4<T> value, vec4<T>& output) noexcept {
    if (!cuda_render_math::is_all_finite(value)) return false;
    const T scale = cuda_render_math::max_abs_component(value);
    if(scale == T(0)) return false;
    const vec4<T> scaled = value / scale;
    const T scaled_length = cuda_render_math::sqrt(cuda_render_math::dot(scaled, scaled));
    const vec4<T> normalized = scaled / scaled_length;
    if(!cuda_render_math::is_all_finite(normalized)) return false;
    output = normalized;
    return true;
}

// 失敗時は全要素が NaN のベクトルを返す
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
normalize(vec4<T> value) noexcept {
    vec4<T> normalized;
    if(cuda_render_math::try_normalize(value, normalized)) return normalized;
    return vec4<T>(detail::quiet_nan<T>());
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
reflect(
    vec4<T> incident,   // サーフェスに向かう方向
    vec4<T> unit_normal // 単位法線ベクトル
) noexcept {
    // 入力が不正な場合，全要素に qNaN を埋めて返す
    if (!cuda_render_math::is_all_finite(incident) || ! cuda_render_math::is_all_finite(unit_normal))
    {
        return vec4<T>(detail::quiet_nan<T>());
    }

    // 法線が全て 0 の場合，反射させずにそのまま方向を返す
    if (cuda_render_math::max_abs_component(incident) == T(0)) return incident;
    
    const T projection = cuda_render_math::dot(incident, unit_normal);
    return {
        cuda_render_math::fma(-projection, detail::mul_rn(T(2), unit_normal.x), incident.x),
        cuda_render_math::fma(-projection, detail::mul_rn(T(2), unit_normal.y), incident.y),
        cuda_render_math::fma(-projection, detail::mul_rn(T(2), unit_normal.z), incident.z),
        cuda_render_math::fma(-projection, detail::mul_rn(T(2), unit_normal.w), incident.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
cross(vec3<T> a, vec3<T> b) noexcept {
    return {
        cuda_render_math::difference_of_products(a.y, b.z, a.z, b.y),
        cuda_render_math::difference_of_products(a.z, b.x, a.x, b.z),
        cuda_render_math::difference_of_products(a.x, b.y, a.y, b.x)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
reflect_from_outward(vec4<T> outward, vec4<T> unit_normal) noexcept {
    return cuda_render_math::reflect(-outward, unit_normal);
}


}