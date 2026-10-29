#pragma once 

#include <cuda_render_math/scalar_math.hpp>
#include <cuda_render_math/vector.hpp>

namespace cuda_render_math {

// -------------------------
// 2 次元
// -------------------------

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
abs(vec2<T> v) noexcept {
    return {
        cuda_render_math::abs(v.x),
        cuda_render_math::abs(v.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
floor(vec2<T> v) noexcept {
    return {
        cuda_render_math::floor(v.x),
        cuda_render_math::floor(v.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
ceil(vec2<T> v) noexcept {
    return {
        cuda_render_math::ceil(v.x),
        cuda_render_math::ceil(v.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
fract(vec2<T> v) noexcept {
    return {
        cuda_render_math::fract(v.x),
        cuda_render_math::fract(v.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec2<T>
(min)(vec2<T> a, vec2<T> b) noexcept {
    return {
        (cuda_render_math::min)(a.x, b.x),
        (cuda_render_math::min)(a.y, b.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec2<T>
(min)(vec2<T> a, T b) noexcept {
    return {
        (cuda_render_math::min)(a.x, b),
        (cuda_render_math::min)(a.y, b)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec2<T>
(min)(T a, vec2<T> b) noexcept {
    return {
        (cuda_render_math::min)(a, b.x),
        (cuda_render_math::min)(a, b.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec2<T>
(max)(vec2<T> a, vec2<T> b) noexcept {
    return {
        (cuda_render_math::max)(a.x, b.x),
        (cuda_render_math::max)(a.y, b.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec2<T>
(max)(vec2<T> a, T b) noexcept {
    return {
        (cuda_render_math::max)(a.x, b),
        (cuda_render_math::max)(a.y, b)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec2<T>
(max)(T a, vec2<T> b) noexcept {
    return {
        (cuda_render_math::max)(a, b.x),
        (cuda_render_math::max)(a, b.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
(fmod)(vec2<T> a, vec2<T> b) noexcept {
    return {
        (cuda_render_math::fmod)(a.x, b.x),
        (cuda_render_math::fmod)(a.y, b.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
(fmod)(vec2<T> a, T b) noexcept {
    return {
        (cuda_render_math::fmod)(a.x, b),
        (cuda_render_math::fmod)(a.y, b)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
(fmod)(T a, vec2<T> b) noexcept {
    return {
        (cuda_render_math::fmod)(a, b.x),
        (cuda_render_math::fmod)(a, b.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
clamp(vec2<T> value, vec2<T> lower, vec2<T> upper) noexcept {
    return {
        cuda_render_math::clamp(value.x, lower.x, upper.x),
        cuda_render_math::clamp(value.y, lower.y, upper.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
clamp(vec2<T> value, T lower, T upper) noexcept {
    return {
        cuda_render_math::clamp(value.x, lower, upper),
        cuda_render_math::clamp(value.y, lower, upper)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
saturate(vec2<T> value) noexcept {
    return {
        cuda_render_math::saturate(value.x),
        cuda_render_math::saturate(value.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
lerp(vec2<T> a, vec2<T> b, T t) noexcept {
    return {
        cuda_render_math::lerp(a.x, b.x, t),
        cuda_render_math::lerp(a.y, b.y, t)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
lerp(vec2<T> a, vec2<T> b, vec2<T> t) noexcept {
    return {
        cuda_render_math::lerp(a.x, b.x, t.x),
        cuda_render_math::lerp(a.y, b.y, t.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
smoothstep(vec2<T> lower, vec2<T> upper, vec2<T> x) noexcept {
    return {
        cuda_render_math::smoothstep(lower.x, upper.x, x.x),
        cuda_render_math::smoothstep(lower.y, upper.y, x.y)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec2<T>
smoothstep(T lower, T upper, vec2<T> x) noexcept {
    return {
        cuda_render_math::smoothstep(lower, upper, x.x),
        cuda_render_math::smoothstep(lower, upper, x.y)
    };
}

// -------------------------
// 3 次元
// -------------------------

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
abs(vec3<T> v) noexcept {
    return {
        cuda_render_math::abs(v.x),
        cuda_render_math::abs(v.y),
        cuda_render_math::abs(v.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
floor(vec3<T> v) noexcept {
    return {
        cuda_render_math::floor(v.x),
        cuda_render_math::floor(v.y),
        cuda_render_math::floor(v.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
ceil(vec3<T> v) noexcept {
    return {
        cuda_render_math::ceil(v.x),
        cuda_render_math::ceil(v.y),
        cuda_render_math::ceil(v.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
fract(vec3<T> v) noexcept {
    return {
        cuda_render_math::fract(v.x),
        cuda_render_math::fract(v.y),
        cuda_render_math::fract(v.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec3<T>
(min)(vec3<T> a, vec3<T> b) noexcept {
    return {
        (cuda_render_math::min)(a.x, b.x),
        (cuda_render_math::min)(a.y, b.y),
        (cuda_render_math::min)(a.z, b.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec3<T>
(min)(vec3<T> a, T b) noexcept {
    return {
        (cuda_render_math::min)(a.x, b),
        (cuda_render_math::min)(a.y, b),
        (cuda_render_math::min)(a.z, b)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec3<T>
(min)(T a, vec3<T> b) noexcept {
    return {
        (cuda_render_math::min)(a, b.x),
        (cuda_render_math::min)(a, b.y),
        (cuda_render_math::min)(a, b.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec3<T>
(max)(vec3<T> a, vec3<T> b) noexcept {
    return {
        (cuda_render_math::max)(a.x, b.x),
        (cuda_render_math::max)(a.y, b.y),
        (cuda_render_math::max)(a.z, b.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec3<T>
(max)(vec3<T> a, T b) noexcept {
    return {
        (cuda_render_math::max)(a.x, b),
        (cuda_render_math::max)(a.y, b),
        (cuda_render_math::max)(a.z, b)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec3<T>
(max)(T a, vec3<T> b) noexcept {
    return {
        (cuda_render_math::max)(a, b.x),
        (cuda_render_math::max)(a, b.y),
        (cuda_render_math::max)(a, b.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
fmod(vec3<T> a, vec3<T> b) noexcept {
    return {
        (cuda_render_math::fmod)(a.x, b.x),
        (cuda_render_math::fmod)(a.y, b.y),
        (cuda_render_math::fmod)(a.z, b.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
fmod(vec3<T> a, T b) noexcept {
    return {
        (cuda_render_math::fmod)(a.x, b),
        (cuda_render_math::fmod)(a.y, b),
        (cuda_render_math::fmod)(a.z, b)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
fmod(T a, vec3<T> b) noexcept {
    return {
        (cuda_render_math::fmod)(a, b.x),
        (cuda_render_math::fmod)(a, b.y),
        (cuda_render_math::fmod)(a, b.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
clamp(vec3<T> value, vec3<T> lower, vec3<T> upper) noexcept {
    return {
        cuda_render_math::clamp(value.x, lower.x, upper.x),
        cuda_render_math::clamp(value.y, lower.y, upper.y),
        cuda_render_math::clamp(value.z, lower.z, upper.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
clamp(vec3<T> value, T lower, T upper) noexcept {
    return {
        cuda_render_math::clamp(value.x, lower, upper),
        cuda_render_math::clamp(value.y, lower, upper),
        cuda_render_math::clamp(value.z, lower, upper)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
saturate(vec3<T> value) noexcept {
    return {
        cuda_render_math::saturate(value.x),
        cuda_render_math::saturate(value.y),
        cuda_render_math::saturate(value.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
lerp(vec3<T> a, vec3<T> b, T t) noexcept {
    return {
        cuda_render_math::lerp(a.x, b.x, t),
        cuda_render_math::lerp(a.y, b.y, t),
        cuda_render_math::lerp(a.z, b.z, t)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
lerp(vec3<T> a, vec3<T> b, vec3<T> t) noexcept {
    return {
        cuda_render_math::lerp(a.x, b.x, t.x),
        cuda_render_math::lerp(a.y, b.y, t.y),
        cuda_render_math::lerp(a.z, b.z, t.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
smoothstep(vec3<T> lower, vec3<T> upper, vec3<T> x) noexcept {
    return {
        cuda_render_math::smoothstep(lower.x, upper.x, x.x),
        cuda_render_math::smoothstep(lower.y, upper.y, x.y),
        cuda_render_math::smoothstep(lower.z, upper.z, x.z)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
smoothstep(T lower, T upper, vec3<T> x) noexcept {
    return {
        cuda_render_math::smoothstep(lower, upper, x.x),
        cuda_render_math::smoothstep(lower, upper, x.y),
        cuda_render_math::smoothstep(lower, upper, x.z)
    };
}


// -------------------------
// 4 次元
// -------------------------

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
abs(vec4<T> v) noexcept {
    return {
        cuda_render_math::abs(v.x),
        cuda_render_math::abs(v.y),
        cuda_render_math::abs(v.z),
        cuda_render_math::abs(v.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
floor(vec4<T> v) noexcept {
    return {
        cuda_render_math::floor(v.x),
        cuda_render_math::floor(v.y),
        cuda_render_math::floor(v.z),
        cuda_render_math::floor(v.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
ceil(vec4<T> v) noexcept {
    return {
        cuda_render_math::ceil(v.x),
        cuda_render_math::ceil(v.y),
        cuda_render_math::ceil(v.z),
        cuda_render_math::ceil(v.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
fract(vec4<T> v) noexcept {
    return {
        cuda_render_math::fract(v.x),
        cuda_render_math::fract(v.y),
        cuda_render_math::fract(v.z),
        cuda_render_math::fract(v.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec4<T>
(min)(vec4<T> a, vec4<T> b) noexcept {
    return {
        (cuda_render_math::min)(a.x, b.x),
        (cuda_render_math::min)(a.y, b.y),
        (cuda_render_math::min)(a.z, b.z),
        (cuda_render_math::min)(a.w, b.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec4<T>
(min)(vec4<T> a, T b) noexcept {
    return {
        (cuda_render_math::min)(a.x, b),
        (cuda_render_math::min)(a.y, b),
        (cuda_render_math::min)(a.z, b),
        (cuda_render_math::min)(a.w, b)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec4<T>
(min)(T a, vec4<T> b) noexcept {
    return {
        (cuda_render_math::min)(a, b.x),
        (cuda_render_math::min)(a, b.y),
        (cuda_render_math::min)(a, b.z),
        (cuda_render_math::min)(a, b.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec4<T>
(max)(vec4<T> a, vec4<T> b) noexcept {
    return {
        (cuda_render_math::max)(a.x, b.x),
        (cuda_render_math::max)(a.y, b.y),
        (cuda_render_math::max)(a.z, b.z),
        (cuda_render_math::max)(a.w, b.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec4<T>
(max)(vec4<T> a, T b) noexcept {
    return {
        (cuda_render_math::max)(a.x, b),
        (cuda_render_math::max)(a.y, b),
        (cuda_render_math::max)(a.z, b),
        (cuda_render_math::max)(a.w, b)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline constexpr vec4<T>
(max)(T a, vec4<T> b) noexcept {
    return {
        (cuda_render_math::max)(a, b.x),
        (cuda_render_math::max)(a, b.y),
        (cuda_render_math::max)(a, b.z),
        (cuda_render_math::max)(a, b.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
fmod(vec4<T> a, vec4<T> b) noexcept {
    return {
        (cuda_render_math::fmod)(a.x, b.x),
        (cuda_render_math::fmod)(a.y, b.y),
        (cuda_render_math::fmod)(a.z, b.z),
        (cuda_render_math::fmod)(a.w, b.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
fmod(vec4<T> a, T b) noexcept {
    return {
        (cuda_render_math::fmod)(a.x, b),
        (cuda_render_math::fmod)(a.y, b),
        (cuda_render_math::fmod)(a.z, b),
        (cuda_render_math::fmod)(a.w, b)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
fmod(T a, vec4<T> b) noexcept {
    return {
        (cuda_render_math::fmod)(a, b.x),
        (cuda_render_math::fmod)(a, b.y),
        (cuda_render_math::fmod)(a, b.z),
        (cuda_render_math::fmod)(a, b.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
clamp(vec4<T> value, vec4<T> lower, vec4<T> upper) noexcept {
    return {
        cuda_render_math::clamp(value.x, lower.x, upper.x),
        cuda_render_math::clamp(value.y, lower.y, upper.y),
        cuda_render_math::clamp(value.z, lower.z, upper.z),
        cuda_render_math::clamp(value.w, lower.w, upper.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
clamp(vec4<T> value, T lower, T upper) noexcept {
    return {
        cuda_render_math::clamp(value.x, lower, upper),
        cuda_render_math::clamp(value.y, lower, upper),
        cuda_render_math::clamp(value.z, lower, upper),
        cuda_render_math::clamp(value.w, lower, upper)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
saturate(vec4<T> value) noexcept {
    return {
        cuda_render_math::saturate(value.x),
        cuda_render_math::saturate(value.y),
        cuda_render_math::saturate(value.z),
        cuda_render_math::saturate(value.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
lerp(vec4<T> a, vec4<T> b, T t) noexcept {
    return {
        cuda_render_math::lerp(a.x, b.x, t),
        cuda_render_math::lerp(a.y, b.y, t),
        cuda_render_math::lerp(a.z, b.z, t),
        cuda_render_math::lerp(a.w, b.w, t)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
lerp(vec4<T> a, vec4<T> b, vec4<T> t) noexcept {
    return {
        cuda_render_math::lerp(a.x, b.x, t.x),
        cuda_render_math::lerp(a.y, b.y, t.y),
        cuda_render_math::lerp(a.z, b.z, t.z),
        cuda_render_math::lerp(a.w, b.w, t.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
smoothstep(vec4<T> lower, vec4<T> upper, vec4<T> x) noexcept {
    return {
        cuda_render_math::smoothstep(lower.x, upper.x, x.x),
        cuda_render_math::smoothstep(lower.y, upper.y, x.y),
        cuda_render_math::smoothstep(lower.z, upper.z, x.z),
        cuda_render_math::smoothstep(lower.w, upper.w, x.w)
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec4<T>
smoothstep(T lower, T upper, vec4<T> x) noexcept {
    return {
        cuda_render_math::smoothstep(lower, upper, x.x),
        cuda_render_math::smoothstep(lower, upper, x.y),
        cuda_render_math::smoothstep(lower, upper, x.z),
        cuda_render_math::smoothstep(lower, upper, x.w)
    };
}

}