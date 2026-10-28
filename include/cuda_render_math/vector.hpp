#pragma once

#include <cuda_render_math/config.hpp>
#include <cuda_render_math/scalar.hpp>

namespace cuda_render_math
{

// --------------------------
// 2 次元実数ベクトル
// --------------------------
template<real_scalar T>
struct vec2 {
    T x{};
    T y{};
};

using vec2f = vec2<float>;
using vec2d = vec2<double>;

// 精度の変更
template<real_scalar To, real_scalar From>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<To>
vector_cast(vec2<From> v) noexcept {
    return {
        static_cast<To>(v.x),
        static_cast<To>(v.y)
    };
}

// 符号の変更
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator-(vec2<T> v) noexcept {
    return {-v.x, -v.y};
}

// 足し算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator+(vec2<T> a, vec2<T> b) noexcept {
    return {
        a.x + b.x, 
        a.y + b.y
    };
}

// 引き算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator-(vec2<T> a, vec2<T> b) noexcept {
    return {
        a.x - b.x, 
        a.y - b.y
    };
}

// スカラとのかけ算 
// ベクトル同士のかけ算は，内積，外積などが紛らわしいので，敢えて * では定義しない
// 推論される T もそろえることで，意図しない異なる型の混在を拒否する
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator*(vec2<T> v, T s) noexcept {
    return {
        v.x * s, 
        v.y * s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator*(T s, vec2<T> v) noexcept {
    return v * s;
}

// スカラによる割り算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator/(vec2<T> v, T s) noexcept {
    return {
        v.x / s, 
        v.y / s 
    };
}

// 複合代入演算子
template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec2<T>&
operator+=(vec2<T>& a, vec2<T> b) noexcept {
    a.x += b.x; 
    a.y += b.y; 
    return a;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec2<T>&
operator-=(vec2<T>& a, vec2<T> b) noexcept {
    a.x -= b.x; 
    a.y -= b.y; 
    return a;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec2<T>&
operator*=(vec2<T>& v, T s) noexcept {
    v.x *= s; 
    v.y *= s; 
    return v;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec2<T>&
operator/=(vec2<T>& v, T s) noexcept {
    v.x /= s; 
    v.y /= s; 
    return v;
}

// 内積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr T
dot(vec2<T> a, vec2<T> b) noexcept{
    return a.x * b.x + a.y * b.y;
}

// 二乗ノルム
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr T
length_squared(vec2<T> v) noexcept {
    return cuda_render_math::dot(v, v);
}

// --------------------------
// 3 次元実数ベクトル
// --------------------------
template<real_scalar T>
struct vec3 {
    T x{};
    T y{};
    T z{};
};

using vec3f = vec3<float>;
using vec3d = vec3<double>;

// ベクトル型の型変換 float <-> double
template<real_scalar To, real_scalar From>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<To>
vector_cast(vec3<From> v) noexcept {
    return {
        static_cast<To>(v.x),
        static_cast<To>(v.y),
        static_cast<To>(v.z)
    };
}

// 符号の変更
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator-(vec3<T> v) noexcept {
    return {-v.x, -v.y, -v.z};
}

// 足し算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator+(vec3<T> a, vec3<T> b) noexcept {
    return {
        a.x + b.x, 
        a.y + b.y, 
        a.z + b.z 
    };
}

// 引き算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator-(vec3<T> a, vec3<T> b) noexcept {
    return {
        a.x - b.x, 
        a.y - b.y, 
        a.z - b.z 
    };
}

// スカラとのかけ算 
// ベクトル同士のかけ算は，内積，外積などが紛らわしいので，敢えて * では定義しない
// 推論される T もそろえることで，意図しない異なる型の混在を拒否する
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator*(vec3<T> v, T s) noexcept {
    return {
        v.x * s, 
        v.y * s, 
        v.z * s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator*(T s, vec3<T> v) noexcept {
    return v * s;
}

// スカラによる割り算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator/(vec3<T> v, T s) noexcept {
    return {
        v.x / s, 
        v.y / s, 
        v.z / s
    };
}

// 複合代入演算子
template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec3<T>&
operator+=(vec3<T>& a, vec3<T> b) noexcept {
    a.x += b.x; 
    a.y += b.y; 
    a.z += b.z;
    return a;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec3<T>&
operator-=(vec3<T>& a, vec3<T> b) noexcept {
    a.x -= b.x; 
    a.y -= b.y; 
    a.z -= b.z;
    return a;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec3<T>&
operator*=(vec3<T>& v, T s) noexcept {
    v.x *= s; 
    v.y *= s; 
    v.z *= s;
    return v;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec3<T>&
operator/=(vec3<T>& v, T s) noexcept {
    v.x /= s; 
    v.y /= s; 
    v.z /= s;
    return v;
}

// 内積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr T
dot(vec3<T> a, vec3<T> b) noexcept{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 外積 (右手系)
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
cross(vec3<T> a, vec3<T> b) noexcept{
    return {
        a.y * b.z - a.z * b.y,   
        a.z * b.x - a.x * b.z,   
        a.x * b.y - a.y * b.x   
    };
}

// 二乗ノルム
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr T
length_squared(vec3<T> v) noexcept {
    return cuda_render_math::dot(v, v);
}


} // namespace cuda_render_math