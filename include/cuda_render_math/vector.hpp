#pragma once

#include <cuda_render_math/config.hpp>
#include <cuda_render_math/scalar.hpp>

namespace cuda_render_math
{

// 2--4 次元の実数ベクトル
template<real_scalar T> struct vec3;
template<real_scalar T> struct vec3;
template<real_scalar T> struct vec4;


// デフォルトでは 0 で初期化
// 単一の値を受け取ったときは全要素に複製して初期化する
template<real_scalar T>
struct vec2 {
    T x{};
    T y{};

    // 基本は 0 初期化
    constexpr vec2() noexcept = default;

    // vec2(s) は {s, s} と初期化する
    CUDA_RENDER_MATH_HD explicit constexpr vec2(T s) noexcept
        : x(s), y(s) {}
    
    // vec2(x, y) は {x, y} と初期化する
    // 意図的に explicit を外している
    CUDA_RENDER_MATH_HD constexpr vec2(T x_value, T y_value) noexcept
        : x(x_value), y(y_value) {}
    
    CUDA_RENDER_MATH_HD explicit constexpr vec2(vec3<T> v) noexcept;
    CUDA_RENDER_MATH_HD explicit constexpr vec2(vec4<T> v) noexcept;
};

template<real_scalar T>
struct vec3 {
    T x{};
    T y{};
    T z{};

    // 基本は 0 初期化
    constexpr vec3() noexcept = default;

    // vec3(s) は {s, s, s} と初期化する
    CUDA_RENDER_MATH_HD explicit constexpr vec3(T s) noexcept
        : x(s), y(s), z(s) {}
    
    // vec3(x, y, z) は {x, y, z} と初期化する
    CUDA_RENDER_MATH_HD constexpr vec3(T x_value, T y_value, T z_value) noexcept
        : x(x_value), y(y_value), z(z_value) {}
    
    // vec3(vec2) は {vex2.x, vec2.y, 0} と初期化する
    CUDA_RENDER_MATH_HD explicit constexpr vec3(vec2<T> xy) noexcept
        : x{xy.x}, y{xy.y}, z{} {}
    
    // vec3(vec2, s) は {vex2.x, vec2.y, s} と初期化する
    CUDA_RENDER_MATH_HD explicit constexpr vec3(vec2<T> xy, T z_value) noexcept
        : x{xy.x}, y{xy.y}, z{z_value} {}
    
    // vec3(s, vec2) は {s, vex2.x, vec2.y} と初期化する
    CUDA_RENDER_MATH_HD explicit constexpr vec3(T x_value, vec2<T> yz) noexcept
        : x{x_value}, y{yz.x}, z{yz.y} {}
    
    CUDA_RENDER_MATH_HD explicit constexpr vec3(vec4<T> v) noexcept;
};

template<real_scalar T>
struct vec4 {
    T x{};
    T y{};
    T z{};
    T w{};

    // 基本は 0 初期化
    constexpr vec4() noexcept = default;

    // vec4(s) は {s, s, s, s} と初期化する
    CUDA_RENDER_MATH_HD explicit constexpr vec4(T s) noexcept
        : x(s), y(s), z(s), w(s) {}
    
    // vec4(x, y, z, w) は {x, y, z, w} と初期化する
    CUDA_RENDER_MATH_HD constexpr vec4(T x_value, T y_value, T z_value, T w_value) noexcept
        : x(x_value), y(y_value), z(z_value), w(w_value) {}
    
    // 足りない成分は 0 で初期化する
    CUDA_RENDER_MATH_HD explicit constexpr vec4(vec2<T> xy) noexcept
        : x{xy.x}, y{xy.y}, z{}, w{} {}
    
    CUDA_RENDER_MATH_HD explicit constexpr vec4(vec3<T> xyz) noexcept
        : x{xyz.x}, y{xyz.y}, z{xyz.z}, w{} {}
    
    CUDA_RENDER_MATH_HD explicit constexpr vec4(vec3<T> xyz, T w_value) noexcept
        : x{xyz.x}, y{xyz.y}, z{xyz.z}, w{w_value} {}
    
    CUDA_RENDER_MATH_HD explicit constexpr vec4(T x_value, vec3<T> yzw) noexcept
        : x{x_value}, y{yzw.x}, z{yzw.y}, w{yzw.z} {}
    
    CUDA_RENDER_MATH_HD explicit constexpr vec4(vec2<T> xy, vec2<T> zw) noexcept
        : x{xy.x}, y{xy.y}, z{zw.x}, w{zw.y} {}
    
    CUDA_RENDER_MATH_HD explicit constexpr vec4(vec2<T> xy, T z_value, T w_value) noexcept
        : x{xy.x}, y{xy.y}, z{z_value}, w{w_value} {}
    
    CUDA_RENDER_MATH_HD explicit constexpr vec4(T x_value, vec2<T> yz, T w_value) noexcept
        : x{x_value}, y{yz.x}, z{yz.y}, w{w_value} {}
    
    CUDA_RENDER_MATH_HD explicit constexpr vec4(T x_value, T y_value, vec2<T> zw) noexcept
        : x{x_value}, y{y_value}, z{zw.x}, w{zw.y} {}
};

// 縮小するときは先頭部分を取り出す
// 取り出すだけで正規化は行わない
template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec2<T>::vec2(vec3<T> v) noexcept
    : x(v.x), y(v.y) {}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec2<T>::vec2(vec4<T> v) noexcept
    : x(v.x), y(v.y) {}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec3<T>::vec3(vec4<T> v) noexcept
    : x(v.x), y(v.y), z(v.z) {}

using vec2f = vec2<float>;
using vec2d = vec2<double>;
using vec3f = vec3<float>;
using vec3d = vec3<double>;
using vec4f = vec4<float>;
using vec4d = vec4<double>;

// --------------------------
// 2 次元実数ベクトルの計算
// --------------------------
// ベクトル型の型変換 float <-> double
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

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator+(vec2<T> v, T s) noexcept {
    return {
        v.x + s, 
        v.y + s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator+(T s, vec2<T> v) noexcept {
    return {
        s + v.x, 
        s + v.y
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

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator-(vec2<T> v, T s) noexcept {
    return {
        v.x - s, 
        v.y - s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator-(T s, vec2<T> v) noexcept {
    return {
        s - v.x, 
        s - v.y
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


template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
operator/(T s, vec2<T> v) noexcept {
    return {
        s / v.x, 
        s / v.y
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
operator+=(vec2<T>& v, T s) noexcept {
    v.x += s; 
    v.y += s; 
    return v;
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
operator-=(vec2<T>& v, T s) noexcept {
    v.x -= s; 
    v.y -= s; 
    return v;
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

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec2<T>
operator/=(T s, vec3<T> v) noexcept {
    return{
        s / v.x,
        s / v.y
    };
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

// 要素積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
cwise_mul(vec2<T> a, vec2<T> b) noexcept {
    return {
        a.x * b.x,
        a.y * b.y
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>&
cwise_mul_assign(vec2<T>& a, vec2<T> b) noexcept {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

// 要素ごとの除算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>
cwise_div(vec2<T> a, vec2<T> b) noexcept {
    return {
        a.x / b.x,
        a.y / b.y
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec2<T>&
cwise_div_assign(vec2<T>& a, vec2<T> b) noexcept {
    a.x /= b.x;
    a.y /= b.y;
    return a;
}

// --------------------------
// 3 次元実数ベクトルの計算
// --------------------------
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

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator+(vec3<T> v, T s) noexcept {
    return {
        v.x + s, 
        v.y + s,
        v.z + s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator+(T s, vec3<T> v) noexcept {
    return {
        s + v.x, 
        s + v.y,
        s + v.z
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

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator-(vec3<T> v, T s) noexcept {
    return {
        v.x - s, 
        v.y - s,
        v.z - s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator-(T s, vec3<T> v) noexcept {
    return {
        s - v.x, 
        s - v.y,
        s - v.z
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

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
operator/(T s, vec3<T> v) noexcept {
    return {
        s / v.x, 
        s / v.y, 
        s / v.z
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
operator+=(vec3<T>& v, T s) noexcept {
    v.x += s; 
    v.y += s; 
    v.z += s; 
    return v;
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
operator-=(vec3<T>& v, T s) noexcept {
    v.x -= s; 
    v.y -= s; 
    v.z -= s; 
    return v;
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

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec3<T>
operator/=(T s, vec3<T> v) noexcept {
    return{
        s / v.x,
        s / v.y,
        s / v.z
    };
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

// 要素積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
cwise_mul(vec3<T> a, vec3<T> b) noexcept {
    return {
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>&
cwise_mul_assign(vec3<T>& a, vec3<T> b) noexcept {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

// 要素ごとの除算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>
cwise_div(vec3<T> a, vec3<T> b) noexcept {
    return {
        a.x / b.x,
        a.y / b.y,
        a.z / b.z
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T>&
cwise_div_assign(vec3<T>& a, vec3<T> b) noexcept {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

// --------------------------
// 4 次元実数ベクトルの計算
// --------------------------
// ベクトル型の型変換 float <-> double
template<real_scalar To, real_scalar From>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<To>
vector_cast(vec4<From> v) noexcept {
    return {
        static_cast<To>(v.x),
        static_cast<To>(v.y),
        static_cast<To>(v.z),
        static_cast<To>(v.w)
    };
}

// 符号の変更
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator-(vec4<T> v) noexcept {
    return {-v.x, -v.y, -v.z, -v.w};
}

// 足し算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator+(vec4<T> a, vec4<T> b) noexcept {
    return {
        a.x + b.x, 
        a.y + b.y, 
        a.z + b.z, 
        a.w + b.w 
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator+(vec4<T> v, T s) noexcept {
    return {
        v.x + s, 
        v.y + s,
        v.z + s,
        v.w + s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator+(T s, vec4<T> v) noexcept {
    return {
        s + v.x, 
        s + v.y,
        s + v.z,
        s + v.w
    };
}

// 引き算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator-(vec4<T> a, vec4<T> b) noexcept {
    return {
        a.x - b.x, 
        a.y - b.y, 
        a.z - b.z, 
        a.w - b.w 
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator-(vec4<T> v, T s) noexcept {
    return {
        v.x - s, 
        v.y - s,
        v.z - s,
        v.w - s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator-(T s, vec4<T> v) noexcept {
    return {
        s - v.x, 
        s - v.y,
        s - v.z,
        s - v.w
    };
}

// スカラとのかけ算 
// ベクトル同士のかけ算は，内積，外積などが紛らわしいので，敢えて * では定義しない
// 推論される T もそろえることで，意図しない異なる型の混在を拒否する
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator*(vec4<T> v, T s) noexcept {
    return {
        v.x * s, 
        v.y * s, 
        v.z * s,
        v.w * s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator*(T s, vec4<T> v) noexcept {
    return v * s;
}

// 割り算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator/(vec4<T> v, T s) noexcept {
    return {
        v.x / s, 
        v.y / s, 
        v.z / s,
        v.w / s
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
operator/(T s, vec4<T> v) noexcept {
    return {
        s / v.x, 
        s / v.y, 
        s / v.z,
        s / v.w
    };
}


// 複合代入演算子
template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec4<T>&
operator+=(vec4<T>& a, vec4<T> b) noexcept {
    a.x += b.x; 
    a.y += b.y; 
    a.z += b.z;
    a.w += b.w;
    return a;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec4<T>&
operator+=(vec4<T>& v, T s) noexcept {
    v.x += s; 
    v.y += s; 
    v.z += s; 
    v.w += s; 
    return v;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec4<T>&
operator-=(vec4<T>& a, vec4<T> b) noexcept {
    a.x -= b.x; 
    a.y -= b.y; 
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec4<T>&
operator-=(vec4<T>& v, T s) noexcept {
    v.x -= s; 
    v.y -= s; 
    v.z -= s; 
    v.w -= s; 
    return v;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec4<T>&
operator*=(vec4<T>& v, T s) noexcept {
    v.x *= s; 
    v.y *= s; 
    v.z *= s;
    v.w *= s;
    return v;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec4<T>&
operator/=(vec4<T>& v, T s) noexcept {
    v.x /= s; 
    v.y /= s; 
    v.z /= s;
    v.w /= s;
    return v;
}

template<real_scalar T>
CUDA_RENDER_MATH_HD constexpr vec4<T>
operator/=(T s, vec4<T> v) noexcept {
    return{
        s / v.x,
        s / v.y,
        s / v.z,
        s / v.w
    };
}

// 内積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr T
dot(vec4<T> a, vec4<T> b) noexcept{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// 二乗ノルム
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr T
length_squared(vec4<T> v) noexcept {
    return cuda_render_math::dot(v, v);
}

// 要素積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
cwise_mul(vec4<T> a, vec4<T> b) noexcept {
    return {
        a.x * b.x,
        a.y * b.y,
        a.z * b.z,
        a.w * b.w
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>&
cwise_mul_assign(vec4<T>& a, vec4<T> b) noexcept {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

// 要素ごとの除算
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>
cwise_div(vec4<T> a, vec4<T> b) noexcept {
    return {
        a.x / b.x,
        a.y / b.y,
        a.z / b.z,
        a.w / b.w
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec4<T>&
cwise_div_assign(vec4<T>& a, vec4<T> b) noexcept {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
    return a;
}



} // namespace cuda_render_math