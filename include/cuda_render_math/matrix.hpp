#pragma once 

#include <cuda_render_math/scalar_numerics.hpp>
#include <cuda_render_math/vector.hpp>

#include <cassert>

namespace cuda_render_math {

// Row-major とする
// (A * B) * v のときは　B を最初に計算する
// デフォルトでは各要素を 0 とする
template<real_scalar T>
struct mat3 {
    T elements[9]{};

    constexpr mat3() noexcept = default;

    // T 単体で初期化されたときは，T を各要素とする対角行列をつくる
    CUDA_RENDER_MATH_HD explicit constexpr mat3(T diagonal) noexcept
        : elements{
            diagonal, T(0), T(0), 
            T(0), diagonal, T(0), 
            T(0), T(0), diagonal 
        } {}

    CUDA_RENDER_MATH_HD constexpr mat3(
        T m00, T m01, T m02, 
        T m10, T m11, T m12, 
        T m20, T m21, T m22 
    ) noexcept : elements {
        m00, m01, m02,
        m10, m11, m12,
        m20, m21, m22
    } {}

    // 単位行列を返す
    [[nodiscard]] CUDA_RENDER_MATH_HD static constexpr mat3 identity() noexcept {
        return mat3(T(1));
    }

    // 行ベクトルからの変換
    [[nodiscard]] CUDA_RENDER_MATH_HD static constexpr mat3 from_rows(
        vec3<T> a, vec3<T> b, vec3<T> c // 行ベクトル
    ) noexcept {
        return {
            a.x, a.y, a.z, 
            b.x, b.y, b.z, 
            c.x, c.y, c.z
        };
    }

    // 列ベクトルからの変換
    [[nodiscard]] CUDA_RENDER_MATH_HD static constexpr mat3 from_columns(
        vec3<T> a, vec3<T> b, vec3<T> c // 列ベクトル
    ) noexcept {
        return {
            a.x, b.x, c.x, 
            a.y, b.y, c.y, 
            a.z, b.z, c.z 
        };
    }

    // 指定した要素の取得
    [[nodiscard]] CUDA_RENDER_MATH_HD constexpr T& operator()(int r, int c) noexcept {
        assert(r >= 0 && r < 3 && c >= 0 && c < 3);
        return elements[3 * r + c];
    }

    [[nodiscard]] CUDA_RENDER_MATH_HD constexpr const T& operator()(int r, int c) const noexcept {
        assert(r >= 0 && r < 3 && c >= 0 && c < 3);
        return elements[3 * r + c];
    }
    
    // 指定した行の取得 (vec3 で返す)
    [[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T> row(int r) noexcept {
        assert(r >= 0 && r < 3);
        return {
            (*this)(r, 0), 
            (*this)(r, 1), 
            (*this)(r, 2) 
        };
    }

    // 指定した列の取得 (vec3 で返す)
    [[nodiscard]] CUDA_RENDER_MATH_HD constexpr vec3<T> column(int c) noexcept {
        assert(c >= 0 && c < 3);
        return {
            (*this)(0, c), 
            (*this)(1, c), 
            (*this)(2, c) 
        };
    }

    [[nodiscard]] CUDA_RENDER_MATH_HD constexpr T* data() noexcept { return elements; }
    [[nodiscard]] CUDA_RENDER_MATH_HD constexpr const T* data() const noexcept { return elements; }

};

using mat3f = mat3<float>;
using mat3d = mat3<double>;

// 型変換
template<real_scalar To, real_scalar From>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<To> matrix_cast(mat3<From> value) noexcept {
    mat3<To> result;
    for(int i = 0; i < 9; ++i){
        result.elements[i] = static_cast<To>(value.elements[i]);
    }
    return result;
}

// 転置
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T> transpose(mat3<T> value) noexcept {
    return mat3<T>::from_columns(value.row(0), value.row(1), value.row(2)); // TODO: from_rows(value.columns) とどっちが効率がよいか考える
}

// 和
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T> operator+(mat3<T> a, mat3<T> b) noexcept {
    for (int i = 0; i < 9; ++i){
        a.elements[i] += b.elements[i];
    }
    return a;
}

// 差
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T> operator-(mat3<T> a, mat3<T> b) noexcept {
    for (int i = 0; i < 9; ++i){
        a.elements[i] -= b.elements[i];
    }
    return a;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T> operator-(mat3<T> value) noexcept {
    for (int i = 0; i < 9; ++i){
        value.elements[i] = -value.elements[i];
    }
    return value;
}

// スカラ積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T> operator*(mat3<T> value, T scale) noexcept {
    for (int i = 0; i < 9; ++i){
        value.elements[i] *= scale;
    }
    return value;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T> operator*(T scale, mat3<T> value) noexcept {
    return value * scale;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T> operator/(mat3<T> value, T scale) noexcept {
    for (int i = 0; i < 9; ++i){
        value.elements[i] /= scale;
    }
    return value;
}

namespace detail {

    template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T matrix_dot3(vec3<T> a, vec3<T> b) noexcept {
    return detail::fma_rn(a.x, b.x, detail::fma_rn(a.y, b.y, detail::mul_rn(a.z, b.z)));
}

} // namespace detail

// ベクトルとの積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T> operator*(mat3<T> a, vec3<T> v) noexcept {
    return {
        detail::matrix_dot3(a.row(0), v),
        detail::matrix_dot3(a.row(1), v),
        detail::matrix_dot3(a.row(2), v)
    };
}

// 行列積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline mat3<T> operator*(mat3<T> a, mat3<T> b) noexcept {
    mat3<T> result;
    for (int r = 0; r < 3; ++r){
        for (int c = 0; c < 3; ++c) {
            result(r, c) = detail::matrix_dot3(a.row(r), b.column(c));
        }
    }
    return result;
}

// 複合代入演算子
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T>& operator+=(mat3<T> a, mat3<T> b) noexcept {
    a = a + b;
    return a;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T>& operator-=(mat3<T> a, mat3<T> b) noexcept {
    a = a - b;
    return a;
}

// スカラ積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T>& operator*=(mat3<T> a, T scale) noexcept {
    a = a * scale;
    return a;
}

// 行列積
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline mat3<T>& operator*=(mat3<T> a, mat3<T> b) noexcept {
    a = a * b;
    return a;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T>& operator/=(mat3<T> a, T scale) noexcept {
    a = a / scale;
    return a;
}

// 値のチェック
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool is_all_finite(mat3<T> value) noexcept {
    for (int i = 0; i < 9; ++i)
    {
        if(!cuda_render_math::isfinite(value.elements[i])) return false;
    }
    return true;
}

// 行列式
// 第 1 行に沿った余因子展開を計算する
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T determinant(mat3<T> m) noexcept {
    const T c0 = cuda_render_math::difference_of_products(m(1, 1), m(2, 2), m(1, 2), m(2, 1));
    const T c1 = cuda_render_math::difference_of_products(m(1, 2), m(2, 0), m(1, 0), m(2, 2));  // 符号は c1 の中に含めてしまう
    const T c2 = cuda_render_math::difference_of_products(m(1, 0), m(2, 1), m(1, 1), m(2, 0));
    return detail::matrix_dot3(m.row(0), vec3<T>{c0, c1, c2});
}

} // namespace cuda_render_math