#pragma once

#include <cuda_render_math/matrix.hpp>

#include <cmath>
#include <limits>

namespace cuda_render_math {

namespace detail {

template<real_scalar T>
inline constexpr T inverse_residual_limit = T(128) * std::numeric_limits<T>::epsilon();


template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool 
inverse_residual_acceptable(
    mat3<T> product,
    T limit
) noexcept {
    for(int r = 0; r < 3; ++r) {
        T row_error = T(0);
        for(int c = 0; c < 3; ++c) {
            const T expected = r == c ? T(1) : T(0);
            row_error = detail::add_rn(
                row_error,
                cuda_render_math::abs(detail::sub_rn(product(r, c), expected))
            );
        }
        if(!(row_error <= limit)) return false;
    }
    return true;
}

// value / 2^E が [0.5, 1) に入るような指数 E を返す
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline int
inverse_scale_exponent(T value) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) {
        return ::ilogbf(value) + 1;
    } else {
        return ::ilogb(value) + 1;
    }
#else
    return std::ilogb(value) + 1;
#endif
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline T
inverse_scale_pow2(T value, int exponent) noexcept {
#if defined(__CUDA_ARCH__)
    if constexpr (std::same_as<T, float>) {
        return scalbnf(value, exponent);
    } else {
        return scalbn(value, exponent);
    }
#else
    return std::scalbn(value, exponent);
#endif
}


} // namespace detail

// 3x3 なので余因子展開で逆行列を計算する
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
try_inverse(
    mat3<T> value,      // 逆行列を計算したい行列
    mat3<T> &output,    // 計算結果
    T max_residual = detail::inverse_residual_limit<T>
) noexcept {
    // 入力が不正な値でないかどうかチェック
    if(
        !cuda_render_math::isfinite(max_residual) || 
        max_residual < T(0) ||
        !(max_residual < T(1)) ||
        !cuda_render_math::is_all_finite(value)
    ) {
        return false;
    }

    // 各行のスケールを揃える
    // 行ごとに値の桁が大きく違う状態を，そのまま処理しない (c.f. equilibration)
    int row_exponent[3]{};
    mat3<T> scaled;                          // スケール後の行列を格納
    
    for(int r = 0; r < 3; ++r) {
        T row_max = T(0);
        // 各行で最もスケールが大きい要素を探す
        for (int c = 0; c < 3; ++c){
            row_max = (cuda_render_math::max)(row_max, cuda_render_math::abs(value(r, c)));
        }
        // 当該行の全要素が 0 だった場合，false を返す
        if (row_max == T(0)) {
            return false;
        }

        row_exponent[r] = detail::inverse_scale_exponent(row_max);

        // スケーリング
        for(int c = 0; c <3; ++c) {
            scaled(r, c) = detail::inverse_scale_pow2(value(r, c), -row_exponent[r]);
        }
    }

    // 行列式を求める
    // 第一行による余因子展開を行う
    // 余因子
    const T c00 = cuda_render_math::difference_of_products(
        scaled(1, 1), scaled(2, 2),
        scaled(1, 2), scaled(2, 1)
    );
    
    // 符号も織り込み済
    const T c01 = cuda_render_math::difference_of_products(
        scaled(1, 2), scaled(2, 0),
        scaled(1, 0), scaled(2, 2)
    );
    
    const T c02 = cuda_render_math::difference_of_products(
        scaled(1, 0), scaled(2, 1),
        scaled(1, 1), scaled(2, 0)
    );

    // 余因子展開
    const T det = detail::matrix_dot3(
        scaled.row(0), vec3<T>{c00, c01, c02}
    );

    if(det == T(0) || !cuda_render_math::isfinite(det)) {
        return false;
    }

    const T c10 = cuda_render_math::difference_of_products(
        scaled(0, 2), scaled(2, 1), 
        scaled(0, 1), scaled(2, 2) 
    );
    
    const T c11 = cuda_render_math::difference_of_products(
        scaled(0, 0), scaled(2, 2), 
        scaled(0, 2), scaled(2, 0) 
    );
    
    const T c12 = cuda_render_math::difference_of_products(
        scaled(0, 1), scaled(2, 0), 
        scaled(0, 0), scaled(2, 1) 
    );

    const T c20 = cuda_render_math::difference_of_products(
        scaled(0, 1), scaled(1, 2), 
        scaled(0, 2), scaled(1, 1) 
    );
    
    const T c21 = cuda_render_math::difference_of_products(
        scaled(0, 2), scaled(1, 0), 
        scaled(0, 0), scaled(1, 2) 
    );
    
    const T c22 = cuda_render_math::difference_of_products(
        scaled(0, 0), scaled(1, 1), 
        scaled(0, 1), scaled(1, 0) 
    );

    mat3<T> candidate{
        c00, c10, c20,
        c01, c11, c21,
        c02, c12, c22
    };

    // 行のスケーリングを戻す
    const int det_exponent = detail::inverse_scale_exponent(
        cuda_render_math::abs(det)
    );

    const T det_mantissa = detail::inverse_scale_pow2(
        det, -det_exponent
    );


    for(int r = 0; r < 3; ++r) {
        // スケーリング
        for(int c = 0; c <3; ++c) {
            const T quotient = detail::div_rn(candidate(r, c), det_mantissa);
            candidate(r, c) = detail::inverse_scale_pow2(quotient, -det_exponent -row_exponent[c]);
        }
    }

    if(!cuda_render_math::is_all_finite(candidate)) return false;

    if(
        !detail::inverse_residual_acceptable(value * candidate, max_residual) ||
        !detail::inverse_residual_acceptable(candidate * value, max_residual) 
    ) return false;

    output = candidate;
    return true;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline mat3<T> inverse(mat3<T> value) noexcept {
    mat3<T> result;
    if(cuda_render_math::try_inverse(value, result)) return result;
    
    // 失敗した場合，全要素に qNaN を入れて返す
    const T nan = detail::quiet_nan<T>();
    for(int i = 0; i < 9; ++i){
        result.elements[i] = nan;
    } 
    return result;
}


} // namespace cuda_render_math