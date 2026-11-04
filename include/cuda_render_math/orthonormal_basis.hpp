#pragma once 

#include <cuda_render_math/matrix.hpp>
#include <cuda_render_math/vector_geometry.hpp>

#include <limits>

namespace cuda_render_math {

template<real_scalar T>
struct orthonormal_basis3 {
    vec3<T> x{T(1), T(0), T(0)};
    vec3<T> y{T(0), T(1), T(0)};
    vec3<T> z{T(0), T(0), T(1)};
};

using orthonormal_basis3f = orthonormal_basis3<float>;
using orthonormal_basis3d = orthonormal_basis3<double>;

namespace detail {

// 誤差の許容度 (64 が適切かどうかは要検討)
template<real_scalar T>
inline constexpr T basis_tolerance = T(64) * std::numeric_limits<T>::epsilon();
} // namespace epsilon


// [Duff et al. 2017] Sign-conditioned orthonormal-basis construction (JCGT)
// https://jcgt.org/published/0006/01/01/
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline orthonormal_basis3<T>
basis_from_unit_normal(vec3<T> n) noexcept {
    const T sign = cuda_render_math::copysign(T(1), n.z);
    const T reciprocal = detail::div_rn(T(-1), detail::add_rn(sign, n.z));
    const T nx_scaled = detail::mul_rn(n.x, reciprocal);
    const T ny_scaled = detail::mul_rn(n.y, reciprocal);
    const T mixed = detail::mul_rn(n.x, ny_scaled);
    return {
        {detail::fma_rn(sign * n.x, nx_scaled, T(1)), sign * mixed, -sign * n.x},
        {mixed, detail::fma_rn(n.y, ny_scaled, sign), -n.y},
        n
    };
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
try_make_orthonormal_basis(vec3<T> normal, orthonormal_basis3<T>& output) noexcept {
    vec3<T> unit;
    if(!cuda_render_math::try_normalize(normal, unit)) return false;
    orthonormal_basis3<T> candidate = cuda_render_math::basis_from_unit_normal(unit);
    if (!cuda_render_math::is_all_finite(candidate.x) || !cuda_render_math::is_all_finite(candidate.y)) return false;
    output = candidate;
    return true;
}

// tangent 方向も与えられている場合
// 局所座標の x 軸を tangent にそわせる
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
try_make_orthonormal_basis(vec3<T> normal, vec3<T> tangent_hint, orthonormal_basis3<T>& output) noexcept {
    orthonormal_basis3<T> candidate;
    vec3<T> tangent;
    // 基底 z を確定
    if(
        !cuda_render_math::try_normalize(normal, candidate.z) ||    // normal を正規化して candidate.z に格納
        !cuda_render_math::try_normalize(tangent_hint, tangent)     // tangent_hint を正規化して tangent に格納
    )
    {
        return false;
    }

    // 基底 y を確定
    if (!cuda_render_math::try_normalize(cuda_render_math::cross(candidate.z, tangent), candidate.y))
    {
        return false;
    }

    // 基底 x を確定
    if (!cuda_render_math::try_normalize(cuda_render_math::cross(candidate.y, candidate.z), candidate.x))
    {
        return false;
    }

    output = candidate;
    return true;
}


// 与えられた法線から直交基底を作る
// 失敗した場合は基底の全要素を qNaN で返す
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline orthonormal_basis3<T>
make_orthonormal_basis(vec3<T> normal) noexcept {
    orthonormal_basis3<T> result;
    if(cuda_render_math::try_make_orthonormal_basis(normal, result)) return result;
    const vec3<T> invalid(detail::quiet_nan<T>());
    return {invalid, invalid, invalid};
}

// orthonormal basis と vec を計算させるための関数
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T>
to_matrix(orthonormal_basis3<T> basis) noexcept {
    return mat3<T>::from_columns(basis.x, basis.y, basis.z);
}

// ワールド空間 -> ローカル空間
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
to_local(orthonormal_basis3<T> basis, vec3<T> world) noexcept {
    return cuda_render_math::transpose(cuda_render_math::to_matrix(basis)) * world;
}

// ローカル空間 -> ワールド空間
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
to_world(orthonormal_basis3<T> basis, vec3<T> local) noexcept {
    return cuda_render_math::to_matrix(basis) * local;
}

// 直交性の確認
template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
is_orthonormal(
    orthonormal_basis3<T> basis,
    T tolerance = detail::basis_tolerance<T>
) noexcept {
    // 不正な値は false
    if(
        !cuda_render_math::isfinite(tolerance) || 
        tolerance <T(0) || 
        !cuda_render_math::is_all_finite(basis.x) ||
        !cuda_render_math::is_all_finite(basis.y) ||
        !cuda_render_math::is_all_finite(basis.z) 
    ) return false;

    // 正規直交基底であればグラム行列が単位行列になるはず
    const mat3<T> m = cuda_render_math::to_matrix(basis);
    const mat3<T> gram = cuda_render_math::transpose(m) * m;

    // 単位行列になっているかどうかをチェック
    // ぴったり 0 / 1 にならない場合もあるので，許容範囲を超えていたら false
    for(int r = 0; r < 3; ++r){
        for(int c = 0; c < 3; ++c){
            if(!(cuda_render_math::abs(gram(r, c) - (r == c ? T(1) : T(0))) <= tolerance))
            {
                return false;
            }
        }
    }

    // 右手系を採用しているので，左手系は false
    // 正規直交基底の行列式は 1 / -1 になる
    const T orientation = cuda_render_math::determinant(m);
    return orientation > T(0) && cuda_render_math::abs(orientation - T(1)) <= T(4) * tolerance; // 4 はマジックナンバーなので再考の余地あり
}

} // namespace cuda_render_math