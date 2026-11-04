#pragma once

#include <cuda_render_math/matrix_numerics.hpp>
#include <cuda_render_math/vector_geometry.hpp>

namespace cuda_render_math {

template<real_scalar T>
struct affine_transform3 {
    mat3<T> linear{T(1)};
    vec3<T> translation{};
};

using affine_transform3f = affine_transform3<float>;
using affine_transform3d = affine_transform3<double>;

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool is_all_finite(affine_transform3<T> value) noexcept {
    return cuda_render_math::is_all_finite(value.linear) &&
        cuda_render_math::is_all_finite(value.translation);
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
transform_point(affine_transform3<T> transform, vec3<T> point) noexcept {
    return transform.linear * point + transform.translation;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
transform_vector(affine_transform3<T> transform, vec3<T> vector) noexcept {
    return transform.linear * vector;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline affine_transform3<T>
operator*(affine_transform3<T> a, affine_transform3<T> b) noexcept {
    return {a.linear * b.linear, a.linear * b.translation + a.translation};
}

template<real_scalar T>
CUDA_RENDER_MATH_HD inline affine_transform3<T>&
operator*=(affine_transform3<T>& a, affine_transform3<T> b) noexcept {
    a = a * b;
    return a;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline bool
try_inverse(
    affine_transform3<T> value, 
    affine_transform3<T>& output,
    T max_residual = detail::inverse_residual_limit<T>
) noexcept {
    if(!cuda_render_math::is_all_finite(value)) return false;
    affine_transform3<T> candidate;
    if(!cuda_render_math::try_inverse(value.linear, candidate.linear, max_residual)) return false;
    candidate.translation = -(candidate.linear * value.translation);
    if(!cuda_render_math::is_all_finite(candidate.translation)) return false;
    output = candidate;
    return true;
} 

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline affine_transform3<T>
inverse(affine_transform3<T> value) noexcept {
    affine_transform3<T> result;
    if (cuda_render_math::try_inverse(value, result)) return result;

    // 失敗した場合，全要素に qNaN を入れて返す
    const T invalid = detail::quiet_nan<T>();
    for(int i = 0; i < 9; ++i) result.linear.elements[i] = invalid;
    result.translation = vec3<T>(invalid);
    return result;
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr mat3<T>
normal_matrix_from_inverse(affine_transform3<T> inverse_transform) noexcept {
    return cuda_render_math::transpose(inverse_transform.linear);
}

template<real_scalar T>
[[nodiscard]] CUDA_RENDER_MATH_HD inline vec3<T>
transform_normal_from_inverse(affine_transform3<T> inverse_transform, vec3<T> normal) noexcept {
    return cuda_render_math::normal_matrix_from_inverse(inverse_transform) * normal;
}


} // namespace cuda_render_math