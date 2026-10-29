#pragma once 

#include <cuda_render_math/scalar_cast.hpp>
#include <cuda_render_math/vector.hpp>

namespace cuda_render_math {

template<supported_arithmetic To, supported_arithmetic From>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr bool
try_vector_cast(vec2<From> value, vec2<To>& output) noexcept {
    vec2<To> converted {};
    if (
        !cuda_render_math::try_scalar_cast(value.x, converted.x) || 
        !cuda_render_math::try_scalar_cast(value.y, converted.y) 
    )
    {
        return false;
    }
    output = converted;
    return true;
}

template<supported_arithmetic To, supported_arithmetic From>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr bool
try_vector_cast(vec3<From> value, vec3<To>& output) noexcept {
    vec3<To> converted {};
    if (
        !cuda_render_math::try_scalar_cast(value.x, converted.x) || 
        !cuda_render_math::try_scalar_cast(value.y, converted.y) ||
        !cuda_render_math::try_scalar_cast(value.z, converted.z) 
    )
    {
        return false;
    }
    output = converted;
    return true;
}

template<supported_arithmetic To, supported_arithmetic From>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr bool
try_vector_cast(vec4<From> value, vec4<To>& output) noexcept {
    vec4<To> converted {};
    if (
        !cuda_render_math::try_scalar_cast(value.x, converted.x) || 
        !cuda_render_math::try_scalar_cast(value.y, converted.y) ||
        !cuda_render_math::try_scalar_cast(value.z, converted.z) ||
        !cuda_render_math::try_scalar_cast(value.w, converted.w) 
    )
    {
        return false;
    }
    output = converted;
    return true;
}

}