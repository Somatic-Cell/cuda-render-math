#include <cuda_render_math/vector.hpp>

cuda_render_math::vec3f vector_odr_sample() noexcept
{
    cuda_render_math::vec3f value{1.0f, 2.0f, 3.0f};
    value += cuda_render_math::vec3f{1.0f, 2.0f, 3.0f};
    return (value * 2.0f) / 2.0f;
}
