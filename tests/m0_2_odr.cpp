#include <cuda_render_math/vector.hpp>
#include <cuda_render_math/scalar_math.hpp>

namespace rm = cuda_render_math;

rm::vec2f m0_2_vector_odr_sample() noexcept {
    rm::vec2f value{1.0f, 2.0f};
    value += rm::vec2f{2.0f, 4.0f};
    return value;
}

float m0_2_scalar_odr_sample(float x) noexcept {
    const float root = rm::sqrt(rm::abs(x));
    return rm::isfinite(root) ? rm::fma(root, 2.0f, rm::copysign(1.0f, x)) : x;
}
