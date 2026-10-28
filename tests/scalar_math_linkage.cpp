#include <cuda_render_math/scalar_math.hpp>

float scalar_math_linkage_sample(float x) noexcept {
    namespace rm = cuda_render_math;
    return rm::sqrt(rm::abs(x)) + rm::fma(0.5f, 2.0f, 0.0f);
}
