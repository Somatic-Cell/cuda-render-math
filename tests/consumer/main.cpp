#include <cuda_render_math/affine_transform.hpp>
#include <cuda_render_math/orthonormal_basis.hpp>
#include <cmath>

cuda_render_math::vec3f consumer_roundtrip(cuda_render_math::vec3f);
int main() {
    namespace rm = cuda_render_math;
    const rm::vec3f original{1.0f,2.0f,-3.0f};
    const auto result = consumer_roundtrip(original);
    return std::fabs(result.x-original.x) < 1e-5f &&
           std::fabs(result.y-original.y) < 1e-5f &&
           std::fabs(result.z-original.z) < 1e-5f ? 0 : 1;
}
