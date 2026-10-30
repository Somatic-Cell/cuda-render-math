#include <cuda_render_math/vector_geometry.hpp>

cuda_render_math::vec3f geometry_linkage_sample(cuda_render_math::vec3f v) {
    const auto normal = cuda_render_math::vec3f{0.0f, 1.0f, 0.0f};
    return cuda_render_math::normalize(cuda_render_math::reflect(v, normal));
}
