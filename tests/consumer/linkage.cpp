#include <cuda_render_math/affine_transform.hpp>
#include <cuda_render_math/orthonormal_basis.hpp>

cuda_render_math::vec3f consumer_roundtrip(cuda_render_math::vec3f p) {
    namespace rm = cuda_render_math;
    const auto frame = rm::make_orthonormal_basis(rm::vec3f{1.0f,2.0f,3.0f});
    const rm::affine_transform3f transform{rm::to_matrix(frame), {4.0f,5.0f,6.0f}};
    rm::affine_transform3f inverted;
    if (!rm::try_inverse(transform,inverted)) return rm::vec3f(99.0f);
    return rm::transform_point(inverted,rm::transform_point(transform,p));
}
