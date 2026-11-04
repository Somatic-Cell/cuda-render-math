// Parse headers after Windows-style macros, without pre-including standard headers.
#define min(a,b) error_min_macro_must_not_expand(a,b)
#define max(a,b) error_max_macro_must_not_expand(a,b)
#include <cuda_render_math/affine_transform.hpp>
#include <cuda_render_math/orthonormal_basis.hpp>
#undef min
#undef max

void cuda_render_math_transform_macro_check() {
    namespace rm = cuda_render_math;
    const rm::affine_transform3f t;
    const auto inv = rm::inverse(t);
    const auto b = rm::make_orthonormal_basis(rm::vec3f{0.0f,0.0f,1.0f});
    (void)rm::transform_normal_from_inverse(inv,rm::to_world(b,rm::vec3f(1.0f)));
}
