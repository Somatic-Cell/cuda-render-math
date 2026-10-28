#include <cuda_render_math/vector.hpp>

namespace rm = cuda_render_math;

namespace {

template<typename V>
V exercise_linkage(V value) noexcept {
    using T = decltype(value.x);
    const V other = value;
    value += other;
    value *= T(2);
    return value / T(2);
}

} // namespace

rm::vec2f vector_linkage_sample(rm::vec2f value) noexcept { return exercise_linkage(value); }
rm::vec2d vector_linkage_sample(rm::vec2d value) noexcept { return exercise_linkage(value); }
rm::vec3f vector_linkage_sample(rm::vec3f value) noexcept { return exercise_linkage(value); }
rm::vec3d vector_linkage_sample(rm::vec3d value) noexcept { return exercise_linkage(value); }
rm::vec4f vector_linkage_sample(rm::vec4f value) noexcept { return exercise_linkage(value); }
rm::vec4d vector_linkage_sample(rm::vec4d value) noexcept { return exercise_linkage(value); }
