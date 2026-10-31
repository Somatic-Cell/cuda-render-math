#include <cuda_render_math/vector_geometry.hpp>

namespace rm = cuda_render_math;

namespace {
template<typename V>
__device__ void exercise_geometry(V* vectors, decltype(V{}.x)* scalars,
                                 bool* valid, V input, V unit_normal) {
    using T = decltype(V{}.x);
    scalars[0] = rm::length(input);
    scalars[1] = rm::max_abs_component(input);
    valid[0] = rm::is_all_finite(input);
    vectors[0] = rm::normalize(input);
    vectors[1] = input;
    valid[1] = rm::try_normalize(input, vectors[1]);
    vectors[2] = input;
    valid[2] = rm::try_normalize(vectors[2], vectors[2]);
    vectors[3] = rm::reflect(input, unit_normal);
    vectors[4] = rm::reflect_from_outward(input, unit_normal);
    scalars[2] = rm::sqrt_clamped(input.x);
    scalars[3] = rm::asin_clamped(input.x);
    scalars[4] = rm::acos_clamped(input.x);
    scalars[5] = rm::difference_of_products(input.x, input.y, T(1), T(1));
}
} // namespace

// Compile-only entry points: no kernel launches and no Runtime API host calls.
// Per dimensionality, a launch would need 5 vectors, 6 scalars, and 3 booleans.
extern "C" __global__ void cuda_render_math_geometry_float(
    rm::vec2f* v2, rm::vec3f* v3, rm::vec4f* v4,
    float* scalars, bool* valid, rm::vec2f a2, rm::vec3f a3, rm::vec4f a4) {
    exercise_geometry(v2, scalars, valid, a2, rm::vec2f{0.0f, 1.0f});
    exercise_geometry(v3, scalars + 6, valid + 3, a3, rm::vec3f{0.0f, 1.0f, 0.0f});
    exercise_geometry(v4, scalars + 12, valid + 6, a4, rm::vec4f{0.0f, 1.0f, 0.0f, 0.0f});
}

extern "C" __global__ void cuda_render_math_geometry_double(
    rm::vec2d* v2, rm::vec3d* v3, rm::vec4d* v4,
    double* scalars, bool* valid, rm::vec2d a2, rm::vec3d a3, rm::vec4d a4) {
    exercise_geometry(v2, scalars, valid, a2, rm::vec2d{0.0, 1.0});
    exercise_geometry(v3, scalars + 6, valid + 3, a3, rm::vec3d{0.0, 1.0, 0.0});
    exercise_geometry(v4, scalars + 12, valid + 6, a4, rm::vec4d{0.0, 1.0, 0.0, 0.0});
}

extern "C" __global__ void cuda_render_math_compensated_cross(
    rm::vec3f* f, rm::vec3d* d, rm::vec3f a, rm::vec3f b, rm::vec3d c, rm::vec3d e) {
    f[0] = rm::cross(a, b);
    d[0] = rm::cross(c, e);
}

// Also instantiate host implementations during nvcc's host pass.
__host__ void cuda_render_math_geometry_host(rm::vec3f* output, rm::vec3f input) {
    *output = rm::normalize(input);
}
