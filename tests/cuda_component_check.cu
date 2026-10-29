#include <cuda_render_math/vector_math.hpp>

namespace rm = cuda_render_math;

namespace {

template<typename V>
__device__ void exercise_components(V* out, V a, V b, decltype(V{}.x) t) {
    using T = decltype(V{}.x);
    out[0] = a + t;
    out[1] = t + a;
    out[2] = a - t;
    out[3] = t - a;
    out[4] = t / a;
    V value = a;
    out[5] = (value += t);
    out[6] = (value -= t);
    out[7] = rm::cwise_mul(a, b);
    out[8] = rm::cwise_div(a, b);
    out[9] = rm::cwise_mul_assign(value, b);
    out[10] = rm::cwise_div_assign(value, b);
    out[11] = rm::abs(a);
    out[12] = rm::floor(a);
    out[13] = rm::ceil(a);
    out[14] = rm::fract(a);
    out[15] = rm::fmod(a, b);
    out[16] = rm::fmod(a, t);
    out[17] = rm::fmod(t, a);
    out[18] = (rm::min)(a, b);
    out[19] = (rm::min)(a, t);
    out[20] = (rm::min)(t, a);
    out[21] = (rm::max)(a, b);
    out[22] = (rm::max)(a, t);
    out[23] = (rm::max)(t, a);
    out[24] = rm::clamp(a, b, V(t));
    out[25] = rm::clamp(a, T(0), t);
    out[26] = rm::saturate(a);
    out[27] = rm::lerp(a, b, t);
    out[28] = rm::lerp(a, b, V(t));
    out[29] = rm::smoothstep(T(0), t, a);
    out[30] = rm::smoothstep(a, b, V(t));
    V default_initialized;
    out[31] = default_initialized;
}

} // namespace

// Compilation only: no launches or dependency on Runtime API host entry points.
// Each vector output buffer would need 32 elements if this kernel were launched.
extern "C" __global__ void cuda_render_math_compile_components_float(
    rm::vec2f* out2, rm::vec3f* out3, rm::vec4f* out4,
    rm::vec2f a2, rm::vec2f b2, rm::vec3f a3, rm::vec3f b3,
    rm::vec4f a4, rm::vec4f b4, float t) {
    exercise_components(out2, a2, b2, t);
    exercise_components(out3, a3, b3, t);
    exercise_components(out4, a4, b4, t);
}

extern "C" __global__ void cuda_render_math_compile_components_double(
    rm::vec2d* out2, rm::vec3d* out3, rm::vec4d* out4,
    rm::vec2d a2, rm::vec2d b2, rm::vec3d a3, rm::vec3d b3,
    rm::vec4d a4, rm::vec4d b4, double t) {
    exercise_components(out2, a2, b2, t);
    exercise_components(out3, a3, b3, t);
    exercise_components(out4, a4, b4, t);
}

__host__ void cuda_render_math_default_construct_host(
    rm::vec2f* a, rm::vec3f* b, rm::vec4f* c,
    rm::vec2d* d, rm::vec3d* e, rm::vec4d* f) {
    *a = rm::vec2f{}; *b = rm::vec3f{}; *c = rm::vec4f{};
    *d = rm::vec2d{}; *e = rm::vec3d{}; *f = rm::vec4d{};
}
