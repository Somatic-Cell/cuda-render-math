#include <cuda_render_math/scalar_math.hpp>
#include <cuda_render_math/vector.hpp>

#include <cstddef>
#include <type_traits>

namespace rm = cuda_render_math;

namespace {

template<typename V, int Dimension>
consteval bool check_layout() {
    using T = decltype(V{}.x);
    static_assert(!std::is_aggregate_v<V>);
    static_assert(std::is_trivially_copyable_v<V> && std::is_standard_layout_v<V>);
    static_assert(sizeof(V) == Dimension * sizeof(T));
    static_assert(alignof(V) == alignof(T));
    static_assert(offsetof(V, x) == 0);
    static_assert(offsetof(V, y) == sizeof(T));
    if constexpr (Dimension >= 3) static_assert(offsetof(V, z) == 2 * sizeof(T));
    if constexpr (Dimension == 4) static_assert(offsetof(V, w) == 3 * sizeof(T));
    return true;
}
static_assert(check_layout<rm::vec2f, 2>() && check_layout<rm::vec2d, 2>());
static_assert(check_layout<rm::vec3f, 3>() && check_layout<rm::vec3d, 3>());
static_assert(check_layout<rm::vec4f, 4>() && check_layout<rm::vec4d, 4>());

template<typename V>
__device__ V exercise_operations(V a, V b, decltype(V{}.x) s) {
    V zero;
    V result = -a;
    result += b;
    result -= a;
    result *= s;
    result /= s;
    result = (result + a) - b;
    result = s * (result * s);
    return result / s + zero;
}

template<rm::real_scalar T>
__device__ void exercise_construction(
    rm::vec2<T>* out2, rm::vec3<T>* out3, rm::vec4<T>* out4,
    rm::vec2<T> a2, rm::vec3<T> a3, rm::vec4<T> a4, T s) {
    // Distinct output elements keep each constructor result observable.
    out2[0] = rm::vec2<T>();
    out2[1] = rm::vec2<T>(s);
    out2[2] = rm::vec2<T>{s};
    out2[3] = rm::vec2<T>(a2.x, s);
    out2[4] = rm::vec2<T>(a3);
    out2[5] = rm::vec2<T>(a4);
    out3[0] = rm::vec3<T>();
    out3[1] = rm::vec3<T>(s);
    out3[2] = rm::vec3<T>{s};
    out3[3] = rm::vec3<T>(a3.x, a3.y, s);
    out3[4] = rm::vec3<T>(a2);
    out3[5] = rm::vec3<T>(a2, s);
    out3[6] = rm::vec3<T>(s, a2);
    out3[7] = rm::vec3<T>(a4);
    out4[0] = rm::vec4<T>();
    out4[1] = rm::vec4<T>(s);
    out4[2] = rm::vec4<T>{s};
    out4[3] = rm::vec4<T>(a4.x, a4.y, a4.z, s);
    out4[4] = rm::vec4<T>(a2);
    out4[5] = rm::vec4<T>(a3);
    out4[6] = rm::vec4<T>(a3, s);
    out4[7] = rm::vec4<T>(s, a3);
    out4[8] = rm::vec4<T>(a2, rm::vec2<T>(a4));
    out4[9] = rm::vec4<T>(a2, a4.z, s);
    out4[10] = rm::vec4<T>(a4.x, a2, s);
    out4[11] = rm::vec4<T>(a4.x, s, a2);
}

template<rm::real_scalar T>
__device__ void exercise_math(T* output, rm::vec3<T>* cross_output,
    rm::vec2<T> a2, rm::vec2<T> b2, rm::vec3<T> a3, rm::vec3<T> b3,
    rm::vec4<T> a4, rm::vec4<T> b4, T s) {
    output[0] = rm::dot(a2, b2);
    output[1] = rm::dot(a3, b3);
    output[2] = rm::dot(a4, b4);
    output[3] = rm::length_squared(a2);
    output[4] = rm::length_squared(a3);
    output[5] = rm::length_squared(a4);
    output[6] = rm::abs(s);
    output[7] = rm::sqrt(rm::abs(s));
    output[8] = rm::fma(a2.x, b2.y, s);
    output[9] = rm::copysign(a2.x, s);
    output[10] = rm::isfinite(s) ? T(1) : T(0);
    cross_output[0] = rm::cross(a3, b3);
}

} // namespace

// Compile-only entry points: no kernel launch or CUDA Runtime API linkage.

extern "C" __global__ void cuda_render_math_compile_float(
    rm::vec2f* out2, rm::vec3f* out3, rm::vec4f* out4,
    rm::vec2f a2, rm::vec2f b2, rm::vec3f a3, rm::vec3f b3,
    rm::vec4f a4, rm::vec4f b4, float s) {
    out2[0] = exercise_operations(a2, b2, s);
    out3[0] = exercise_operations(a3, b3, s);
    out4[0] = exercise_operations(a4, b4, s);
}

extern "C" __global__ void cuda_render_math_compile_construction_float(
    rm::vec2f* out2, rm::vec3f* out3, rm::vec4f* out4,
    rm::vec2f a2, rm::vec3f a3, rm::vec4f a4, float s) {
    exercise_construction(out2, out3, out4, a2, a3, a4, s);
}

extern "C" __global__ void cuda_render_math_compile_math_float(
    float* output, rm::vec3f* cross_output,
    rm::vec2f a2, rm::vec2f b2, rm::vec3f a3, rm::vec3f b3,
    rm::vec4f a4, rm::vec4f b4, float s) {
    exercise_math(output, cross_output, a2, b2, a3, b3, a4, b4, s);
}

extern "C" __global__ void cuda_render_math_compile_double(
    rm::vec2d* out2, rm::vec3d* out3, rm::vec4d* out4,
    rm::vec2d a2, rm::vec2d b2, rm::vec3d a3, rm::vec3d b3,
    rm::vec4d a4, rm::vec4d b4, double s) {
    out2[0] = exercise_operations(a2, b2, s);
    out3[0] = exercise_operations(a3, b3, s);
    out4[0] = exercise_operations(a4, b4, s);
}

extern "C" __global__ void cuda_render_math_compile_construction_double(
    rm::vec2d* out2, rm::vec3d* out3, rm::vec4d* out4,
    rm::vec2d a2, rm::vec3d a3, rm::vec4d a4, double s) {
    exercise_construction(out2, out3, out4, a2, a3, a4, s);
}

extern "C" __global__ void cuda_render_math_compile_math_double(
    double* output, rm::vec3d* cross_output,
    rm::vec2d a2, rm::vec2d b2, rm::vec3d a3, rm::vec3d b3,
    rm::vec4d a4, rm::vec4d b4, double s) {
    exercise_math(output, cross_output, a2, b2, a3, b3, a4, b4, s);
}

extern "C" __global__ void cuda_render_math_compile_casts(
    rm::vec2f* out2f, rm::vec2d* out2d, rm::vec3f* out3f,
    rm::vec3d* out3d, rm::vec4f* out4f, rm::vec4d* out4d,
    rm::vec2f in2f, rm::vec2d in2d, rm::vec3f in3f,
    rm::vec3d in3d, rm::vec4f in4f, rm::vec4d in4d) {
    out2f[0] = rm::vector_cast<float>(in2d);
    out2d[0] = rm::vector_cast<double>(in2f);
    out3f[0] = rm::vector_cast<float>(in3d);
    out3d[0] = rm::vector_cast<double>(in3f);
    out4f[0] = rm::vector_cast<float>(in4d);
    out4d[0] = rm::vector_cast<double>(in4f);
}
