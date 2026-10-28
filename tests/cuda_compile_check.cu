#include <cuda_render_math/scalar_math.hpp>
#include <cuda_render_math/vector.hpp>

#include <cstddef>
#include <type_traits>

namespace rm = cuda_render_math;

namespace {

template<rm::real_scalar T>
consteval bool check_layouts() {
    using V2 = rm::vec2<T>;
    using V3 = rm::vec3<T>;
    static_assert(std::is_trivially_copyable_v<V2> && std::is_standard_layout_v<V2>);
    static_assert(std::is_trivially_copyable_v<V3> && std::is_standard_layout_v<V3>);
    static_assert(sizeof(V2) == 2 * sizeof(T));
    static_assert(sizeof(V3) == 3 * sizeof(T));
    static_assert(alignof(V2) == alignof(T) && alignof(V3) == alignof(T));
    static_assert(offsetof(V2, y) == sizeof(T));
    static_assert(offsetof(V3, z) == 2 * sizeof(T));
    return true;
}
static_assert(check_layouts<float>());
static_assert(check_layouts<double>());

template<rm::real_scalar T>
__device__ rm::vec2<T> exercise_operations(rm::vec2<T> a, rm::vec2<T> b, T s) {
    rm::vec2<T> zero;
    rm::vec2<T> result = -a;
    result += b;
    result -= a;
    result *= s;
    result /= s;
    result = (result + a) - b;
    result = s * (result * s);
    return result / s + zero;
}

template<rm::real_scalar T>
__device__ rm::vec3<T> exercise_operations(rm::vec3<T> a, rm::vec3<T> b, T s) {
    rm::vec3<T> zero;
    rm::vec3<T> result = -a;
    result += b;
    result -= a;
    result *= s;
    result /= s;
    result = (result + a) - b;
    result = s * (result * s);
    return result / s + zero;
}

template<rm::real_scalar T>
__device__ void exercise_math(T* output, rm::vec3<T>* cross_output,
                              rm::vec2<T> a2, rm::vec2<T> b2,
                              rm::vec3<T> a3, rm::vec3<T> b3, T s) {
    // Separate outputs keep every function's result observable to the compiler.
    output[0] = rm::dot(a2, b2);
    output[1] = rm::dot(a3, b3);
    output[2] = rm::length_squared(a2);
    output[3] = rm::length_squared(a3);
    output[4] = rm::abs(s);
    output[5] = rm::sqrt(rm::abs(s));
    output[6] = rm::fma(a2.x, b2.y, s);
    output[7] = rm::copysign(a2.x, s);
    output[8] = rm::isfinite(s) ? T(1) : T(0);
    cross_output[0] = rm::cross(a3, b3);
}

} // namespace

// Compile-only entry points: no kernel launch and no CUDA Runtime API linkage.
extern "C" __global__ void cuda_render_math_compile_float(
    rm::vec2f* output2, rm::vec3f* output3,
    rm::vec2f a2, rm::vec2f b2, rm::vec3f a3, rm::vec3f b3, float s) {
    output2[0] = exercise_operations(a2, b2, s);
    output3[0] = exercise_operations(a3, b3, s);
}

extern "C" __global__ void cuda_render_math_compile_double(
    rm::vec2d* output2, rm::vec3d* output3,
    rm::vec2d a2, rm::vec2d b2, rm::vec3d a3, rm::vec3d b3, double s) {
    output2[0] = exercise_operations(a2, b2, s);
    output3[0] = exercise_operations(a3, b3, s);
}

extern "C" __global__ void cuda_render_math_compile_math_float(
    float* output, rm::vec3f* cross_output,
    rm::vec2f a2, rm::vec2f b2, rm::vec3f a3, rm::vec3f b3, float s) {
    exercise_math(output, cross_output, a2, b2, a3, b3, s);
}

extern "C" __global__ void cuda_render_math_compile_math_double(
    double* output, rm::vec3d* cross_output,
    rm::vec2d a2, rm::vec2d b2, rm::vec3d a3, rm::vec3d b3, double s) {
    exercise_math(output, cross_output, a2, b2, a3, b3, s);
}

extern "C" __global__ void cuda_render_math_compile_casts(
    rm::vec2f* float2_output, rm::vec2d* double2_output,
    rm::vec3f* float3_output, rm::vec3d* double3_output,
    rm::vec2f float2_input, rm::vec2d double2_input,
    rm::vec3f float3_input, rm::vec3d double3_input) {
    float2_output[0] = rm::vector_cast<float>(double2_input);
    double2_output[0] = rm::vector_cast<double>(float2_input);
    float3_output[0] = rm::vector_cast<float>(double3_input);
    double3_output[0] = rm::vector_cast<double>(float3_input);
}
