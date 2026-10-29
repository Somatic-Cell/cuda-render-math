#include <cuda_render_math/vector_cast.hpp>
#include <cuda_render_math/vector_math.hpp>

#include <type_traits>

namespace rm = cuda_render_math;

namespace {
template<typename V>
__device__ void exercise_integer_vector(V* output, bool* valid, V a, V b) {
    using T = decltype(V{}.x);
    output[0] = V{};
        if constexpr (std::is_unsigned_v<T>) {
        output[1] = V(T(0)) - a;
    } else {
        output[1] = -a;
    }
    output[2] = a + b;
    output[3] = a - b;
    output[4] = a * T(2);
    output[5] = a / T(2);
    output[6] = rm::cwise_mul(a, b);
    output[7] = rm::cwise_div(a, V(T(1)));
    output[8] = (rm::min)(a, b);
    output[9] = (rm::max)(a, T(1));
    output[10] = rm::abs(a);
    output[11] = rm::clamp(a, T(0), T(7));
    V clamped{};
    valid[0] = rm::try_clamp(a, T(0), T(7), clamped);
    output[12] = clamped;
    output[13] = V(rm::dot(a, b));
    output[14] = V(rm::length_squared(a));
    V mutable_value = a;
    mutable_value += b; mutable_value -= b;
    mutable_value *= T(2); mutable_value /= T(2);
    rm::cwise_mul_assign(mutable_value, V(T(1)));
    rm::cwise_div_assign(mutable_value, V(T(1)));
    output[15] = mutable_value;
}

template<typename V>
__device__ void exercise_casts(V source, bool* flags) {
    using Float = decltype(rm::vector_cast<float>(source));
    using Double = decltype(rm::vector_cast<double>(source));
    using Signed = decltype(rm::vector_cast<std::int32_t>(source));
    using Unsigned = decltype(rm::vector_cast<std::uint32_t>(source));
    Float f{}; Double d{}; Signed i{}; Unsigned u{};
    flags[0] = rm::try_vector_cast(source, f);
    flags[1] = rm::try_vector_cast(source, d);
    flags[2] = rm::try_vector_cast(source, i);
    flags[3] = rm::try_vector_cast(source, u);
    // Instantiate the unchecked conversions only with a known in-range value.
    using T = decltype(V{}.x);
    f = rm::vector_cast<float>(V(T(1)));
    d = rm::vector_cast<double>(V(T(1)));
    i = rm::vector_cast<std::int32_t>(V(T(1)));
    u = rm::vector_cast<std::uint32_t>(V(T(1)));
    flags[4] = f.x == 1.0f && d.x == 1.0 && i.x == 1 && u.x == 1u;
}
} // namespace

// Compilation only; there are no CUDA Runtime API host calls.
// Not an execution harness. If launched, each output needs 16 elements and the
// caller must additionally respect the unchecked arithmetic preconditions.
extern "C" __global__ void cuda_render_math_compile_signed_vectors(
    rm::vec2i* out2, rm::vec3i* out3, rm::vec4i* out4, bool* flags,
    rm::vec2i a2, rm::vec3i a3, rm::vec4i a4) {
    exercise_integer_vector(out2, flags, a2, rm::vec2i(1));
    exercise_integer_vector(out3, flags + 1, a3, rm::vec3i(1));
    exercise_integer_vector(out4, flags + 2, a4, rm::vec4i(1));
}
extern "C" __global__ void cuda_render_math_compile_unsigned_vectors(
    rm::vec2u* out2, rm::vec3u* out3, rm::vec4u* out4, bool* flags,
    rm::vec2u a2, rm::vec3u a3, rm::vec4u a4) {
    exercise_integer_vector(out2, flags, a2, rm::vec2u(1u));
    exercise_integer_vector(out3, flags + 1, a3, rm::vec3u(1u));
    exercise_integer_vector(out4, flags + 2, a4, rm::vec4u(1u));
}
extern "C" __global__ void cuda_render_math_compile_numeric_casts(
    bool* flags, float f, double d, std::int32_t i, std::uint32_t u) {
    exercise_casts(rm::vec2f(f), flags);      exercise_casts(rm::vec3f(f), flags + 5);
    exercise_casts(rm::vec4f(f), flags + 10); exercise_casts(rm::vec2d(d), flags + 15);
    exercise_casts(rm::vec3d(d), flags + 20); exercise_casts(rm::vec4d(d), flags + 25);
    exercise_casts(rm::vec2i(i), flags + 30); exercise_casts(rm::vec3i(i), flags + 35);
    exercise_casts(rm::vec4i(i), flags + 40); exercise_casts(rm::vec2u(u), flags + 45);
    exercise_casts(rm::vec3u(u), flags + 50); exercise_casts(rm::vec4u(u), flags + 55);
}
