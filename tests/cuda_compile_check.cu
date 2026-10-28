#include <render_math/vector.hpp>

#include <cstddef>
#include <type_traits>

static_assert(std::is_trivially_copyable_v<render_math::Vec3f>);
static_assert(std::is_trivially_copyable_v<render_math::Vec3d>);
static_assert(std::is_standard_layout_v<render_math::Vec3f>);
static_assert(std::is_standard_layout_v<render_math::Vec3d>);
static_assert(sizeof(render_math::Vec3f) == 3 * sizeof(float));
static_assert(sizeof(render_math::Vec3d) == 3 * sizeof(double));
static_assert(alignof(render_math::Vec3f) == alignof(float));
static_assert(alignof(render_math::Vec3d) == alignof(double));
static_assert(offsetof(render_math::Vec3f, z) == 2 * sizeof(float));
static_assert(offsetof(render_math::Vec3d, z) == 2 * sizeof(double));

namespace {

template <render_math::Real T>
__device__ render_math::Vec3<T> exercise_operations(
    render_math::Vec3<T> a, render_math::Vec3<T> b, T scalar)
{
    render_math::Vec3<T> zero;
    render_math::Vec3<T> result = -a;
    result += b;
    result -= a;
    result *= scalar;
    result /= scalar;
    result = (result + a) - b;
    result = scalar * (result * scalar);
    return result / scalar + zero;
}

} // namespace

// Compile-only entry points: the test target does not launch kernels or link
// a host program against the CUDA Runtime API.
extern "C" __global__ void render_math_compile_float(
    render_math::Vec3f* output, render_math::Vec3f a,
    render_math::Vec3f b, float scalar)
{
    output[0] = exercise_operations(a, b, scalar);
}

extern "C" __global__ void render_math_compile_double(
    render_math::Vec3d* output, render_math::Vec3d a,
    render_math::Vec3d b, double scalar)
{
    output[0] = exercise_operations(a, b, scalar);
}

// Explicit precision conversion is exercised separately from the arithmetic paths.
extern "C" __global__ void render_math_compile_casts(
    render_math::Vec3f* float_output, render_math::Vec3d* double_output,
    render_math::Vec3f float_input, render_math::Vec3d double_input)
{
    float_output[0] = render_math::vector_cast<float>(double_input);
    double_output[0] = render_math::vector_cast<double>(float_input);
}
