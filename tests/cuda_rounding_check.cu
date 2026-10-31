#include <cuda_render_math/detail/rounded_arithmetic.hpp>
#include <cuda_render_math/scalar_numerics.hpp>

namespace rm = cuda_render_math;
namespace rd = cuda_render_math::detail;

namespace {
template<rm::real_scalar T>
__device__ void exercise_rounding(T* output, T a, T b, T c, T d) {
    output[0] = rd::add_rn(a, b);
    output[1] = rd::sub_rn(a, b);
    output[2] = rd::mul_rn(a, b);
    output[3] = rd::fma_rn(a, b, c);
    output[4] = rm::difference_of_products(a, b, c, d);
    output[5] = rd::add_rn(rd::mul_rn(a, b), c);
    output[6] = rm::fma(a, b, c);
}
} // namespace

// Compile-only coverage with runtime inputs and externally observable outputs.
// No launch, allocation, or CUDA Runtime API dependency. Each output needs 7 elements.
extern "C" __global__ void cuda_render_math_rounding_float(
    float* output, float a, float b, float c, float d) {
    exercise_rounding(output, a, b, c, d);
}
extern "C" __global__ void cuda_render_math_rounding_double(
    double* output, double a, double b, double c, double d) {
    exercise_rounding(output, a, b, c, d);
}

// Dedicated instruction-inspection witnesses: these must retain a rounded
// multiplication followed by a rounded addition even with --fmad=true.
extern "C" __global__ void cuda_render_math_separate_mul_add_float(
    float* output, float a, float b, float c) {
    *output = rd::add_rn(rd::mul_rn(a, b), c);
}
extern "C" __global__ void cuda_render_math_separate_mul_add_double(
    double* output, double a, double b, double c) {
    *output = rd::add_rn(rd::mul_rn(a, b), c);
}

__host__ void cuda_render_math_rounding_host(float* f, double* d, float x, double y) {
    *f = rd::add_rn(rd::mul_rn(x, x), rd::sub_rn(x, 1.0f));
    *d = rd::fma_rn(y, y, rd::sub_rn(y, 1.0));
}
