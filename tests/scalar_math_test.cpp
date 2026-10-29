#include <cuda_render_math/scalar_math.hpp>

#include <cmath>
#include <concepts>
#include <cstdio>
#include <limits>

namespace rm = cuda_render_math;

float scalar_math_linkage_sample(float x) noexcept;

namespace {

template<typename T>
concept can_sqrt = requires(T x) { rm::sqrt(x); };
template<typename T>
concept can_abs = requires(T x) { rm::abs(x); };
template<typename T>
concept can_isfinite = requires(T x) { rm::isfinite(x); };
template<typename A, typename B, typename C>
concept can_fma = requires(A a, B b, C c) { rm::fma(a, b, c); };
template<typename A, typename B>
concept can_copysign = requires(A a, B b) { rm::copysign(a, b); };

template<rm::real_scalar T>
consteval bool check_scalar_result_types() {
    static_assert(std::same_as<decltype(rm::abs(T(1))), T>);
    static_assert(std::same_as<decltype(rm::sqrt(T(1))), T>);
    static_assert(std::same_as<decltype(rm::fma(T(1), T(2), T(3))), T>);
    static_assert(std::same_as<decltype(rm::copysign(T(1), T(-1))), T>);
    static_assert(std::same_as<decltype(rm::isfinite(T(1))), bool>);
    static_assert(noexcept(rm::sqrt(T(1))));
    static_assert(noexcept(rm::fma(T(1), T(2), T(3))));
    return true;
}
static_assert(check_scalar_result_types<float>());
static_assert(check_scalar_result_types<double>());
static_assert(!can_sqrt<int> && !can_sqrt<long double>);
static_assert(can_abs<std::int32_t> && can_abs<std::uint32_t>);
static_assert(!can_abs<bool> && !can_abs<std::int64_t> && !can_abs<long double>);
static_assert(!can_isfinite<int> && !can_isfinite<long double>);
static_assert(!can_fma<float, double, float>);
static_assert(!can_fma<double, float, double>);
static_assert(!can_fma<float, float, double>);
static_assert(!can_fma<double, double, float>);
static_assert(!can_fma<int, float, float>);
static_assert(!can_copysign<float, double>);
static_assert(!can_copysign<double, float>);
static_assert(!can_copysign<float, int>);

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

template<rm::real_scalar T>
void check_scalar_math_runtime() {
    static_assert(std::numeric_limits<T>::is_iec559);
    volatile T four_source = T(4);
    volatile T zero_source = T(0);
    volatile T negative_source = T(-4);
    volatile T infinity_source = std::numeric_limits<T>::infinity();
    volatile T nan_source = std::numeric_limits<T>::quiet_NaN();
    const T four = four_source;
    const T zero = zero_source;
    const T negative = negative_source;
    const T inf = infinity_source;
    const T nan = nan_source;
    const T negative_zero = std::copysign(zero, T(-1));
    check(rm::sqrt(four) == T(2), "sqrt(4)");
    check(rm::sqrt(inf) == inf, "sqrt(+Inf)");
    check(std::isnan(rm::sqrt(negative)), "sqrt(negative) is not silently clamped");
    check(std::isnan(rm::sqrt(nan)), "sqrt(NaN)");
    check(std::signbit(rm::sqrt(negative_zero)), "sqrt preserves -0");
    check(rm::abs(negative) == four, "abs(-4)");
    check(rm::abs(-inf) == inf, "abs(-Inf)");
    check(!std::signbit(rm::abs(negative_zero)), "abs(-0) produces +0");
    check(std::isnan(rm::abs(nan)), "abs(NaN)");
    check(rm::copysign(four, negative_zero) == -four, "copysign from -0");
    check(rm::copysign(negative, four) == four, "copysign replaces sign");
    check(std::signbit(rm::copysign(zero, negative)), "copysign produces -0");
    check(rm::copysign(inf, negative) == -inf, "copysign(-Inf)");
    check(std::isnan(rm::copysign(nan, negative)), "copysign preserves NaN classification");
    check(rm::isfinite(zero) && rm::isfinite(negative_zero), "zeros are finite");
    check(rm::isfinite(std::numeric_limits<T>::max()), "max is finite");
    check(rm::isfinite(std::numeric_limits<T>::denorm_min()), "subnormal is finite");
    check(!rm::isfinite(inf) && !rm::isfinite(-inf), "infinities are not finite");
    check(!rm::isfinite(nan), "NaN is not finite");
    check(rm::fma(T(2), four, T(1)) == T(9), "fma(2,4,1)");
    check(std::isnan(rm::fma(nan, T(1), zero)), "fma propagates NaN");

    // (1+e)*(1-e)-1 = -e^2. A separately rounded product loses this residual.
    const T e = std::numeric_limits<T>::epsilon();
    volatile T x_source = T(1) + e;
    volatile T y_source = T(1) - e;
    const T x = x_source;
    const T y = y_source;
    check(rm::fma(x, y, T(-1)) == -(e * e), "fma retains the fused residual");

    // Broad, deterministic sqrt inputs, with the same-precision host reference.
    for (int exponent = -100; exponent <= 100; ++exponent) {
        volatile T input_source = std::ldexp(T(1.25), exponent);
        const T input = input_source;
        check(rm::sqrt(input) == std::sqrt(input), "sqrt matches host reference");
    }
}

} // namespace

int main() {
    check_scalar_math_runtime<float>();
    check_scalar_math_runtime<double>();
    check(scalar_math_linkage_sample(-4.0f) == 3.0f, "scalar math second translation unit");
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return 1;
    }
    std::puts("All scalar math checks passed.");
    return 0;
}
