#include <cuda_render_math/scalar_numerics.hpp>
#include "vector_test_support.hpp"

#include <cmath>
#include <concepts>
#include <cstdio>
#include <limits>

namespace rm = cuda_render_math;
namespace ts = test_support;

namespace {
template<typename T>
concept has_domain_functions = requires(T x) {
    rm::sqrt_clamped(x); rm::asin_clamped(x); rm::acos_clamped(x);
};
template<typename A, typename B>
concept has_product_difference = requires(A a, B b) {
    rm::difference_of_products(a, b, a, b);
};
static_assert(has_domain_functions<float> && has_domain_functions<double>);
static_assert(!has_domain_functions<int> && !has_domain_functions<unsigned>);
static_assert(!has_domain_functions<long double>);
static_assert(!has_product_difference<float, double>);
static_assert(!has_product_difference<float, int>);
static_assert(std::same_as<decltype(rm::sqrt_clamped(1.0f)), float>);
static_assert(std::same_as<decltype(rm::acos_clamped(1.0)), double>);
static_assert(noexcept(rm::difference_of_products(1.0f, 1.0f, 1.0f, 1.0f)));

template<typename T>
T runtime_value(T v) { volatile T stored = v; return stored; }

template<rm::real_scalar T>
void check_values(ts::test_runner& r) {
    const int failures_before = r.failures;
    const T nan = runtime_value(std::numeric_limits<T>::quiet_NaN());
    const T inf = runtime_value(std::numeric_limits<T>::infinity());
    const T negative_zero = runtime_value(-T(0));
    const T tiny = runtime_value(std::numeric_limits<T>::denorm_min());
    const T eps = std::numeric_limits<T>::epsilon();
    r.check(rm::sqrt_clamped(T(4)) == T(2), "clamped sqrt interior");
    r.check(rm::sqrt_clamped(-tiny) == T(0) && rm::sqrt_clamped(T(-100)) == T(0),
            "finite negative sqrt inputs explicitly clamp");
    r.check(!std::signbit(rm::sqrt_clamped(T(-1))) &&
            std::signbit(rm::sqrt_clamped(negative_zero)), "sqrt signed-zero contract");
    r.check(rm::sqrt_clamped(inf) == inf && std::isnan(rm::sqrt_clamped(-inf)),
            "sqrt infinity policy");
    r.check(std::isnan(rm::sqrt_clamped(nan)) && std::isnan(rm::asin_clamped(nan)) &&
            std::isnan(rm::acos_clamped(nan)), "domain correction preserves NaN");
    r.check(std::isnan(rm::asin_clamped(inf)) && std::isnan(rm::asin_clamped(-inf)) &&
            std::isnan(rm::acos_clamped(inf)) && std::isnan(rm::acos_clamped(-inf)),
            "inverse trigonometric correction rejects infinities");
    r.check(rm::acos_clamped(T(2)) == T(0) && rm::acos_clamped(T(-2)) == std::acos(T(-1)),
            "acos finite clipping endpoints");
    r.check(rm::asin_clamped(T(2)) == std::asin(T(1)) &&
            rm::asin_clamped(T(-2)) == std::asin(T(-1)), "asin finite clipping endpoints");
    r.check(rm::acos_clamped(std::nextafter(T(1), inf)) == T(0) &&
            rm::asin_clamped(std::nextafter(T(-1), -inf)) == std::asin(T(-1)),
            "one-ULP domain excursion");
    r.check(std::signbit(rm::asin_clamped(negative_zero)), "asin preserves negative zero");
    r.check(std::abs(rm::acos_clamped(T(0.5)) - std::acos(T(0.5))) <= T(4) * eps,
            "acos interior accuracy");
    r.check(std::abs(rm::asin_clamped(T(-0.5)) - std::asin(T(-0.5))) <= T(4) * eps,
            "asin interior accuracy");

    // Exact representable cancellation case; no long-double oracle is needed.
    const int exponent = std::same_as<T, float> ? -13 : -27;
    const T e = runtime_value(std::ldexp(T(1), exponent));
    const T c = T(1) + e;
    const T d = T(1) - e;
    volatile T rounded_product = c * d;
    const T plain = T(1) - rounded_product;
    const T expected = e * e;
    r.check(plain == T(0), "chosen products lose residual when rounded separately");
    r.check(rm::difference_of_products(T(1), T(1), c, d) == expected,
            "compensated difference recovers positive residual");
    r.check(rm::difference_of_products(c, d, T(1), T(1)) == -expected,
            "compensated difference recovers negative residual");
    r.check(rm::difference_of_products(T(2), T(3), T(4), T(5)) == T(-14),
            "product difference ordinary inputs");
    r.check(rm::difference_of_products(c, d, c, d) == T(0),
            "equal products cancel after compensation");
    r.check(std::isnan(rm::difference_of_products(nan, T(1), T(1), T(1))),
            "product difference does not conceal NaN");
    if (r.failures != failures_before)
        std::fprintf(stderr, "  scalar type: %s\n", std::same_as<T, float> ? "float" : "double");
}
} // namespace

int main() {
    ts::test_runner r;
    check_values<float>(r);
    check_values<double>(r);
    return r.finish("All scalar numerical-contract checks passed.");
}
