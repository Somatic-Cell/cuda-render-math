#include <cuda_render_math/detail/rounded_arithmetic.hpp>
#include <cuda_render_math/scalar_numerics.hpp>
#include "vector_test_support.hpp"

#include <cfenv>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <limits>

namespace rm = cuda_render_math;
namespace rd = cuda_render_math::detail;
namespace ts = test_support;

namespace {
template<typename A, typename B>
concept can_add_rn = requires(A a, B b) { rd::add_rn(a, b); };
template<typename T>
concept can_round = requires(T a) {
    rd::add_rn(a, a); rd::sub_rn(a, a); rd::mul_rn(a, a); rd::fma_rn(a, a, a);
};
static_assert(can_round<float> && can_round<double>);
static_assert(!can_round<int> && !can_round<unsigned> && !can_round<long double>);
static_assert(!can_add_rn<float, double> && !can_add_rn<float, int>);
static_assert(std::same_as<decltype(rd::fma_rn(1.0f, 2.0f, 3.0f)), float>);
static_assert(std::same_as<decltype(rd::mul_rn(1.0, 2.0)), double>);
static_assert(noexcept(rd::add_rn(1.0f, 2.0f)) && noexcept(rd::sub_rn(1.0, 2.0)));

template<typename T>
T runtime_value(T v) { volatile T stored = v; return stored; }

template<rm::real_scalar T>
void check_rounding(ts::test_runner& r) {
    const int before = r.failures;
    const T one = runtime_value(T(1));
    const T eps = runtime_value(std::numeric_limits<T>::epsilon());
    const T half_ulp = eps / T(2);
    r.check(rd::add_rn(one, half_ulp) == one, "nearest-even addition rounds a tie to even");
    r.check(rd::add_rn(one + eps, half_ulp) == one + T(2) * eps,
            "nearest-even addition rounds an odd tie upwards");
    r.check(rd::sub_rn(-one, half_ulp) == -one, "nearest-even subtraction handles negative tie");

    // A TwoSum-shaped witness verifies the prescribed sequence of rounded
    // additions/subtractions. It is not a new public TwoSum implementation.
    const T small = eps / T(4);
    const T sum = rd::add_rn(one, small);
    const T virtual_small = rd::sub_rn(sum, one);
    const T error = rd::add_rn(rd::sub_rn(one, rd::sub_rn(sum, virtual_small)),
                               rd::sub_rn(small, virtual_small));
    r.check(sum == one && error == small, "separate rounding retains the addition residual");

    const T e = runtime_value(std::ldexp(T(1), std::same_as<T, float> ? -13 : -27));
    const T a = one + e, b = one - e;
    const T product = rd::mul_rn(a, b);
    const T residual = rd::fma_rn(a, b, -product);
    r.check(product == one && residual == -(e * e), "rounded product and FMA recover product residual");
    r.check(rd::add_rn(rd::mul_rn(a, b), -one) == T(0),
            "multiply followed by add must not contract");
    r.check(rd::fma_rn(a, b, -one) == -(e * e), "explicit FMA must remain fused");
    r.check(rm::fma(a, b, -one) == residual, "public fma uses the same fused operation");
    r.check(rm::difference_of_products(one, one, a, b) == e * e,
            "product difference uses the rounded residual");

    const T tiny = runtime_value(std::numeric_limits<T>::denorm_min());
    r.check(rd::add_rn(tiny, tiny) == T(2) * tiny &&
            rd::sub_rn(T(2) * tiny, tiny) == tiny &&
            rd::mul_rn(tiny, one) == tiny && rd::fma_rn(tiny, one, tiny) == T(2) * tiny,
            "gradual underflow is part of the test environment");
    const T neg_zero = runtime_value(-T(0));
    r.check(std::signbit(rd::mul_rn(neg_zero, one)), "rounded multiply preserves negative zero");
    const T high = runtime_value((std::numeric_limits<T>::max)());
    r.check(std::isinf(rd::mul_rn(high, T(2))), "standalone overflowing product is infinity");
    r.check(rd::fma_rn(high, T(2), -high) == high,
            "FMA avoids an intermediate product overflow");
    const T nan = runtime_value(std::numeric_limits<T>::quiet_NaN());
    r.check(std::isnan(rd::add_rn(nan, one)) && std::isnan(rd::sub_rn(one, nan)) &&
            std::isnan(rd::mul_rn(nan, one)) && std::isnan(rd::fma_rn(one, one, nan)),
            "rounded operations do not mask NaN");
    if (r.failures != before)
        std::fprintf(stderr, "  rounding scalar type: %s\n", std::same_as<T, float> ? "float" : "double");
}
} // namespace

int main() {
    ts::test_runner r;
    r.check(std::fegetround() == FE_TONEAREST, "tests require round-to-nearest environment");
    check_rounding<float>(r);
    check_rounding<double>(r);
    return r.finish("All rounded-arithmetic checks passed.");
}
