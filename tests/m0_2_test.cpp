#include <cuda_render_math/scalar_math.hpp>
#include <cuda_render_math/vector.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <utility>

namespace rm = cuda_render_math;

// Defined in a second translation unit, including both new and existing headers.
rm::vec2f m0_2_vector_odr_sample() noexcept;
float m0_2_scalar_odr_sample(float x) noexcept;

namespace {

template<typename L, typename R>
concept can_add = requires(L a, R b) { a + b; };
template<typename L, typename R>
concept can_subtract = requires(L a, R b) { a - b; };
template<typename L, typename R>
concept can_multiply = requires(L a, R b) { a * b; };
template<typename L, typename R>
concept can_divide = requires(L a, R b) { a / b; };
template<typename L, typename R>
concept can_add_assign = requires(L a, R b) { a += b; };
template<typename L, typename R>
concept can_subtract_assign = requires(L a, R b) { a -= b; };
template<typename L, typename R>
concept can_multiply_assign = requires(L a, R b) { a *= b; };
template<typename L, typename R>
concept can_divide_assign = requires(L a, R b) { a /= b; };
template<typename L, typename R>
concept can_dot = requires(L a, R b) { rm::dot(a, b); };
template<typename L, typename R>
concept can_cross = requires(L a, R b) { rm::cross(a, b); };
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
constexpr bool equal(rm::vec2<T> a, rm::vec2<T> b) noexcept {
    return a.x == b.x && a.y == b.y;
}
template<rm::real_scalar T>
constexpr bool equal(rm::vec3<T> a, rm::vec3<T> b) noexcept {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

template<rm::real_scalar T>
consteval bool check_vec2_contracts() {
    using V = rm::vec2<T>;
    static_assert(std::is_aggregate_v<V>);
    static_assert(std::is_trivially_copyable_v<V>);
    static_assert(std::is_standard_layout_v<V>);
    static_assert(sizeof(V) == 2 * sizeof(T));
    static_assert(alignof(V) == alignof(T));
    static_assert(offsetof(V, x) == 0);
    static_assert(offsetof(V, y) == sizeof(T));
    static_assert(std::same_as<decltype(-V{}), V>);
    static_assert(std::same_as<decltype(V{} + V{}), V>);
    static_assert(std::same_as<decltype(V{} - V{}), V>);
    static_assert(std::same_as<decltype(V{} * T(2)), V>);
    static_assert(std::same_as<decltype(T(2) * V{}), V>);
    static_assert(std::same_as<decltype(V{} / T(2)), V>);
    static_assert(std::same_as<decltype(std::declval<V&>() += V{}), V&>);
    static_assert(std::same_as<decltype(std::declval<V&>() -= V{}), V&>);
    static_assert(std::same_as<decltype(std::declval<V&>() *= T(2)), V&>);
    static_assert(std::same_as<decltype(std::declval<V&>() /= T(2)), V&>);
    static_assert(!can_add_assign<const V, V>);
    static_assert(!can_subtract_assign<const V, V>);
    static_assert(!can_multiply_assign<const V, T>);
    static_assert(!can_divide_assign<const V, T>);
    constexpr V zero;
    constexpr V partial{T(3)};
    return equal(zero, V{T(0), T(0)}) && equal(partial, V{T(3), T(0)});
}
static_assert(check_vec2_contracts<float>());
static_assert(check_vec2_contracts<double>());

template<rm::real_scalar T, rm::real_scalar U>
consteval bool check_mixed_vectors() {
    static_assert(!can_add<rm::vec2<T>, rm::vec2<U>>);
    static_assert(!can_subtract<rm::vec2<T>, rm::vec2<U>>);
    static_assert(!can_add_assign<rm::vec2<T>, rm::vec2<U>>);
    static_assert(!can_subtract_assign<rm::vec2<T>, rm::vec2<U>>);
    static_assert(!can_dot<rm::vec2<T>, rm::vec2<U>>);
    static_assert(!can_dot<rm::vec3<T>, rm::vec3<U>>);
    static_assert(!can_cross<rm::vec3<T>, rm::vec3<U>>);
    return true;
}
static_assert(check_mixed_vectors<float, double>());
static_assert(check_mixed_vectors<double, float>());

template<rm::real_scalar T, typename S>
consteval bool check_mixed_scalars() {
    static_assert(!can_multiply<rm::vec2<T>, S>);
    static_assert(!can_multiply<S, rm::vec2<T>>);
    static_assert(!can_divide<rm::vec2<T>, S>);
    static_assert(!can_multiply_assign<rm::vec2<T>, S>);
    static_assert(!can_divide_assign<rm::vec2<T>, S>);
    return true;
}
static_assert(check_mixed_scalars<float, double>());
static_assert(check_mixed_scalars<double, float>());
static_assert(check_mixed_scalars<float, int>());
static_assert(check_mixed_scalars<double, int>());
static_assert(!can_add<rm::vec2f, rm::vec3f>);
static_assert(!can_dot<rm::vec2f, rm::vec3f>);
static_assert(!can_multiply<rm::vec2f, rm::vec2f>);
static_assert(!can_divide<rm::vec2f, rm::vec2f>);
static_assert(!can_cross<rm::vec2f, rm::vec2f>);
static_assert(!std::is_convertible_v<rm::vec2f, rm::vec2d>);
static_assert(!std::is_convertible_v<rm::vec2d, rm::vec2f>);
static_assert(std::same_as<decltype(rm::vector_cast<double>(rm::vec2f{})), rm::vec2d>);
static_assert(std::same_as<decltype(rm::vector_cast<float>(rm::vec2d{})), rm::vec2f>);
static_assert(equal(rm::vector_cast<double>(rm::vec2f{0.5f, -2.0f}),
                   rm::vec2d{0.5, -2.0}));
static_assert(equal(rm::vector_cast<float>(rm::vec2d{0.5, -2.0}),
                   rm::vec2f{0.5f, -2.0f}));

template<rm::real_scalar T>
consteval bool check_vec2_arithmetic() {
    using V = rm::vec2<T>;
    const V a{T(2), T(4)};
    const V b{T(1), T(2)};
    if (!equal(-a, V{T(-2), T(-4)}) ||
        !equal(a + b, V{T(3), T(6)}) || !equal(a - b, b) ||
        !equal(b * T(2), a) || !equal(T(2) * b, a) || !equal(a / T(2), b)) {
        return false;
    }
    V v = a;
    if (&(v += b) != &v || !equal(v, V{T(3), T(6)})) return false;
    if (&(v -= b) != &v || !equal(v, a)) return false;
    if (&(v *= T(2)) != &v || !equal(v, V{T(4), T(8)})) return false;
    if (&(v /= T(2)) != &v || !equal(v, a)) return false;
    // The right operand is copied before the left operand is modified.
    if (&(v += v) != &v || !equal(v, V{T(4), T(8)})) return false;
    if (&(v -= v) != &v || !equal(v, V{})) return false;
    return true;
}
static_assert(check_vec2_arithmetic<float>());
static_assert(check_vec2_arithmetic<double>());

// Regression: compound assignment must return the actual left operand, not a copy.
template<rm::real_scalar T>
consteval bool check_vec3_reference_identity() {
    using V = rm::vec3<T>;
    V v{T(1), T(2), T(4)};
    const V b{T(1), T(1), T(1)};
    if (&(v += b) != &v || !equal(v, V{T(2), T(3), T(5)})) return false;
    if (&(v -= b) != &v || !equal(v, V{T(1), T(2), T(4)})) return false;
    if (&(v *= T(2)) != &v || !equal(v, V{T(2), T(4), T(8)})) return false;
    if (&(v /= T(2)) != &v || !equal(v, V{T(1), T(2), T(4)})) return false;
    if (&(v += v) != &v || !equal(v, V{T(2), T(4), T(8)})) return false;
    return &(v -= v) == &v && equal(v, V{});
}
static_assert(check_vec3_reference_identity<float>());
static_assert(check_vec3_reference_identity<double>());

template<rm::real_scalar T>
consteval bool check_vector_math() {
    using V2 = rm::vec2<T>;
    using V3 = rm::vec3<T>;
    static_assert(std::same_as<decltype(rm::dot(V2{}, V2{})), T>);
    static_assert(std::same_as<decltype(rm::dot(V3{}, V3{})), T>);
    static_assert(std::same_as<decltype(rm::length_squared(V2{})), T>);
    static_assert(std::same_as<decltype(rm::length_squared(V3{})), T>);
    static_assert(std::same_as<decltype(rm::cross(V3{}, V3{})), V3>);
    const V3 x{T(1), T(0), T(0)};
    const V3 y{T(0), T(1), T(0)};
    const V3 z{T(0), T(0), T(1)};
    const V3 a{T(1), T(2), T(3)};
    const V3 b{T(4), T(5), T(6)};
    return rm::dot(V2{T(1), T(2)}, V2{T(3), T(4)}) == T(11) &&
           rm::length_squared(V2{T(3), T(4)}) == T(25) &&
           rm::dot(a, b) == T(32) && rm::length_squared(a) == T(14) &&
           rm::length_squared(V3{}) == T(0) &&
           equal(rm::cross(x, y), z) && equal(rm::cross(y, z), x) &&
           equal(rm::cross(z, x), y) && equal(rm::cross(y, x), -z) &&
           equal(rm::cross(a, b), V3{T(-3), T(6), T(-3)}) &&
           equal(rm::cross(a, a), V3{}) && rm::dot(rm::cross(a, b), a) == T(0);
}
static_assert(check_vector_math<float>());
static_assert(check_vector_math<double>());

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
static_assert(!can_abs<int> && !can_abs<long double>);
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

template<rm::real_scalar T>
void check_vec2_runtime_division() {
    volatile T source = std::numeric_limits<T>::denorm_min();
    const T tiny = source;
    const rm::vec2<T> a{tiny, -tiny};
    check(equal(a / tiny, rm::vec2<T>{T(1), T(-1)}), "vec2 direct subnormal division");
    auto b = a;
    b /= tiny;
    check(equal(b, rm::vec2<T>{T(1), T(-1)}), "vec2 compound subnormal division");
}

template<rm::real_scalar T>
void check_vector_math_runtime() {
    // Small integer inputs make these identities exact in both supported types.
    // This tests signs, axes, and argument order without relying on tolerances.
    std::uint32_t state = 12345u;
    auto component = [&state]() -> T {
        state = state * 1664525u + 1013904223u;
        return static_cast<T>(static_cast<int>((state >> 16) % 17u) - 8);
    };
    for (int i = 0; i < 2048; ++i) {
        const rm::vec3<T> a{component(), component(), component()};
        const rm::vec3<T> b{component(), component(), component()};
        const auto c = rm::cross(a, b);
        check(rm::dot(c, a) == T(0) && rm::dot(c, b) == T(0), "cross orthogonality");
        check(equal(rm::cross(b, a), -c), "cross antisymmetry");
        check(rm::dot(a, b) == rm::dot(b, a), "dot symmetry for exact test values");
        const T ab = rm::dot(a, b);
        check(rm::length_squared(c) == rm::length_squared(a) * rm::length_squared(b) - ab * ab,
              "Lagrange identity for exact test values");
        const rm::vec2<T> a2{a.x, a.y};
        const rm::vec2<T> b2{b.x, b.y};
        check(rm::dot(a2, b2) == a.x * b.x + a.y * b.y, "vec2 dot runtime");
    }
}

} // namespace

int main() {
    check_scalar_math_runtime<float>();
    check_scalar_math_runtime<double>();
    check_vec2_runtime_division<float>();
    check_vec2_runtime_division<double>();
    check_vector_math_runtime<float>();
    check_vector_math_runtime<double>();
    check(equal(m0_2_vector_odr_sample(), rm::vec2f{3.0f, 6.0f}), "vec2 second translation unit");
    check(m0_2_scalar_odr_sample(-4.0f) == 3.0f, "scalar math second translation unit");
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return 1;
    }
    std::puts("All M0.2 contract checks passed.");
    return 0;
}
