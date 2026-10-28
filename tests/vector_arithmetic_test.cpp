#include <cuda_render_math/vector.hpp>

#include "vector_test_support.hpp"

#include <cmath>
#include <concepts>
#include <limits>
#include <initializer_list>
#include <utility>

namespace rm = cuda_render_math;
namespace ts = test_support;

// Separate translation unit: exercise definitions from the public header twice.
rm::vec2f vector_linkage_sample(rm::vec2f) noexcept;
rm::vec2d vector_linkage_sample(rm::vec2d) noexcept;
rm::vec3f vector_linkage_sample(rm::vec3f) noexcept;
rm::vec3d vector_linkage_sample(rm::vec3d) noexcept;
rm::vec4f vector_linkage_sample(rm::vec4f) noexcept;
rm::vec4d vector_linkage_sample(rm::vec4d) noexcept;

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

template<typename V>
consteval bool check_arithmetic_contracts() {
    using T = ts::scalar_type<V>;
    using U = std::conditional_t<std::same_as<T, float>, double, float>;
    using Other = decltype(rm::vector_cast<U>(V{}));
    static_assert(std::same_as<decltype(-V{}), V>);
    static_assert(std::same_as<decltype(V{} + V{}), V>);
    static_assert(std::same_as<decltype(V{} - V{}), V>);
    static_assert(std::same_as<decltype(V{} * T(1)), V>);
    static_assert(std::same_as<decltype(T(1) * V{}), V>);
    static_assert(std::same_as<decltype(V{} / T(1)), V>);
    static_assert(std::same_as<decltype(std::declval<V&>() += V{}), V&>);
    static_assert(std::same_as<decltype(std::declval<V&>() -= V{}), V&>);
    static_assert(std::same_as<decltype(std::declval<V&>() *= T(1)), V&>);
    static_assert(std::same_as<decltype(std::declval<V&>() /= T(1)), V&>);
    static_assert(!can_add_assign<const V, V> && !can_subtract_assign<const V, V>);
    static_assert(!can_multiply_assign<const V, T> && !can_divide_assign<const V, T>);
    static_assert(!can_add<V, Other> && !can_subtract<V, Other>);
    static_assert(!can_add_assign<V, Other> && !can_subtract_assign<V, Other>);
    static_assert(!can_multiply<V, U> && !can_multiply<U, V>);
    static_assert(!can_divide<V, U>);
    static_assert(!can_multiply_assign<V, U> && !can_divide_assign<V, U>);
    static_assert(!can_multiply<V, int> && !can_multiply<int, V>);
    static_assert(!can_divide<V, int>);
    static_assert(!can_multiply_assign<V, int> && !can_divide_assign<V, int>);
    static_assert(!can_multiply<V, V> && !can_divide<V, V>);
    static_assert(!can_multiply_assign<V, V> && !can_divide_assign<V, V>);
    static_assert(noexcept(-V{}) && noexcept(V{} + V{}) && noexcept(V{} - V{}));
    static_assert(noexcept(V{} * T(1)) && noexcept(V{} / T(1)));
    static_assert(noexcept(std::declval<V&>() += V{}));
    return true;
}

template<typename V>
constexpr bool check_arithmetic_values() {
    using T = ts::scalar_type<V>;
    const V a = ts::components<V>(T(2), T(4), T(8), T(16));
    const V b = ts::components<V>(T(1), T(2), T(4), T(8));
    const V sum = ts::components<V>(T(3), T(6), T(12), T(24));
    const V twice_a = ts::components<V>(T(4), T(8), T(16), T(32));
    if (!ts::equal(-a, ts::components<V>(T(-2), T(-4), T(-8), T(-16))) ||
        !ts::equal(a + b, sum) || !ts::equal(a - b, b) ||
        !ts::equal(b * T(2), a) || !ts::equal(T(2) * b, a) ||
        !ts::equal(a / T(2), b)) return false;
    V value = a;
    if (&(value += b) != &value || !ts::equal(value, sum)) return false;
    if (&(value -= b) != &value || !ts::equal(value, a)) return false;
    if (&(value *= T(2)) != &value || !ts::equal(value, twice_a)) return false;
    if (&(value /= T(2)) != &value || !ts::equal(value, a)) return false;
    if (&(value += value) != &value || !ts::equal(value, twice_a)) return false;
    return &(value -= value) == &value && ts::equal(value, V{});
}

template<typename V>
consteval bool check_arithmetic() {
    return check_arithmetic_contracts<V>() && check_arithmetic_values<V>();
}
static_assert(check_arithmetic<rm::vec2f>());
static_assert(check_arithmetic<rm::vec2d>());
static_assert(check_arithmetic<rm::vec3f>());
static_assert(check_arithmetic<rm::vec3d>());
static_assert(check_arithmetic<rm::vec4f>());
static_assert(check_arithmetic<rm::vec4d>());

template<typename L, typename R>
consteval bool check_rejected_dimensions() {
    static_assert(!can_add<L, R> && !can_subtract<L, R>);
    static_assert(!can_add_assign<L, R> && !can_subtract_assign<L, R>);
    return true;
}

template<rm::real_scalar T>
consteval bool check_all_dimension_pairs() {
    return check_rejected_dimensions<rm::vec2<T>, rm::vec3<T>>() &&
           check_rejected_dimensions<rm::vec2<T>, rm::vec4<T>>() &&
           check_rejected_dimensions<rm::vec3<T>, rm::vec2<T>>() &&
           check_rejected_dimensions<rm::vec3<T>, rm::vec4<T>>() &&
           check_rejected_dimensions<rm::vec4<T>, rm::vec2<T>>() &&
           check_rejected_dimensions<rm::vec4<T>, rm::vec3<T>>();
}
static_assert(check_all_dimension_pairs<float>());
static_assert(check_all_dimension_pairs<double>());

template<typename V>
void check_runtime(ts::test_runner& runner) {
    using T = ts::scalar_type<V>;
    runner.check(check_arithmetic_values<V>(), "vector arithmetic and reference identity");
    for (T source : {std::numeric_limits<T>::denorm_min(), T(1e-30)}) {
        volatile T volatile_source = source;
        const T s = volatile_source;
        const V input = ts::components<V>(s, -s, s, -s);
        const V expected = ts::components<V>(T(1), T(-1), T(1), T(-1));
        const V divided = input / s;
        runner.check(ts::equal(divided, expected) &&
                     ts::all_components(divided, [](T x) { return std::isfinite(x); }),
                     "direct division keeps representable subnormal quotients");
        V assigned = input;
        assigned /= s;
        runner.check(ts::equal(assigned, expected), "compound direct division");
    }
    const V input = ts::components<V>(T(1), T(2), T(3), T(4));
    const V expected = ts::components<V>(T(2), T(4), T(6), T(8));
    runner.check(ts::equal(vector_linkage_sample(input), expected), "second translation unit");
}

} // namespace

int main() {
    ts::test_runner runner;
    check_runtime<rm::vec2f>(runner);
    check_runtime<rm::vec2d>(runner);
    check_runtime<rm::vec3f>(runner);
    check_runtime<rm::vec3d>(runner);
    check_runtime<rm::vec4f>(runner);
    check_runtime<rm::vec4d>(runner);
    return runner.finish("All vector arithmetic and linkage checks passed.");
}
