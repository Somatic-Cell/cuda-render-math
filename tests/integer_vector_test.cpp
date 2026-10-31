#include <cuda_render_math/vector_geometry.hpp>
#include <cuda_render_math/vector_math.hpp>
#include "vector_test_support.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace rm = cuda_render_math;
namespace ts = test_support;

namespace {
template<typename T> concept can_form_vector = requires { typename rm::vec3<T>; };
template<typename A, typename B> concept can_add = requires(A a, B b) { a + b; };
template<typename A, typename B> concept can_mul = requires(A a, B b) { a * b; };
template<typename A, typename B> concept can_div_assign = requires(A a, B b) { a /= b; };
template<typename V> concept has_lerp = requires(V v) { rm::lerp(v, v, decltype(v.x)(0)); };
template<typename V> concept has_sqrt = requires(V v) { rm::sqrt(v.x); };
template<typename V> concept has_fract = requires(V v) { rm::fract(v); };
template<typename V> concept has_cross = requires(V v) { rm::cross(v, v); };

static_assert(rm::integer_scalar<std::int32_t> && rm::integer_scalar<std::uint32_t>);
static_assert(!rm::real_scalar<std::int32_t> && !rm::real_scalar<std::uint32_t>);
static_assert(rm::supported_arithmetic<float> && rm::supported_arithmetic<double>);
static_assert(!can_form_vector<bool> && !can_form_vector<char>);
static_assert(!can_form_vector<std::int16_t> && !can_form_vector<std::uint16_t>);
static_assert(!can_form_vector<std::int64_t> && !can_form_vector<std::uint64_t>);
static_assert(!can_form_vector<const std::int32_t> && !can_form_vector<volatile std::uint32_t>);

template<typename V>
consteval bool check_type_contracts() {
    using T = ts::scalar_type<V>;
    using Opposite = std::conditional_t<std::same_as<T, std::int32_t>, std::uint32_t, std::int32_t>;
    using O = decltype(rm::vector_cast<Opposite>(V{}));
    using F = decltype(rm::vector_cast<float>(V{}));
    static_assert(!std::is_aggregate_v<V>);
    static_assert(std::is_trivially_copyable_v<V> && std::is_standard_layout_v<V>);
    static_assert(sizeof(V) == ts::component_count<V> * sizeof(T));
    static_assert(alignof(V) == alignof(T));
    static_assert(offsetof(V, x) == 0 && offsetof(V, y) == sizeof(T));
    if constexpr (ts::component_count<V> >= 3) static_assert(offsetof(V, z) == 2 * sizeof(T));
    if constexpr (ts::component_count<V> == 4) static_assert(offsetof(V, w) == 3 * sizeof(T));
    static_assert(!std::is_convertible_v<T, V> && !std::is_convertible_v<V, O>);
    static_assert(!std::is_constructible_v<V, O> && !std::is_constructible_v<V, F>);
    static_assert(!can_add<V, O> && !can_add<V, F> && !can_add<V, Opposite>);
    static_assert(!can_mul<V, V> && !can_mul<V, float> && !can_mul<V, Opposite>);
    static_assert(!can_div_assign<T, V> && !can_div_assign<const V, T>);
    static_assert(!has_lerp<V> && !has_sqrt<V> && !has_fract<V> && !has_cross<V>);
    static_assert(std::same_as<decltype(rm::dot(V{}, V{})), T>);
    static_assert(std::same_as<decltype(rm::length_squared(V{})), T>);
    static_assert(std::same_as<decltype(std::declval<V&>() += V{}), V&>);
    constexpr V zero;
    constexpr V splat{T(7)};
    return ts::all_components(zero, [](T x) { return x == T(0); }) &&
           ts::all_components(splat, [](T x) { return x == T(7); });
}

template<rm::integer_scalar T>
constexpr bool check_construction() {
    using V2 = rm::vec2<T>; using V3 = rm::vec3<T>; using V4 = rm::vec4<T>;
    const V2 a{T(1), T(2)};
    const V3 b{T(1), T(2), T(3)};
    const V4 c{T(1), T(2), T(3), T(4)};
    static_assert(!std::is_convertible_v<V2, V3> && !std::is_convertible_v<V4, V2>);
    return ts::equal(V2(b), a) && ts::equal(V2(c), a) && ts::equal(V3(c), b) &&
           ts::equal(V3(a), V3{T(1), T(2), T(0)}) &&
           ts::equal(V4(a), V4{T(1), T(2), T(0), T(0)}) &&
           ts::equal(V4(b), V4{T(1), T(2), T(3), T(0)}) &&
           ts::equal(V3(a, T(3)), b) && ts::equal(V3(T(1), V2{T(2), T(3)}), b) &&
           ts::equal(V4(b, T(4)), c) && ts::equal(V4(T(1), V3{T(2), T(3), T(4)}), c) &&
           ts::equal(V4(a, V2{T(3), T(4)}), c) && ts::equal(V4(a, T(3), T(4)), c) &&
           ts::equal(V4(T(1), V2{T(2), T(3)}, T(4)), c) &&
           ts::equal(V4(T(1), T(2), V2{T(3), T(4)}), c);
}

template<typename V>
constexpr bool check_values() {
    using T = ts::scalar_type<V>;
    const V a = ts::components<V>(T(8), T(12), T(16), T(20));
    const V b = ts::components<V>(T(2), T(3), T(4), T(5));
    if (!ts::equal(a / T(4), b) || !ts::equal(T(4) * b, a) ||
        !ts::equal(a - b, b * T(3)) || !ts::equal(a + b, b * T(5))) return false;
    if (!ts::equal(a + T(2), T(2) + a) || !ts::equal(a - T(2), a + T(0) - T(2))) return false;
    if (!ts::equal(T(24) / b, ts::components<V>(T(12), T(8), T(6), T(4)))) return false;
    if (!ts::equal(rm::cwise_div(a, b), V(T(4))) ||
        !ts::equal(rm::cwise_mul(b, b), ts::components<V>(T(4), T(9), T(16), T(25)))) return false;
    V v = a;
    if (&(v += b) != &v || !ts::equal(v, a + b)) return false;
    if (&(v -= b) != &v || !ts::equal(v, a)) return false;
    if (&(v += T(2)) != &v || &(v -= T(2)) != &v || !ts::equal(v, a)) return false;
    if (&(v *= T(2)) != &v || &(v /= T(2)) != &v || !ts::equal(v, a)) return false;
    if (&rm::cwise_mul_assign(v, b) != &v || &rm::cwise_div_assign(v, b) != &v ||
        !ts::equal(v, a)) return false;
    v = b;
    if (&rm::cwise_mul_assign(v, v) != &v || !ts::equal(v, rm::cwise_mul(b, b))) return false;
    if (&rm::cwise_div_assign(v, v) != &v || !ts::equal(v, V(T(1)))) return false;
    T expected = T(4 + 9);
    if constexpr (ts::component_count<V> >= 3) expected += T(16);
    if constexpr (ts::component_count<V> == 4) expected += T(25);
    if (rm::dot(b, b) != expected || rm::length_squared(b) != expected) return false;
    if (!ts::equal((rm::min)(a, b), b) || !ts::equal((rm::max)(a, b), a)) return false;
    if (!ts::equal((rm::min)(a, T(6)), V(T(6))) || !ts::equal((rm::max)(T(24), a), V(T(24)))) return false;
    const V bounded = ts::components<V>(T(3), T(3), T(4), T(4));
    if (!ts::equal(rm::clamp(b, T(3), T(4)), bounded)) return false;
    v = V(T(99));
    if (rm::try_clamp(b, T(5), T(4), v) || !ts::equal(v, V(T(99)))) return false;
    V upper(T(4)); upper.y = T(2);
    if (rm::try_clamp(b, V(T(3)), upper, v) || !ts::equal(v, V(T(99)))) return false;
    if (!rm::try_clamp(b, T(3), T(4), v) || !ts::equal(v, bounded)) return false;
    if (!rm::try_clamp(v, T(4), T(4), v) || !ts::equal(v, V(T(4)))) return false;
    if constexpr (std::same_as<T, std::int32_t>) {
        if (!ts::equal(-b, T(0) - b) || !ts::equal(rm::abs(-b), b)) return false;
        if (!ts::equal(V(T(-7)) / T(3), V(T(-2)))) return false;
        const T low = (std::numeric_limits<T>::lowest)();
        const T high = (std::numeric_limits<T>::max)();
        if (!ts::equal(V(high) - T(1), V(T(high - 1))) ||
            !ts::equal(V(low) + T(1), V(T(low + 1)))) return false;
    } else {
        const T high = (std::numeric_limits<T>::max)();
        if (!ts::equal(V(high) + T(1), V(T(0))) ||
            !ts::equal(V(T(0)) - T(1), V(high))) return false;

        if (!ts::equal(V(high) * T(2), V(T(high - 1))) ||
            !ts::equal(rm::abs(b), b)) return false;
    }
    return true;
}

template<typename V>
consteval bool check_all() { return check_type_contracts<V>() && check_values<V>(); }
static_assert(check_construction<std::int32_t>() && check_construction<std::uint32_t>());
static_assert(check_all<rm::vec2i>() && check_all<rm::vec3i>() && check_all<rm::vec4i>());
static_assert(check_all<rm::vec2u>() && check_all<rm::vec3u>() && check_all<rm::vec4u>());
static_assert(!can_add<rm::vec2i, rm::vec3i> && !can_add<rm::vec4u, rm::vec2u>);

// Volatile input forces runtime coverage in optimized builds.
template<typename V>
void check_runtime(ts::test_runner& r) {
    using T = ts::scalar_type<V>;
    volatile T source = T(7);
    const V v(source);
    r.check(ts::equal(v / T(3), V(T(2))), "runtime integer division truncates toward zero");
    r.check(check_values<V>(), "integer vector contracts");
}
} // namespace

int main() {
    ts::test_runner r;
    check_runtime<rm::vec2i>(r); check_runtime<rm::vec3i>(r); check_runtime<rm::vec4i>(r);
    check_runtime<rm::vec2u>(r); check_runtime<rm::vec3u>(r); check_runtime<rm::vec4u>(r);
    r.check(check_construction<std::int32_t>() && check_construction<std::uint32_t>(), "integer constructors");
    std::int32_t unchanged = 17;
    r.check(!rm::try_clamp(1, 2, 0, unchanged) && unchanged == 17, "scalar reversed clamp preserves output");
    return r.finish("All integer-vector checks passed.");
}
