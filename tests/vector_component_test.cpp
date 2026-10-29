#include <cuda_render_math/vector_math.hpp>
#include "vector_test_support.hpp"

#include <cmath>
#include <concepts>
#include <limits>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace rm = cuda_render_math;
namespace ts = test_support;

namespace {
template<typename L, typename R> concept can_add = requires(L a, R b) { a + b; };
template<typename L, typename R> concept can_sub = requires(L a, R b) { a - b; };
template<typename L, typename R> concept can_div = requires(L a, R b) { a / b; };
template<typename L, typename R> concept can_add_assign = requires(L a, R b) { a += b; };
template<typename L, typename R> concept can_sub_assign = requires(L a, R b) { a -= b; };
template<typename L, typename R> concept can_cmul = requires(L a, R b) { rm::cwise_mul(a, b); };
template<typename L, typename R> concept can_cdiv = requires(L a, R b) { rm::cwise_div(a, b); };
template<typename L, typename R> concept can_cmul_assign = requires(L a, R b) { rm::cwise_mul_assign(a, b); };
template<typename L, typename R> concept can_cdiv_assign = requires(L a, R b) { rm::cwise_div_assign(a, b); };
template<typename L, typename R> concept can_min = requires(L a, R b) { (rm::min)(a, b); };
template<typename L, typename R> concept can_max = requires(L a, R b) { (rm::max)(a, b); };
template<typename L, typename R> concept can_fmod = requires(L a, R b) { rm::fmod(a, b); };
template<typename V, typename S> concept can_clamp = requires(V v, S s) { rm::clamp(v, s, s); };
template<typename V, typename S> concept can_lerp = requires(V v, S s) { rm::lerp(v, v, s); };
template<typename V, typename S> concept can_smoothstep = requires(V v, S s) { rm::smoothstep(s, s, v); };

template<typename V, typename U>
consteval bool rejects_scalar() {
    static_assert(!can_add<V, U> && !can_add<U, V>);
    static_assert(!can_sub<V, U> && !can_sub<U, V> && !can_div<U, V>);
    static_assert(!can_add_assign<V, U> && !can_sub_assign<V, U>);
    static_assert(!can_min<V, U> && !can_min<U, V> && !can_max<V, U> && !can_max<U, V>);
    static_assert(!can_fmod<V, U> && !can_fmod<U, V>);
    static_assert(!can_clamp<V, U> && !can_lerp<V, U> && !can_smoothstep<V, U>);
    return true;
}

template<typename V>
constexpr bool check_arithmetic() {
    using T = ts::scalar_type<V>;
    const V a = ts::components<V>(T(2), T(4), T(8), T(16));
    const V b = ts::components<V>(T(1), T(2), T(4), T(8));
    if (!ts::equal(a + T(2), ts::components<V>(T(4), T(6), T(10), T(18)))) return false;
    if (!ts::equal(T(2) + a, a + T(2))) return false;
    if (!ts::equal(a - T(2), ts::components<V>(T(0), T(2), T(6), T(14)))) return false;
    if (!ts::equal(T(2) - a, -(a - T(2)))) return false;
    if (!ts::equal(T(16) / a, ts::components<V>(T(8), T(4), T(2), T(1)))) return false;
    if (!ts::equal(rm::cwise_mul(a, b), ts::components<V>(T(2), T(8), T(32), T(128)))) return false;
    if (!ts::equal(rm::cwise_div(a, b), V(T(2)))) return false;
    V v = a;
    if (&(v += T(2)) != &v || !ts::equal(v, a + T(2))) return false;
    if (&(v -= T(2)) != &v || !ts::equal(v, a)) return false;
    if (&rm::cwise_mul_assign(v, b) != &v || !ts::equal(v, rm::cwise_mul(a, b))) return false;
    if (&rm::cwise_div_assign(v, b) != &v || !ts::equal(v, a)) return false;
    if (&rm::cwise_mul_assign(v, v) != &v || !ts::equal(v, rm::cwise_mul(a, a))) return false;
    return &rm::cwise_div_assign(v, v) == &v && ts::equal(v, V(T(1)));
}

template<typename V>
consteval bool check_contracts() {
    using T = ts::scalar_type<V>;
    using U = std::conditional_t<std::same_as<T, float>, double, float>;
    using Other = decltype(rm::vector_cast<U>(V{}));
    static_assert(rejects_scalar<V, U>() && rejects_scalar<V, int>());
    static_assert(!can_cmul<V, Other> && !can_cdiv<V, Other>);
    static_assert(!can_cmul_assign<V, Other> && !can_cdiv_assign<V, Other>);
    static_assert(!can_cmul_assign<const V, V> && !can_cdiv_assign<const V, V>);
    static_assert(!can_add_assign<const V, T> && !can_sub_assign<const V, T>);
    static_assert(!can_min<V, Other> && !can_max<V, Other> && !can_fmod<V, Other>);
    static_assert(!can_clamp<V, Other> && !can_lerp<V, Other>);
    static_assert(std::same_as<decltype(V{} + T(1)), V> && std::same_as<decltype(T(1) / V{}), V>);
    static_assert(std::same_as<decltype(std::declval<V&>() += T(1)), V&>);
    static_assert(std::same_as<decltype(rm::cwise_mul_assign(std::declval<V&>(), V{})), V&>);
    static_assert(std::same_as<decltype(rm::clamp(V{}, T(0), T(1))), V>);
    static_assert(std::same_as<decltype(rm::lerp(V{}, V{}, T(0))), V>);
    static_assert(noexcept(rm::cwise_mul(V{}, V{})) && noexcept(rm::clamp(V{}, T(0), T(1))));
    return check_arithmetic<V>();
}
static_assert(check_contracts<rm::vec2f>() && check_contracts<rm::vec2d>());
static_assert(check_contracts<rm::vec3f>() && check_contracts<rm::vec3d>());
static_assert(check_contracts<rm::vec4f>() && check_contracts<rm::vec4d>());
static_assert(!can_cmul<rm::vec2f, rm::vec3f> && !can_cmul<rm::vec3f, rm::vec4f>);
static_assert(!can_cdiv<rm::vec4d, rm::vec2d>);

template<typename V, typename F>
V component_reference(V v, F f) {
    v.x = f(v.x); v.y = f(v.y);
    if constexpr (ts::component_count<V> >= 3) v.z = f(v.z);
    if constexpr (ts::component_count<V> == 4) v.w = f(v.w);
    return v;
}

template<typename V>
void check_runtime(ts::test_runner& r) {
    using T = ts::scalar_type<V>;
    const V v = ts::components<V>(T(-1.25), T(0.25), T(1.75), T(3.5));
    const V lower = ts::components<V>(T(-1), T(0), T(1), T(2));
    const V upper = ts::components<V>(T(0), T(1), T(2), T(3));
    r.check(check_arithmetic<V>(), "component arithmetic runtime");
    r.check(ts::equal(rm::abs(v), component_reference(v, [](T x) { return std::abs(x); })), "vector abs");
    r.check(ts::equal(rm::floor(v), component_reference(v, [](T x) { return std::floor(x); })), "vector floor");
    r.check(ts::equal(rm::ceil(v), component_reference(v, [](T x) { return std::ceil(x); })), "vector ceil");
    r.check(ts::equal(rm::fract(v), component_reference(v, [](T x) { return x - std::floor(x); })), "vector fract");
    r.check(ts::equal((rm::min)(v, T(1)), component_reference(v, [](T x) { return x < T(1) ? x : T(1); })), "vector-scalar min");
    r.check(ts::equal((rm::min)(T(1), v), (rm::min)(v, T(1))) &&
            ts::equal((rm::max)(T(1), v), (rm::max)(v, T(1))), "scalar-vector min/max");
    r.check(ts::equal((rm::max)(v, T(1)), component_reference(v, [](T x) { return x > T(1) ? x : T(1); })), "vector-scalar max");
    r.check(ts::equal((rm::min)(v, lower), ts::components<V>(T(-1.25), T(0), T(1), T(2))) &&
            ts::equal((rm::max)(v, lower), ts::components<V>(T(-1), T(0.25), T(1.75), T(3.5))), "vector-vector min/max");
    r.check(ts::equal(rm::clamp(v, lower, upper), ts::components<V>(T(-1), T(0.25), T(1.75), T(3))), "per-component clamp bounds");
    r.check(ts::equal(rm::clamp(v, T(0), T(1)), rm::saturate(v)) &&
            ts::equal(rm::saturate(v), ts::components<V>(T(0), T(0.25), T(1), T(1))), "scalar clamp bounds and saturate");
    r.check(ts::equal(rm::fmod(v, T(2)), component_reference(v, [](T x) { return std::fmod(x, T(2)); })) &&
            ts::equal(rm::fmod(T(2), v), component_reference(v, [](T x) { return std::fmod(T(2), x); })) &&
            ts::equal(rm::fmod(v, V(T(2))), rm::fmod(v, T(2))), "fmod operand combinations");
    r.check(ts::equal(rm::lerp(lower, upper, T(0.25)), lower + T(0.25)), "lerp scalar factor");
    const V t = ts::components<V>(T(0), T(0.25), T(0.5), T(1));
    r.check(ts::equal(rm::lerp(lower, upper, t), lower + t), "lerp component factors");
    const V curve = ts::components<V>(T(0), T(0.15625), T(0.5), T(1));
    r.check(ts::equal(rm::smoothstep(T(0), T(1), t), curve), "smoothstep scalar edges");
    r.check(ts::equal(rm::smoothstep(lower, upper, lower + t), curve), "smoothstep component edges");

    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T inf = std::numeric_limits<T>::infinity();
    V faulty = V(T(0.5)); faulty.y = nan;
    for (V result : {(rm::min)(faulty, V(T(1))), (rm::max)(faulty, V(T(0))),
                     rm::clamp(faulty, T(0), T(1)), rm::lerp(faulty, V(T(1)), T(0.5)),
                     rm::smoothstep(T(0), T(1), faulty)}) {
        r.check(std::isfinite(result.x) && std::isnan(result.y), "invalid component does not poison valid neighbor");
    }
    r.check(ts::all_components(rm::clamp(v, T(2), T(1)), [](T x) { return std::isnan(x); }), "reversed scalar bounds mark every component");
    V bad_upper = V(T(1)); bad_upper.y = T(-1);
    const V bad_clamp = rm::clamp(V(T(0.5)), V(T(0)), bad_upper);
    r.check(bad_clamp.x == T(0.5) && std::isnan(bad_clamp.y), "invalid component bounds");
    const V tails = rm::smoothstep(T(0), T(1), ts::components<V>(-inf, inf, T(0), T(1)));
    r.check(tails.x == T(0) && tails.y == T(1), "smoothstep infinite vector values");
    volatile T stored = std::numeric_limits<T>::denorm_min();
    const T tiny = stored;
    const V signs = ts::components<V>(tiny, -tiny, tiny, -tiny);
    const V expected = ts::components<V>(T(1), T(-1), T(1), T(-1));
    r.check(ts::equal(tiny / signs, expected) && ts::equal(rm::cwise_div(signs, V(tiny)), expected),
            "component division does not precompute an overflowing reciprocal");
    V assigned = signs;
    rm::cwise_div_assign(assigned, V(tiny));
    r.check(ts::equal(assigned, expected), "component division assignment subnormal regression");
}
} // namespace

int main() {
    ts::test_runner r;
    check_runtime<rm::vec2f>(r); check_runtime<rm::vec2d>(r);
    check_runtime<rm::vec3f>(r); check_runtime<rm::vec3d>(r);
    check_runtime<rm::vec4f>(r); check_runtime<rm::vec4d>(r);
    return r.finish("All vector component-operation checks passed.");
}
