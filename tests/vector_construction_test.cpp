#include <cuda_render_math/vector.hpp>

#include "vector_test_support.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace rm = cuda_render_math;
namespace ts = test_support;

namespace {

static_assert(rm::real_scalar<float> && rm::real_scalar<double>);
static_assert(!rm::real_scalar<int> && !rm::real_scalar<long double>);
static_assert(!rm::real_scalar<const float> && !rm::real_scalar<volatile double>);

template<typename V>
void accept_vector(V);

template<typename V>
concept accepts_scalar_copy_list = requires(ts::scalar_type<V> s) {
    accept_vector<V>({s});
};

template<typename V>
consteval bool check_construction_and_layout() {
    using T = ts::scalar_type<V>;
    static_assert(!std::is_aggregate_v<V>);
    static_assert(std::is_trivially_copyable_v<V>);
    static_assert(std::is_standard_layout_v<V>);
    static_assert(std::is_trivially_copy_constructible_v<V>);
    static_assert(std::is_trivially_copy_assignable_v<V>);
    static_assert(std::is_trivially_destructible_v<V>);
    static_assert(std::is_nothrow_default_constructible_v<V>);
    static_assert(std::is_nothrow_constructible_v<V, T>);
    static_assert(std::is_nothrow_move_constructible_v<V>);
    static_assert(sizeof(V) == ts::component_count<V> * sizeof(T));
    static_assert(alignof(V) == alignof(T));
    static_assert(offsetof(V, x) == 0);
    static_assert(offsetof(V, y) == sizeof(T));
    if constexpr (ts::component_count<V> >= 3) {
        static_assert(offsetof(V, z) == 2 * sizeof(T));
    }
    if constexpr (ts::component_count<V> == 4) {
        static_assert(offsetof(V, w) == 3 * sizeof(T));
    }
    static_assert(!std::is_convertible_v<T, V>);
    static_assert(!accepts_scalar_copy_list<V>);
    // Scalar arguments use ordinary C++ conversions to T at construction.
    // This is distinct from mixed-precision vector arithmetic, which is rejected.
    static_assert(std::is_constructible_v<V, int>);
    static_assert(std::is_constructible_v<V, double>);

    constexpr V default_initialized;
    constexpr V value_initialized{};
    constexpr V copy_list_initialized = {};
    constexpr V splat_parentheses(T(3));
    constexpr V splat_braces{T(3)};
    const V expected_splat = ts::components<V>(T(3), T(3), T(3), T(3));
    if (!ts::all_components(default_initialized, [](T x) { return x == T(0); }) ||
        !ts::equal(value_initialized, default_initialized) ||
        !ts::equal(copy_list_initialized, default_initialized) ||
        !ts::equal(splat_parentheses, expected_splat) ||
        !ts::equal(splat_braces, expected_splat)) return false;

    const V a = ts::components<V>(T(1), T(2), T(3), T(4));
    V copy = a;
    V assigned;
    assigned = copy;
    V moved = std::move(copy);
    V move_assigned;
    move_assigned = std::move(assigned);
    return ts::equal(moved, a) && ts::equal(move_assigned, a);
}
static_assert(check_construction_and_layout<rm::vec2f>());
static_assert(check_construction_and_layout<rm::vec2d>());
static_assert(check_construction_and_layout<rm::vec3f>());
static_assert(check_construction_and_layout<rm::vec3d>());
static_assert(check_construction_and_layout<rm::vec4f>());
static_assert(check_construction_and_layout<rm::vec4d>());

template<rm::real_scalar T>
consteval bool check_component_and_dimension_constructors() {
    using V2 = rm::vec2<T>;
    using V3 = rm::vec3<T>;
    using V4 = rm::vec4<T>;
    const V2 xy{T(1), T(2)};
    const V2 yz{T(2), T(3)};
    const V2 zw{T(3), T(4)};
    const V3 xyz{T(1), T(2), T(3)};
    const V3 yzw{T(2), T(3), T(4)};
    const V4 xyzw{T(1), T(2), T(3), T(4)};
    // Every partition of a component list into scalars and smaller vectors.
    if (!ts::equal(V2(T(1), T(2)), xy) ||
        !ts::equal(V3(T(1), T(2), T(3)), xyz) ||
        !ts::equal(V4(T(1), T(2), T(3), T(4)), xyzw) ||
        !ts::equal(V3(xy, T(3)), xyz) ||
        !ts::equal(V3(T(1), yz), xyz) ||
        !ts::equal(V4(xyz, T(4)), xyzw) ||
        !ts::equal(V4(T(1), yzw), xyzw) ||
        !ts::equal(V4(xy, zw), xyzw) ||
        !ts::equal(V4(xy, T(3), T(4)), xyzw) ||
        !ts::equal(V4(T(1), yz, T(4)), xyzw) ||
        !ts::equal(V4(T(1), T(2), zw), xyzw)) return false;

    // No default-filled component lists such as vec3(x,y) or vec4(x,y,z).
    static_assert(!std::is_constructible_v<V3, T, T>);
    static_assert(!std::is_constructible_v<V4, T, T>);
    static_assert(!std::is_constructible_v<V4, T, T, T>);
    static_assert(!std::is_constructible_v<V4, V2, T>);
    static_assert(!std::is_convertible_v<V2, V3>);
    static_assert(!std::is_convertible_v<V2, V4>);
    static_assert(!std::is_convertible_v<V3, V2>);
    static_assert(!std::is_convertible_v<V3, V4>);
    static_assert(!std::is_convertible_v<V4, V2>);
    static_assert(!std::is_convertible_v<V4, V3>);
    static_assert(std::is_nothrow_constructible_v<V2, V3>);
    static_assert(std::is_nothrow_constructible_v<V2, V4>);
    static_assert(std::is_nothrow_constructible_v<V3, V2>);
    static_assert(std::is_nothrow_constructible_v<V3, V4>);
    static_assert(std::is_nothrow_constructible_v<V4, V2>);
    static_assert(std::is_nothrow_constructible_v<V4, V3>);

    // w=4 proves truncation is NOT a perspective divide.
    return ts::equal(V2(xyz), xy) && ts::equal(V2{xyzw}, xy) &&
           ts::equal(V3(xyzw), xyz) &&
           ts::equal(V3(xy), V3{T(1), T(2), T(0)}) &&
           ts::equal(V4(xy), V4{T(1), T(2), T(0), T(0)}) &&
           ts::equal(V4(xyz), V4{T(1), T(2), T(3), T(0)}) &&
           ts::equal(V4{xyz, T(1)}, V4{T(1), T(2), T(3), T(1)});
}
static_assert(check_component_and_dimension_constructors<float>());
static_assert(check_component_and_dimension_constructors<double>());

template<typename V, typename U2, typename U3, typename U4>
consteval bool check_rejected_vector_precision() {
    static_assert(!std::is_constructible_v<V, U2>);
    static_assert(!std::is_constructible_v<V, U3>);
    static_assert(!std::is_constructible_v<V, U4>);
    static_assert(!std::is_convertible_v<U2, V>);
    static_assert(!std::is_convertible_v<U3, V>);
    static_assert(!std::is_convertible_v<U4, V>);
    using T = ts::scalar_type<V>;
    static_assert(!std::is_constructible_v<V, U2, T>);
    static_assert(!std::is_constructible_v<V, U2, T, T>);
    static_assert(!std::is_constructible_v<V, U3, T>);
    return true;
}
static_assert(check_rejected_vector_precision<rm::vec2f, rm::vec2d, rm::vec3d, rm::vec4d>());
static_assert(check_rejected_vector_precision<rm::vec3f, rm::vec2d, rm::vec3d, rm::vec4d>());
static_assert(check_rejected_vector_precision<rm::vec4f, rm::vec2d, rm::vec3d, rm::vec4d>());
static_assert(check_rejected_vector_precision<rm::vec2d, rm::vec2f, rm::vec3f, rm::vec4f>());
static_assert(check_rejected_vector_precision<rm::vec3d, rm::vec2f, rm::vec3f, rm::vec4f>());
static_assert(check_rejected_vector_precision<rm::vec4d, rm::vec2f, rm::vec3f, rm::vec4f>());

template<typename V, typename W>
consteval bool check_precision_cast() {
    using T = ts::scalar_type<V>;
    using U = ts::scalar_type<W>;
    const V a = ts::components<V>(T(0.5), T(-2), T(4), T(-8));
    const W b = ts::components<W>(U(0.5), U(-2), U(4), U(-8));
    static_assert(std::same_as<decltype(rm::vector_cast<U>(a)), W>);
    static_assert(noexcept(rm::vector_cast<U>(a)));
    return ts::equal(rm::vector_cast<U>(a), b) &&
           ts::equal(rm::vector_cast<T>(a), a);
}
static_assert(check_precision_cast<rm::vec2f, rm::vec2d>());
static_assert(check_precision_cast<rm::vec2d, rm::vec2f>());
static_assert(check_precision_cast<rm::vec3f, rm::vec3d>());
static_assert(check_precision_cast<rm::vec3d, rm::vec3f>());
static_assert(check_precision_cast<rm::vec4f, rm::vec4d>());
static_assert(check_precision_cast<rm::vec4d, rm::vec4f>());

template<typename V>
void check_runtime_splat(ts::test_runner& runner) {
    using T = ts::scalar_type<V>;
    volatile T negative_zero_source = -T(0);
    volatile T nan_source = std::numeric_limits<T>::quiet_NaN();
    volatile T inf_source = std::numeric_limits<T>::infinity();
    const T negative_zero = negative_zero_source;
    const T nan = nan_source;
    const T inf = inf_source;
    runner.check(ts::all_components(V(negative_zero), [](T x) {
        return x == T(0) && std::signbit(x);
    }), "splat preserves negative zero in every component");
    runner.check(ts::all_components(V{nan}, [](T x) { return std::isnan(x); }),
                 "splat preserves NaN classification");
    runner.check(ts::all_components(V(inf), [inf](T x) { return x == inf; }),
                 "splat preserves positive infinity");
}

} // namespace

int main() {
    ts::test_runner runner;
    runner.check(check_construction_and_layout<rm::vec2f>(), "vec2f construction");
    runner.check(check_construction_and_layout<rm::vec2d>(), "vec2d construction");
    runner.check(check_construction_and_layout<rm::vec3f>(), "vec3f construction");
    runner.check(check_construction_and_layout<rm::vec3d>(), "vec3d construction");
    runner.check(check_construction_and_layout<rm::vec4f>(), "vec4f construction");
    runner.check(check_construction_and_layout<rm::vec4d>(), "vec4d construction");
    check_runtime_splat<rm::vec2f>(runner);
    check_runtime_splat<rm::vec2d>(runner);
    check_runtime_splat<rm::vec3f>(runner);
    check_runtime_splat<rm::vec3d>(runner);
    check_runtime_splat<rm::vec4f>(runner);
    check_runtime_splat<rm::vec4d>(runner);
    return runner.finish("All vector construction and layout checks passed.");
}
