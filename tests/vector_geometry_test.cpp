#include <cuda_render_math/vector_geometry.hpp>

#include "vector_test_support.hpp"

#include <concepts>
#include <cstdint>

namespace rm = cuda_render_math;
namespace ts = test_support;

namespace {

template<typename L, typename R>
concept can_dot = requires(L a, R b) { rm::dot(a, b); };
template<typename L, typename R>
concept can_cross = requires(L a, R b) { rm::cross(a, b); };
static_assert(std::same_as<decltype(rm::cross(rm::vec3f{}, rm::vec3f{})), rm::vec3f>);
static_assert(std::same_as<decltype(rm::cross(rm::vec3d{}, rm::vec3d{})), rm::vec3d>);

template<typename V, typename W>
consteval bool check_rejected_dot() {
    static_assert(!can_dot<V, W>);
    return true;
}

template<rm::real_scalar T>
consteval bool check_rejected_geometry() {
    using U = std::conditional_t<std::same_as<T, float>, double, float>;
    static_assert(!can_cross<rm::vec2<T>, rm::vec2<T>>);
    static_assert(!can_cross<rm::vec4<T>, rm::vec4<T>>);
    static_assert(!can_cross<rm::vec3<T>, rm::vec3<U>>);
    return check_rejected_dot<rm::vec2<T>, rm::vec2<U>>() &&
           check_rejected_dot<rm::vec3<T>, rm::vec3<U>>() &&
           check_rejected_dot<rm::vec4<T>, rm::vec4<U>>() &&
           check_rejected_dot<rm::vec2<T>, rm::vec3<T>>() &&
           check_rejected_dot<rm::vec2<T>, rm::vec4<T>>() &&
           check_rejected_dot<rm::vec3<T>, rm::vec2<T>>() &&
           check_rejected_dot<rm::vec3<T>, rm::vec4<T>>() &&
           check_rejected_dot<rm::vec4<T>, rm::vec2<T>>() &&
           check_rejected_dot<rm::vec4<T>, rm::vec3<T>>();
}
static_assert(check_rejected_geometry<float>());
static_assert(check_rejected_geometry<double>());

template<rm::real_scalar T>
consteval bool check_vector_math() {
    using V2 = rm::vec2<T>;
    using V3 = rm::vec3<T>;
    using V4 = rm::vec4<T>;
    static_assert(std::same_as<decltype(rm::dot(V2{}, V2{})), T>);
    static_assert(std::same_as<decltype(rm::dot(V3{}, V3{})), T>);
    static_assert(std::same_as<decltype(rm::length_squared(V2{})), T>);
    static_assert(std::same_as<decltype(rm::length_squared(V3{})), T>);
    static_assert(std::same_as<decltype(rm::cross_uncompensated(V3{}, V3{})), V3>);
    static_assert(std::same_as<decltype(rm::dot(V4{}, V4{})), T>);
    static_assert(std::same_as<decltype(rm::length_squared(V4{})), T>);
    const V3 x{T(1), T(0), T(0)};
    const V3 y{T(0), T(1), T(0)};
    const V3 z{T(0), T(0), T(1)};
    const V3 a{T(1), T(2), T(3)};
    const V3 b{T(4), T(5), T(6)};
    return rm::dot(V4{T(1), T(2), T(3), T(4)}, V4{T(5), T(6), T(7), T(8)}) == T(70) &&
           rm::length_squared(V4{T(1), T(2), T(3), T(4)}) == T(30) &&
           rm::length_squared(V4{}) == T(0) &&
           rm::dot(V2{T(1), T(2)}, V2{T(3), T(4)}) == T(11) &&
           rm::length_squared(V2{T(3), T(4)}) == T(25) &&
           rm::dot(a, b) == T(32) && rm::length_squared(a) == T(14) &&
           rm::length_squared(V3{}) == T(0) &&
           ts::equal(rm::cross_uncompensated(x, y), z) && ts::equal(rm::cross_uncompensated(y, z), x) &&
           ts::equal(rm::cross_uncompensated(z, x), y) && ts::equal(rm::cross_uncompensated(y, x), -z) &&
           ts::equal(rm::cross_uncompensated(a, b), V3{T(-3), T(6), T(-3)}) &&
           ts::equal(rm::cross_uncompensated(a, a), V3{}) && rm::dot(rm::cross_uncompensated(a, b), a) == T(0);
}
static_assert(check_vector_math<float>());
static_assert(check_vector_math<double>());

template<rm::real_scalar T>
void check_vector_math_runtime(ts::test_runner& runner) {
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
        runner.check(rm::dot(c, a) == T(0) && rm::dot(c, b) == T(0), "cross orthogonality");
        runner.check(ts::equal(rm::cross(b, a), -c), "cross antisymmetry");
        runner.check(rm::dot(a, b) == rm::dot(b, a), "dot symmetry for exact test values");
        const T ab = rm::dot(a, b);
        runner.check(rm::length_squared(c) == rm::length_squared(a) * rm::length_squared(b) - ab * ab,
              "Lagrange identity for exact test values");
        const rm::vec4<T> a4(a, component());
        const rm::vec4<T> b4(b, component());
        runner.check(rm::dot(a4, b4) == a.x * b.x + a.y * b.y + a.z * b.z + a4.w * b4.w,
                     "vec4 dot runtime includes w");
        runner.check(rm::length_squared(a4) == a.x * a.x + a.y * a.y + a.z * a.z + a4.w * a4.w,
                     "vec4 squared length includes w");
        const rm::vec2<T> a2{a.x, a.y};
        const rm::vec2<T> b2{b.x, b.y};
        runner.check(rm::dot(a2, b2) == a.x * b.x + a.y * b.y, "vec2 dot runtime");
    }
}

} // namespace

int main() {
    ts::test_runner runner;
    check_vector_math_runtime<float>(runner);
    check_vector_math_runtime<double>(runner);
    return runner.finish("All vector geometric operation checks passed.");
}
