#include <cuda_render_math/vector_geometry.hpp>
#include "vector_test_support.hpp"

#include <cmath>
#include <concepts>
#include <cstdio>
#include <limits>

namespace rm = cuda_render_math;
namespace ts = test_support;

namespace {
template<typename A, typename B>
concept can_cross_compensated = requires(A a, B b) { rm::cross_compensated(a, b); };
static_assert(can_cross_compensated<rm::vec3f, rm::vec3f>);
static_assert(can_cross_compensated<rm::vec3d, rm::vec3d>);
static_assert(!can_cross_compensated<rm::vec3f, rm::vec3d>);
static_assert(!can_cross_compensated<rm::vec2f, rm::vec2f>);
static_assert(!can_cross_compensated<rm::vec4f, rm::vec4f>);
static_assert(!can_cross_compensated<rm::vec3i, rm::vec3i>);

template<typename V>
void check_reflection(ts::test_runner& r) {
    const int failures_before = r.failures;
    using T = ts::scalar_type<V>;
    const V normal = ts::components<V>(T(0), T(1));
    const V incident = ts::components<V>(T(3), T(-4), T(1), T(-2));
    const V expected = ts::components<V>(T(3), T(4), T(1), T(-2));
    r.check(ts::equal(rm::reflect(incident, normal), expected), "incident-towards-surface convention");
    r.check(ts::equal(rm::reflect(incident, -normal), expected), "normal orientation does not change reflection");
    r.check(ts::equal(rm::reflect(rm::reflect(incident, normal), normal), incident),
            "axis reflection is an involution");
    const T hi = (std::numeric_limits<T>::max)();
    const V huge = ts::components<V>(T(0), hi);
    r.check(ts::equal(rm::reflect(huge, normal), ts::components<V>(T(0), -hi)),
            "reflection avoids overflowing twice the projection");
    volatile T denorm_source = std::numeric_limits<T>::denorm_min();
    const T tiny = denorm_source;
    r.check(ts::equal(rm::reflect(ts::components<V>(tiny, -tiny), normal),
                     ts::components<V>(tiny, tiny)), "reflection supports subnormal incident vectors");
    r.check(ts::equal(rm::reflect(ts::components<V>(tiny, -hi), normal),
                     ts::components<V>(tiny, hi)), "tiny tangent survives reflection of a huge incident vector");
    const V zeros = rm::reflect(V(-T(0)), normal);
    r.check(ts::all_components(zeros, [](T x) { return x == T(0) && std::signbit(x); }),
            "zero incident vector preserves zero signs");
    const V oblique = rm::normalize(ts::components<V>(T(1), T(2), T(-3), T(4)));
    const V reflected = rm::reflect(incident, oblique);
    const V twice = rm::reflect(reflected, oblique);
    const T tolerance = T(64) * std::numeric_limits<T>::epsilon() * rm::length(incident);
    r.check(rm::length(twice - incident) <= tolerance, "oblique reflection is an approximate involution");
    r.check(std::abs(rm::length(reflected) - rm::length(incident)) <= tolerance,
            "unit-normal reflection preserves incident length");
    r.check(std::abs(rm::dot(reflected, oblique) + rm::dot(incident, oblique)) <= tolerance,
            "reflection reverses the normal component");
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T inf = std::numeric_limits<T>::infinity();
    r.check(ts::all_components(rm::reflect(V(nan), normal), [](T x) { return std::isnan(x); }) &&
            ts::all_components(rm::reflect(incident, V(inf)), [](T x) { return std::isnan(x); }),
            "nonfinite reflection input is not concealed");
    if (r.failures != failures_before)
        std::fprintf(stderr, "  reflection type: vec%d%s\n", ts::component_count<V>,
                     std::same_as<T, float> ? "f" : "d");
}

template<rm::real_scalar T>
void check_cross(ts::test_runner& r) {
    const int exponent = std::same_as<T, float> ? -13 : -27;
    volatile T stored = std::ldexp(T(1), exponent);
    const T e = stored;
    const rm::vec3<T> a{T(1), T(1) + e, T(0)};
    const rm::vec3<T> b{T(1) - e, T(1), T(0)};
    const auto result = rm::cross_compensated(a, b);
    r.check(result.x == T(0) && result.y == T(0) && result.z == e * e,
            "compensated cross retains near-parallel residual");
    const auto reversed = rm::cross_compensated(b, a);
    r.check(reversed.z == -(e * e), "compensated cross has right-handed orientation");
    r.check(ts::equal(rm::cross_compensated(a, a), rm::vec3<T>{}), "self cross is zero");
    const rm::vec3<T> x{T(1), T(0), T(0)}, y{T(0), T(1), T(0)};
    r.check(ts::equal(rm::cross_compensated(x, y), rm::vec3<T>{T(0), T(0), T(1)}),
            "compensated cross basis orientation");
}
} // namespace

int main() {
    ts::test_runner r;
    check_reflection<rm::vec2f>(r); check_reflection<rm::vec2d>(r);
    check_reflection<rm::vec3f>(r); check_reflection<rm::vec3d>(r);
    check_reflection<rm::vec4f>(r); check_reflection<rm::vec4d>(r);
    check_cross<float>(r); check_cross<double>(r);
    return r.finish("All reflection and compensated-cross checks passed.");
}
