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
concept can_cross = requires(A a, B b) { rm::cross(a, b); };
static_assert(can_cross<rm::vec3f, rm::vec3f>);
static_assert(can_cross<rm::vec3d, rm::vec3d>);
static_assert(!can_cross<rm::vec3f, rm::vec3d>);
static_assert(!can_cross<rm::vec2f, rm::vec2f>);
static_assert(!can_cross<rm::vec4f, rm::vec4f>);
static_assert(!can_cross<rm::vec3i, rm::vec3i>);

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
    r.check(ts::equal(rm::reflect_from_outward(-incident, normal), expected),
            "outward adapter maps both surface directions outwards");
    r.check(ts::equal(rm::reflect_from_outward(normal, normal), normal),
            "normal incidence preserves outward normal direction");
    r.check(ts::equal(rm::reflect_from_outward(-incident, -normal), expected),
            "outward adapter is independent of normal orientation");
    r.check(ts::equal(rm::reflect_from_outward(-huge, normal),
                     rm::reflect(huge, normal)), "outward adapter preserves large-vector handling");
    r.check(ts::equal(rm::reflect_from_outward(
                         rm::reflect_from_outward(-incident, normal), normal), -incident),
            "outward reflection is an involution");
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
    r.check(ts::all_components(rm::reflect_from_outward(V(nan), normal),
                              [](T x) { return std::isnan(x); }) &&
            ts::all_components(rm::reflect_from_outward(incident, V(inf)),
                              [](T x) { return std::isnan(x); }),
            "outward adapter preserves nonfinite-input rejection");
    if (r.failures != failures_before)
        std::fprintf(stderr, "  reflection type: vec%d%s\n", ts::component_count<V>,
                     std::same_as<T, float> ? "f" : "d");
}

template<typename V>
void check_single_fma_update(ts::test_runner& r) {
    using T = ts::scalar_type<V>;
    std::uint32_t state = 0x614ef27du;
    auto component = [&state]() -> T {
        state = state * 1664525u + 1013904223u;
        return static_cast<T>(static_cast<int>(state >> 16) - 32768) / T(4096);
    };
    int old_formula_differences = 0;
    for (int sample = 0; sample < 1024; ++sample) {
        // Declarator initialization order fixes the random draw order.
        const T ix = component(), iy = component(), iz = component(), iw = component();
        const T nx = component(), ny = component(), nz = component(), nw = component();
        const V incident = ts::components<V>(ix, iy, iz, iw);
        V normal{};
        if (!rm::try_normalize(ts::components<V>(nx, ny, nz, nw), normal)) continue;
        const T projection = rm::dot(incident, normal);
        const V actual = rm::reflect(incident, normal);
        const V bsdf = rm::reflect_from_outward(-incident, normal);
        r.check(ts::equal(actual, bsdf), "both direction conventions select the same reflected ray");
        const auto check_component = [&](T i, T n, T out) {
            const T expected = std::fma(-projection, T(2) * n, i);
            const T old = std::fma(-projection, n, std::fma(-projection, n, i));
            if (expected != old) ++old_formula_differences;
            if (out != expected) {
                std::fprintf(stderr, "  vec%d%s sample %d: i=%a n=%a p=%a actual=%a expected=%a\n",
                    ts::component_count<V>, std::same_as<T, float> ? "f" : "d", sample,
                    static_cast<double>(i), static_cast<double>(n), static_cast<double>(projection),
                    static_cast<double>(out), static_cast<double>(expected));
            }
            r.check(out == expected, "reflection update matches one-rounding FMA for computed projection");
        };
        check_component(incident.x, normal.x, actual.x);
        check_component(incident.y, normal.y, actual.y);
        if constexpr (ts::component_count<V> >= 3) check_component(incident.z, normal.z, actual.z);
        if constexpr (ts::component_count<V> == 4) check_component(incident.w, normal.w, actual.w);
    }
    r.check(old_formula_differences > 0, "single-FMA regression set distinguishes the old two-FMA formula");
}

template<rm::real_scalar T>
void check_cross(ts::test_runner& r) {
    const int exponent = std::same_as<T, float> ? -13 : -27;
    volatile T stored = std::ldexp(T(1), exponent);
    const T e = stored;
    const rm::vec3<T> a{T(1), T(1) + e, T(0)};
    const rm::vec3<T> b{T(1) - e, T(1), T(0)};
    const auto plain = rm::cross_uncompensated(a, b);
    r.check(plain.z == T(0), "uncompensated cross loses the chosen residual in strict mode");
    const auto result = rm::cross(a, b);
    r.check(result.x == T(0) && result.y == T(0) && result.z == e * e,
            "compensated cross retains near-parallel residual");
    const auto reversed = rm::cross(b, a);
    r.check(reversed.z == -(e * e), "compensated cross has right-handed orientation");
    r.check(ts::equal(rm::cross(a, a), rm::vec3<T>{}), "self cross is zero");
    const rm::vec3<T> x{T(1), T(0), T(0)}, y{T(0), T(1), T(0)};
    r.check(ts::equal(rm::cross(x, y), rm::vec3<T>{T(0), T(0), T(1)}),
            "compensated cross basis orientation");
}
} // namespace

int main() {
    ts::test_runner r;
    check_reflection<rm::vec2f>(r); check_reflection<rm::vec2d>(r);
    check_reflection<rm::vec3f>(r); check_reflection<rm::vec3d>(r);
    check_reflection<rm::vec4f>(r); check_reflection<rm::vec4d>(r);
    check_single_fma_update<rm::vec2f>(r); check_single_fma_update<rm::vec2d>(r);
    check_single_fma_update<rm::vec3f>(r); check_single_fma_update<rm::vec3d>(r);
    check_single_fma_update<rm::vec4f>(r); check_single_fma_update<rm::vec4d>(r);
    check_cross<float>(r); check_cross<double>(r);
    return r.finish("All reflection and compensated-cross checks passed.");
}
