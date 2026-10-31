#include <cuda_render_math/vector_geometry.hpp>
#include "vector_test_support.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

namespace rm = cuda_render_math;
namespace ts = test_support;

// Defined in another translation unit to exercise header-only linkage.
rm::vec3f geometry_linkage_sample(rm::vec3f v);

namespace {
template<typename V> concept can_length = requires(V v) { rm::length(v); };
template<typename V> concept can_normalize = requires(V v) { rm::normalize(v); };
template<typename V, typename O> concept can_try_normalize = requires(V v, O& out) {
    rm::try_normalize(v, out);
};
template<typename V, typename N>
concept can_reflect_from_outward = requires(V v, N n) { rm::reflect_from_outward(v, n); };
static_assert(!can_reflect_from_outward<rm::vec3f, rm::vec3d>);
static_assert(!can_reflect_from_outward<rm::vec2f, rm::vec3f>);
static_assert(!can_reflect_from_outward<rm::vec3i, rm::vec3i>);
template<typename V, typename N> concept can_reflect = requires(V v, N n) { rm::reflect(v, n); };
static_assert(!can_length<rm::vec2i> && !can_normalize<rm::vec3u>);
static_assert(!can_length<rm::vec4i> && !can_normalize<float>);
static_assert(!can_try_normalize<rm::vec3f, rm::vec3d>);
static_assert(!can_try_normalize<rm::vec3f, const rm::vec3f>);
static_assert(!can_try_normalize<rm::vec2f, rm::vec3f>);
static_assert(!can_reflect<rm::vec3f, rm::vec3d> && !can_reflect<rm::vec3u, rm::vec3u>);

template<typename V>
consteval bool check_signature() {
    using T = ts::scalar_type<V>;
    static_assert(std::same_as<decltype(rm::length(V{})), T>);
    static_assert(std::same_as<decltype(rm::max_abs_component(V{})), T>);
    static_assert(std::same_as<decltype(rm::is_all_finite(V{})), bool>);
    static_assert(std::same_as<decltype(rm::normalize(V{})), V>);
    static_assert(std::same_as<decltype(rm::reflect(V{}, V{})), V>);
    static_assert(std::same_as<decltype(rm::reflect_from_outward(V{}, V{})), V>);
    static_assert(noexcept(rm::reflect_from_outward(V{}, V{})));
    static_assert(can_try_normalize<V, V>);
    static_assert(noexcept(rm::length(V{})) && noexcept(rm::normalize(V{})));
    return true;
}
static_assert(check_signature<rm::vec2f>() && check_signature<rm::vec2d>());
static_assert(check_signature<rm::vec3f>() && check_signature<rm::vec3d>());
static_assert(check_signature<rm::vec4f>() && check_signature<rm::vec4d>());

template<typename V>
std::array<ts::scalar_type<V>, 4> unpack(V v) {
    std::array<ts::scalar_type<V>, 4> a{v.x, v.y, 0, 0};
    if constexpr (ts::component_count<V> >= 3) a[2] = v.z;
    if constexpr (ts::component_count<V> == 4) a[3] = v.w;
    return a;
}

template<typename T>
bool same_bits(T a, T b) {
    using U = std::conditional_t<std::same_as<T, float>, std::uint32_t, std::uint64_t>;
    return std::bit_cast<U>(a) == std::bit_cast<U>(b);
}

template<typename V>
bool same_vector_bits(V a, V b) {
    const auto aa = unpack(a), bb = unpack(b);
    for (std::size_t j = 0; j < aa.size(); ++j) {
        if (!same_bits(aa[j], bb[j])) return false;
    }
    return true;
}

template<typename V>
void report_case(V input, V output) {
    const auto a = unpack(input), b = unpack(output);
    using T = ts::scalar_type<V>;
    std::fprintf(stderr, "  vec%d%s input=(%a,%a,%a,%a), output=(%a,%a,%a,%a)\n",
        ts::component_count<V>, std::same_as<T, float> ? "f" : "d",
        static_cast<double>(a[0]), static_cast<double>(a[1]),
        static_cast<double>(a[2]), static_cast<double>(a[3]),
        static_cast<double>(b[0]), static_cast<double>(b[1]),
        static_cast<double>(b[2]), static_cast<double>(b[3]));
}

// Reference computes in long double AFTER scaling. This remains usable on
// MSVC, where long double need not be wider than double, and cannot overflow
// just because the original norm does not fit in the component type.
template<typename V>
void check_finite_vector(V v, ts::test_runner& r) {
    const int failures_before = r.failures;
    using T = ts::scalar_type<V>;
    const auto input = unpack(v);
    long double scale = 0;
    for (T x : input) scale = (std::max)(scale, std::abs(static_cast<long double>(x)));
    if (scale == 0) return;
    long double sum = 0;
    for (T x : input) {
        const long double q = static_cast<long double>(x) / scale;
        sum += q * q;
    }
    const long double scaled_norm = std::sqrt(sum);
    V out(T(7));
    const bool ok = rm::try_normalize(v, out);
    r.check(ok && rm::is_all_finite(out), "finite nonzero vector can be normalized");
    if (!ok) { report_case(v, out); return; }
    const auto normalized = unpack(out);
    const long double eps = static_cast<long double>(std::numeric_limits<T>::epsilon());
    long double output_squared = 0;
    for (std::size_t j = 0; j < input.size(); ++j) {
        const long double expected = (static_cast<long double>(input[j]) / scale) / scaled_norm;
        const long double actual = static_cast<long double>(normalized[j]);
        r.check(std::abs(actual - expected) <= 8 * eps, "normalized direction matches scaled reference");
        output_squared += actual * actual;
    }
    r.check(std::abs(output_squared - 1) <= 16 * eps, "normalized length squared is near one");
    r.check(same_vector_bits(out, rm::normalize(v)), "normalize uses the same successful algorithm");
    V alias = v;
    r.check(rm::try_normalize(alias, alias) && same_vector_bits(alias, out), "successful aliasing normalization");

    // Avoid overflowing the reference multiplication on platforms with binary64
    // long double. Only compare when the true norm is safely inside T's range.
    const long double hi = static_cast<long double>((std::numeric_limits<T>::max)());
    if (scale <= (hi / scaled_norm) * 0.99L) {
        const long double expected_length = scale * scaled_norm;
        const T len = rm::length(v);
        const long double tolerance = 8 * eps * expected_length +
            2 * static_cast<long double>(std::numeric_limits<T>::denorm_min());
        r.check(std::isfinite(len) && len > T(0) &&
                std::abs(static_cast<long double>(len) - expected_length) <= tolerance,
                "scaled length matches reference without square overflow or underflow");
    }
    if (r.failures != failures_before) report_case(v, out);
}

template<typename V>
void check_rejected(V v, ts::test_runner& r) {
    using T = ts::scalar_type<V>;
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const V sentinel = ts::components<V>(-T(0), nan, T(13), T(-17));
    V out = sentinel;
    r.check(!rm::try_normalize(v, out) && same_vector_bits(out, sentinel),
            "failure leaves all output bits unchanged");
    const V result = rm::normalize(v);
    r.check(ts::all_components(result, [](T x) { return std::isnan(x); }),
            "normalize returns all NaNs on failure");
    const V original = v;
    r.check(!rm::try_normalize(v, v) && same_vector_bits(v, original),
            "failed aliasing normalization leaves input unchanged");
}

std::uint32_t next_random(std::uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

template<typename V>
void check_type(ts::test_runner& r) {
    using T = ts::scalar_type<V>;
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T inf = std::numeric_limits<T>::infinity();
    const T hi = (std::numeric_limits<T>::max)();
    volatile T stored = std::numeric_limits<T>::denorm_min();
    const T tiny = stored;
    r.check(rm::is_all_finite(V{}) && rm::length(V{}) == T(0) && !std::signbit(rm::length(V(-T(0)))),
            "zero length is positive zero");
    check_rejected(V{}, r); check_rejected(V(-T(0)), r);
    for (int component = 0; component < ts::component_count<V>; ++component) {
        for (T bad : {nan, inf, -inf}) {
            std::array<T, 4> a{T(1), T(1), T(1), T(1)};
            a[static_cast<std::size_t>(component)] = bad;
            const V v = ts::components<V>(a[0], a[1], a[2], a[3]);
            r.check(!rm::is_all_finite(v), "nonfinite component is detected in every position");
            check_rejected(v, r);
            if (std::isnan(bad)) r.check(std::isnan(rm::length(v)), "length propagates NaN");
            else r.check(rm::length(v) == inf, "infinite component gives positive infinite length");
        }
    }
    r.check(std::isnan(rm::length(ts::components<V>(inf, nan))) &&
            std::isnan(rm::length(ts::components<V>(nan, inf))), "length prioritizes NaN over infinity");
    const V axis = ts::components<V>(-tiny, -T(0));
    V out;
    r.check(rm::try_normalize(axis, out) && out.x == T(-1) && std::signbit(out.y),
            "smallest subnormal axis normalizes and preserves zero sign");
    r.check(rm::length(axis) == tiny, "smallest subnormal length is not lost");
    r.check(rm::length(V(hi)) == inf && rm::try_normalize(V(hi), out) && rm::is_all_finite(out),
            "normalization works even when the original norm overflows");
    for (T scale : {tiny, (std::numeric_limits<T>::min)(), T(1), hi / T(8)}) {
        check_finite_vector(ts::components<V>(scale, -scale, scale, -scale), r);
        check_finite_vector(ts::components<V>(T(3) * scale, T(-4) * scale), r);
    }
    check_finite_vector(ts::components<V>(hi, tiny, -tiny, T(0)), r);
    check_finite_vector(ts::components<V>(T(1), T(-2), T(3), T(-4)), r);
    check_finite_vector(ts::components<V>(std::nextafter(T(1), T(2)), T(-1), T(1), T(-1)), r);

    // Fixed seed and independent exponents cover mixed-scale and subnormal inputs.
    std::uint32_t state = 0x6a09e667u;
    constexpr int low = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits;
    constexpr int high = std::numeric_limits<T>::max_exponent - 1;
    for (int sample = 0; sample < 2048; ++sample) {
        std::array<T, 4> a{};
        for (T& x : a) {
            const int e = low + static_cast<int>(next_random(state) % static_cast<std::uint32_t>(high - low + 1));
            const T m = T(0.5) + static_cast<T>(next_random(state) % 1024u) / T(2048);
            x = std::ldexp(m, e);
            if (next_random(state) & 1u) x = -x;
        }
        const int failures_before = r.failures;
        check_finite_vector(ts::components<V>(a[0], a[1], a[2], a[3]), r);
        if (r.failures != failures_before) {
            std::fprintf(stderr, "  deterministic sample index=%d\n", sample);
            break;
        }
    }
}
} // namespace

int main() {
    ts::test_runner r;
    check_type<rm::vec2f>(r); check_type<rm::vec2d>(r);
    check_type<rm::vec3f>(r); check_type<rm::vec3d>(r);
    check_type<rm::vec4f>(r); check_type<rm::vec4d>(r);
    r.check(ts::equal(geometry_linkage_sample(rm::vec3f{3.0f, -4.0f, 0.0f}),
                      rm::vec3f{0.6f, 0.8f, 0.0f}), "geometry functions link across translation units");
    return r.finish("All vector norm and normalization checks passed.");
}
