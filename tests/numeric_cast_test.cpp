#include <cuda_render_math/vector_cast.hpp>
#include "vector_test_support.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rm = cuda_render_math;
namespace ts = test_support;

namespace {
using I = std::int32_t;
using U = std::uint32_t;

template<typename From, typename To>
concept can_try_scalar_cast = requires(From x, To& out) { rm::try_scalar_cast(x, out); };
static_assert(!can_try_scalar_cast<bool, I> && !can_try_scalar_cast<float, bool>);
static_assert(!can_try_scalar_cast<std::int64_t, float> && !can_try_scalar_cast<double, const I>);

consteval bool check_constant_conversion() {
    I i = 7; U u = 9; float f = 0.0f; double d = 0.0;
    if (!rm::try_scalar_cast(3.75f, i) || i != 3) return false;
    if (!rm::try_scalar_cast(-3.75, i) || i != -3) return false;
    if (rm::try_scalar_cast(2147483648.0f, i) || i != -3) return false;
    if (!rm::try_scalar_cast(2147483647.75, i) || i != 2147483647) return false;
    if (!rm::try_scalar_cast(-2147483648.75, i) || i != (-2147483647 - 1)) return false;
    if (rm::try_scalar_cast(-2147483649.0, i) || i != (-2147483647 - 1)) return false;
    if (!rm::try_scalar_cast(-0.75, u) || u != 0u) return false;
    if (rm::try_scalar_cast(-1.0f, u) || u != 0u) return false;
    if (!rm::try_scalar_cast(4294967295.75, u) || u != 4294967295u) return false;
    if (rm::try_scalar_cast(4294967296.0, u) || u != 4294967295u) return false;
    if (rm::try_scalar_cast(I(-1), u) || u != 4294967295u) return false;
    if (rm::try_scalar_cast(U(2147483648u), i) || i != (-2147483647 - 1)) return false;
    if (!rm::try_scalar_cast(I(42), u) || u != 42u) return false;
    if (!rm::try_scalar_cast(U(42), i) || i != 42) return false;
    if (!rm::try_scalar_cast(U(4294967295u), d) || d != 4294967295.0) return false;
    if (!rm::try_scalar_cast(1.25, f) || f != 1.25f) return false;
    // The checked operation permits rounding; it is not an exactness test.
    if (!rm::try_scalar_cast(I(16777217), f)) return false;
    rm::vec3i out(11);
    if (!rm::try_vector_cast(rm::vec3f{1.9f, -2.9f, 3.0f}, out) ||
        !ts::equal(out, rm::vec3i{1, -2, 3})) return false;
    if (rm::try_vector_cast(rm::vec3d{4.0, 2147483648.0, 6.0}, out) ||
        !ts::equal(out, rm::vec3i{1, -2, 3})) return false;
    if (!rm::try_vector_cast(out, out)) return false;
    // Explicit integer casts keep native modulo conversion semantics in C++20.
    const auto raw_unsigned = rm::vector_cast<U>(rm::vec3i(-1));
    const auto raw_signed = rm::vector_cast<I>(rm::vec3u(4294967295u));
    return ts::equal(raw_unsigned, rm::vec3u(4294967295u)) &&
           ts::equal(raw_signed, rm::vec3i(-1));
}
static_assert(check_constant_conversion());

// An independent oracle: truncate in long double, then compare against exactly
// representable 32-bit integer bounds. Never cast an out-of-range value to int.
template<rm::integer_scalar To, rm::real_scalar From>
void check_integer_case(ts::test_runner& r, From source) {
    volatile From stored = source;
    const From value = stored;
    const long double truncated = std::trunc(static_cast<long double>(value));
    const bool expected = std::isfinite(truncated) &&
        truncated >= static_cast<long double>((std::numeric_limits<To>::lowest)()) &&
        truncated <= static_cast<long double>((std::numeric_limits<To>::max)());
    To output = To(123);
    const bool actual = rm::try_scalar_cast(value, output);
    r.check(actual == expected, "scalar cast success agrees with truncation/range oracle");
    if (expected && actual) {
        r.check(output == static_cast<To>(truncated), "accepted integer result");
    } else {
        r.check(output == To(123), "failed scalar cast preserves output");
    }
}

template<rm::real_scalar From>
void check_integer_boundaries(ts::test_runner& r) {
    const From inf = std::numeric_limits<From>::infinity();
    const From nan = std::numeric_limits<From>::quiet_NaN();
    const From low = From(-2147483648.0);
    const From high = From(2147483648.0);
    const From unsigned_high = From(4294967296.0);
    const From cases[] = {
        From(0), -From(0), From(1), From(-1), From(0.75), From(-0.75),
        From(3.9), From(-3.9), From(2147483647.0), From(2147483647.75),
        low, From(-2147483648.75), From(-2147483649.0), high,
        std::nextafter(low, -inf), std::nextafter(low, inf),
        std::nextafter(high, -inf), std::nextafter(high, inf),
        From(4294967295.0), From(4294967295.75), unsigned_high,
        std::nextafter(unsigned_high, -inf), std::nextafter(unsigned_high, inf),
        std::nextafter(From(-1), From(0)), std::nextafter(From(-1), -inf),
        std::numeric_limits<From>::denorm_min(), -std::numeric_limits<From>::denorm_min(),
        (std::numeric_limits<From>::max)(), -(std::numeric_limits<From>::max)(), inf, -inf, nan
    };
    for (From value : cases) {
        check_integer_case<I>(r, value);
        check_integer_case<U>(r, value);
    }
    // Reproducible sweep, independent of a random library or global RNG state.
    U bits = 0x412aa3f1u;
    for (int k = 0; k < 4096; ++k) {
        bits = bits * 1664525u + 1013904223u;
        const From center = static_cast<From>(static_cast<double>(bits) - 2147483648.0);
        check_integer_case<I>(r, center + From(0.75));
        check_integer_case<U>(r, center + From(0.75));
    }
}

template<rm::supported_arithmetic To, rm::supported_arithmetic From>
void check_vector_pair(ts::test_runner& r) {
    // All 16 type pairs, all three dimensions, both checked and explicit paths.
    const rm::vec2<From> v2{From(1), From(2)};
    const rm::vec3<From> v3{From(1), From(2), From(3)};
    const rm::vec4<From> v4{From(1), From(2), From(3), From(4)};
    rm::vec2<To> o2(To(99)); rm::vec3<To> o3(To(99)); rm::vec4<To> o4(To(99));
    r.check(rm::try_vector_cast(v2, o2) && ts::equal(o2, rm::vector_cast<To>(v2)), "checked vec2 type-pair cast");
    r.check(rm::try_vector_cast(v3, o3) && ts::equal(o3, rm::vector_cast<To>(v3)), "checked vec3 type-pair cast");
    r.check(rm::try_vector_cast(v4, o4) && ts::equal(o4, rm::vector_cast<To>(v4)), "checked vec4 type-pair cast");
}
template<rm::supported_arithmetic From>
void check_destinations(ts::test_runner& r) {
    check_vector_pair<float, From>(r); check_vector_pair<double, From>(r);
    check_vector_pair<I, From>(r); check_vector_pair<U, From>(r);
}

template<typename V, rm::integer_scalar To>
void check_vector_failure(ts::test_runner& r) {
    using T = ts::scalar_type<V>;
    using Out = decltype(rm::vector_cast<To>(V{}));
    for (int index = 0; index < ts::component_count<V>; ++index) {
        V source(T(1));
        // Do not index through &x: distinct struct members are not a C++ array.
        const T invalid = std::numeric_limits<T>::quiet_NaN();
        if (index == 0) source.x = invalid;
        else if (index == 1) source.y = invalid;
        else if constexpr (ts::component_count<V> >= 3) {
            if (index == 2) source.z = invalid;
            else if constexpr (ts::component_count<V> == 4) source.w = invalid;
        }
        Out output(To(71));
        r.check(!rm::try_vector_cast(source, output) && ts::equal(output, Out(To(71))),
                "failure at each component preserves the whole vector");
    }
}
template<typename V>
void check_vector_failures(ts::test_runner& r) {
    check_vector_failure<V, I>(r); check_vector_failure<V, U>(r);
}

void check_real_conversions(ts::test_runner& r) {
    float output = 7.0f;
    const double max_float = static_cast<double>((std::numeric_limits<float>::max)());
    r.check(rm::try_scalar_cast(max_float, output) && output == (std::numeric_limits<float>::max)(), "float finite maximum");
    output = 7.0f;
    r.check(!rm::try_scalar_cast(std::nextafter(max_float, std::numeric_limits<double>::infinity()), output) && output == 7.0f,
            "finite overflow rejected before narrowing");
    r.check(!rm::try_scalar_cast(-(std::numeric_limits<double>::max)(), output) && output == 7.0f, "negative finite overflow");
    r.check(rm::try_scalar_cast(std::numeric_limits<double>::infinity(), output) && std::isinf(output) && output > 0.0f, "positive infinity preserved");
    r.check(rm::try_scalar_cast(-std::numeric_limits<double>::infinity(), output) && std::isinf(output) && output < 0.0f, "negative infinity preserved");
    r.check(rm::try_scalar_cast(std::numeric_limits<double>::quiet_NaN(), output) && std::isnan(output), "NaN remains NaN for real destination");
    r.check(rm::try_scalar_cast(-0.0, output) && output == 0.0f && std::signbit(output), "negative zero preserved");
    r.check(rm::try_scalar_cast(std::numeric_limits<double>::denorm_min(), output) && output == 0.0f, "underflow to zero is allowed");
    r.check(rm::try_scalar_cast(U(4294967295u), output) && std::isfinite(output), "uint32 to float permits precision loss");
    rm::vec3f destination(7.0f);
    r.check(!rm::try_vector_cast(rm::vec3d{1.0, (std::numeric_limits<double>::max)(), 3.0}, destination) &&
            ts::equal(destination, rm::vec3f(7.0f)), "narrowing vector overflow preserves every component");
}
} // namespace

int main() {
    ts::test_runner r;
    check_integer_boundaries<float>(r); check_integer_boundaries<double>(r);
    check_destinations<I>(r); check_destinations<U>(r);
    check_destinations<float>(r); check_destinations<double>(r);
    check_vector_failures<rm::vec2f>(r); check_vector_failures<rm::vec3f>(r); check_vector_failures<rm::vec4f>(r);
    check_vector_failures<rm::vec2d>(r); check_vector_failures<rm::vec3d>(r); check_vector_failures<rm::vec4d>(r);
    check_real_conversions(r);
    return r.finish("All checked numeric-cast checks passed.");
}
