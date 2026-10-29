#include <cuda_render_math/scalar_math.hpp>
#include "vector_test_support.hpp"

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>

namespace rm = cuda_render_math;
namespace ts = test_support;

namespace {

template<typename L, typename R>
concept can_min = requires(L a, R b) { (rm::min)(a, b); };
template<typename L, typename R>
concept can_max = requires(L a, R b) { (rm::max)(a, b); };
template<typename L, typename R>
concept can_fmod = requires(L a, R b) { rm::fmod(a, b); };
template<typename T, typename U>
concept can_clamp = requires(T x, U a) { rm::clamp(x, a, a); };
template<typename T, typename U>
concept can_lerp = requires(T x, U t) { rm::lerp(x, x, t); };
template<typename T, typename U>
concept can_smoothstep = requires(T x, U t) { rm::smoothstep(x, x, t); };

static_assert(!can_min<float, double> && !can_min<float, int>);
static_assert(!can_max<float, double> && !can_max<float, int>);
static_assert(!can_fmod<float, double> && !can_fmod<float, int>);
static_assert(!can_clamp<float, double> && !can_lerp<float, double>);
static_assert(!can_smoothstep<float, double>);
static_assert((rm::min)(2.0f, 3.0f) == 2.0f);
static_assert((rm::max)(2.0, 3.0) == 3.0);
static_assert(std::same_as<decltype(rm::lerp(1.0f, 2.0f, 0.5f)), float>);
static_assert(std::same_as<decltype(rm::smoothstep(0.0, 1.0, 0.5)), double>);

// Runtime reads prevent the edge tests from exercising only constant folding.
template<typename T>
T runtime_value(T v) { volatile T stored = v; return stored; }

template<rm::real_scalar T>
void check_values(ts::test_runner& r) {
    const T zero = runtime_value(T(0));
    const T negzero = runtime_value(-T(0));
    const T nan = runtime_value(std::numeric_limits<T>::quiet_NaN());
    const T inf = runtime_value(std::numeric_limits<T>::infinity());
    const T largest = runtime_value((std::numeric_limits<T>::max)());
    const T denorm = runtime_value(std::numeric_limits<T>::denorm_min());
    r.check((rm::min)(T(-2), T(3)) == T(-2) &&
            (rm::max)(T(-2), T(3)) == T(3), "ordered min/max");
    r.check(std::isnan((rm::min)(nan, T(2))) && std::isnan((rm::min)(T(2), nan)) &&
            std::isnan((rm::max)(nan, T(2))) && std::isnan((rm::max)(T(2), nan)),
            "min/max propagate NaN from either operand");
    r.check(std::signbit((rm::min)(negzero, zero)) &&
            !std::signbit((rm::min)(zero, negzero)) &&
            std::signbit((rm::max)(negzero, zero)) &&
            !std::signbit((rm::max)(zero, negzero)), "min/max ties preserve first operand");
    r.check((rm::min)(-inf, T(2)) == -inf && (rm::max)(inf, T(2)) == inf,
            "min/max accept infinities");
    r.check(rm::clamp(T(-2), T(-1), T(3)) == T(-1) &&
            rm::clamp(T(4), T(-1), T(3)) == T(3) &&
            rm::clamp(T(2), T(-1), T(3)) == T(2), "clamp boundaries and interior");
    r.check(rm::clamp(inf, T(-1), T(1)) == T(1) &&
            rm::clamp(T(2), -inf, inf) == T(2) &&
            rm::clamp(T(2), T(1), T(1)) == T(1), "clamp infinite/equal bounds");
    r.check(std::signbit(rm::clamp(negzero, T(-1), T(1))), "clamp preserves signed zero");
    r.check(std::isnan(rm::clamp(nan, T(0), T(1))) &&
            std::isnan(rm::clamp(T(0), nan, T(1))) &&
            std::isnan(rm::clamp(T(0), T(-1), nan)) &&
            std::isnan(rm::clamp(T(0), T(1), T(-1))), "invalid clamp returns NaN");
    r.check(rm::saturate(-inf) == T(0) && rm::saturate(inf) == T(1) &&
            std::isnan(rm::saturate(nan)), "saturate never hides NaN");

    r.check(rm::floor(T(-1.25)) == T(-2) && rm::ceil(T(-1.25)) == T(-1),
            "floor/ceil negative input");
    r.check(std::signbit(rm::floor(negzero)) && std::signbit(rm::ceil(negzero)),
            "floor/ceil preserve negative zero");
    r.check(rm::fract(T(-1.25)) == T(0.75) && rm::fract(T(1.25)) == T(0.25) &&
            rm::fract(T(-2)) == T(0), "fract uses floor, not truncation");
    r.check(rm::fract(-denorm) == T(1), "fract documents rounded one for tiny negative values");
    r.check(std::isnan(rm::fract(inf)) && std::isnan(rm::fract(nan)), "fract non-finite input");
    r.check(rm::fmod(T(-5.5), T(2)) == T(-1.5) &&
            rm::fmod(T(5.5), T(-2)) == T(1.5), "fmod remainder follows numerator sign");
    r.check(std::signbit(rm::fmod(negzero, T(2))) &&
            rm::fmod(T(2), inf) == T(2) && std::isnan(rm::fmod(T(1), zero)) &&
            std::isnan(rm::fmod(inf, T(2))), "fmod exceptional values");

    r.check(rm::lerp(T(2), T(6), T(0.25)) == T(3) &&
            rm::lerp(T(2), T(6), T(-1)) == T(-2) &&
            rm::lerp(T(2), T(6), T(2)) == T(10), "lerp interpolation and extrapolation");
    r.check(rm::lerp(-largest, largest, zero) == -largest &&
            rm::lerp(-largest, largest, T(1)) == largest &&
            rm::lerp(-largest, largest, T(0.5)) == T(0), "lerp avoids overflowing endpoint difference");
    r.check(rm::lerp(largest, largest, largest) == largest &&
            std::signbit(rm::lerp(negzero, T(1), zero)), "lerp constant curve and signed endpoint");
    r.check(std::isnan(rm::lerp(inf, T(1), zero)) &&
            std::isnan(rm::lerp(T(1), inf, zero)) &&
            std::isnan(rm::lerp(T(1), T(2), inf)) &&
            std::isnan(rm::lerp(T(1), T(2), nan)), "lerp rejects non-finite inputs");
    r.check(std::isinf(rm::lerp(-largest, largest, T(2))), "lerp extrapolation can overflow");

    r.check(rm::smoothstep(T(0), T(1), T(0.25)) == T(0.15625) &&
            rm::smoothstep(T(0), T(1), T(0.5)) == T(0.5) &&
            rm::smoothstep(T(0), T(1), -inf) == T(0) &&
            rm::smoothstep(T(0), T(1), inf) == T(1), "smoothstep curve and saturated tails");
    r.check(rm::smoothstep(-largest, largest, T(0)) == T(0.5), "smoothstep wide interval");
    r.check(rm::smoothstep(T(0), T(2) * denorm, denorm) == T(0.5),
            "smoothstep subnormal interval uses direct division");
    r.check(std::isnan(rm::smoothstep(T(1), T(1), T(2))) &&
            std::isnan(rm::smoothstep(T(2), T(1), T(0))) &&
            std::isnan(rm::smoothstep(-inf, T(1), T(0))) &&
            std::isnan(rm::smoothstep(T(0), inf, T(0))) &&
            std::isnan(rm::smoothstep(T(0), T(1), nan)), "smoothstep invalid inputs return NaN");

    // Exercise wide exponent ranges without relying on long double being wider on MSVC.
    std::uint64_t state = 0x91abc43U;
    auto next = [&] { state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407); return state; };
    const int radius = std::numeric_limits<T>::max_exponent - 4;
    for (int i = 0; i < 2000; ++i) {
        const int ea = int(next() % std::uint64_t(2 * radius + 1)) - radius;
        const int eb = int(next() % std::uint64_t(2 * radius + 1)) - radius;
        const T a = std::ldexp((next() & 1U) ? T(0.75) : T(-0.75), ea);
        const T b = std::ldexp((next() & 1U) ? T(0.625) : T(-0.625), eb);
        const T t = T(next() % 1025U) / T(1024);
        const T value = rm::lerp(a, b, t);
        r.check(std::isfinite(value) && value >= (rm::min)(a, b) &&
                value <= (rm::max)(a, b), "lerp random finite convex interpolation");
        const T reference = std::lerp(a, b, t);
        const T scale = (rm::max)(std::abs(a), std::abs(b));
        r.check(std::abs(value - reference) <= T(8) * std::numeric_limits<T>::epsilon() * scale +
                T(8) * denorm, "lerp agrees with standard library within scaled tolerance");
    }
}
} // namespace

int main() {
    ts::test_runner r;
    check_values<float>(r);
    check_values<double>(r);
    return r.finish("All scalar range, rounding and interpolation checks passed.");
}
