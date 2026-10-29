// Emulate Windows min/max macros without requiring Windows headers.
#include <concepts>
#include <cmath>
#include <limits>
#define min(a, b) windows_min_macro_must_not_expand
#define max(a, b) windows_max_macro_must_not_expand
#include <cuda_render_math/vector_math.hpp>
#include <cuda_render_math/vector_math.hpp>
#ifndef min
#error "The library must not undefine caller macros."
#endif
#ifndef max
#error "The library must not undefine caller macros."
#endif
static_assert((cuda_render_math::min)(1.0f, 2.0f) == 1.0f);
static_assert((cuda_render_math::max)(cuda_render_math::vec4f(1.0f), 2.0f).w == 2.0f);
namespace {

template<typename V>
consteval bool check_constant_minmax() {
    using T = decltype(V{}.x);
    const V a(T(1));
    const V b(T(2));
    const auto all_equal = [](V v, T expected) constexpr {
        if (v.x != expected || v.y != expected) return false;
        if constexpr (requires { v.z; }) {
            if (v.z != expected) return false;
        }
        if constexpr (requires { v.w; }) {
            if (v.w != expected) return false;
        }
        return true;
    };
    return all_equal((cuda_render_math::min)(a, b), T(1)) &&
           all_equal((cuda_render_math::min)(a, T(2)), T(1)) &&
           all_equal((cuda_render_math::min)(T(1), b), T(1)) &&
           all_equal((cuda_render_math::max)(a, b), T(2)) &&
           all_equal((cuda_render_math::max)(a, T(2)), T(2)) &&
           all_equal((cuda_render_math::max)(T(1), b), T(2));
}

static_assert(check_constant_minmax<cuda_render_math::vec2f>());
static_assert(check_constant_minmax<cuda_render_math::vec2d>());
static_assert(check_constant_minmax<cuda_render_math::vec3f>());
static_assert(check_constant_minmax<cuda_render_math::vec3d>());
static_assert(check_constant_minmax<cuda_render_math::vec4f>());
static_assert(check_constant_minmax<cuda_render_math::vec4d>());

} // namespace

#undef min
#undef max
