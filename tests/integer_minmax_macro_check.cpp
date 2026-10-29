// Isolate Windows-style macro pollution from the ordinary tests.
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#define min(a, b) use_of_unprotected_min_is_an_error
#define max(a, b) use_of_unprotected_max_is_an_error
#include <cuda_render_math/vector_cast.hpp>
#include <cuda_render_math/vector_math.hpp>

namespace {
template<typename V>
consteval bool check_integer_macro_interop() {
    using T = decltype(V{}.x);
    V output(T(0));
    return (cuda_render_math::min)(V(T(1)), T(2)).x == T(1) &&
           (cuda_render_math::min)(T(1), V(T(2))).y == T(1) &&
           (cuda_render_math::max)(V(T(1)), V(T(2))).x == T(2) &&
           cuda_render_math::try_vector_cast(V(T(2)), output) && output.y == T(2) &&
           cuda_render_math::clamp(output, T(0), T(1)).x == T(1);
}
static_assert(check_integer_macro_interop<cuda_render_math::vec2i>());
static_assert(check_integer_macro_interop<cuda_render_math::vec3i>());
static_assert(check_integer_macro_interop<cuda_render_math::vec4i>());
static_assert(check_integer_macro_interop<cuda_render_math::vec2u>());
static_assert(check_integer_macro_interop<cuda_render_math::vec3u>());
static_assert(check_integer_macro_interop<cuda_render_math::vec4u>());
} // namespace
#undef min
#undef max
