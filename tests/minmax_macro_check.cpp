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
#undef min
#undef max
