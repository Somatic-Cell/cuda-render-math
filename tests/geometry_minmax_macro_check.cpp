#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#define min(a, b) do_not_expand_min(a, b)
#define max(a, b) do_not_expand_max(a, b)
#include <cuda_render_math/vector_geometry.hpp>

#ifndef min
#error "A public header must not remove the caller min macro."
#endif
#ifndef max
#error "A public header must not remove the caller max macro."
#endif

// Compile instantiations rather than assuming runtime math is constexpr.
cuda_render_math::vec3f geometry_with_minmax_macros(cuda_render_math::vec3f v) {
    return cuda_render_math::reflect(cuda_render_math::normalize(v),
                                    cuda_render_math::vec3f{0.0f, 1.0f, 0.0f});
}
#undef min
#undef max
