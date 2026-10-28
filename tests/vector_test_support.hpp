#pragma once

#include <cuda_render_math/vector.hpp>

#include <cstdio>
#include <type_traits>

namespace test_support {

template<typename V>
using scalar_type = std::remove_cv_t<decltype(V{}.x)>;

template<typename V>
constexpr int component_count = [] {
    if constexpr (requires(V v) { v.w; }) return 4;
    else if constexpr (requires(V v) { v.z; }) return 3;
    else return 2;
}();

template<typename V>
constexpr V components(scalar_type<V> x, scalar_type<V> y,
                       scalar_type<V> z = {}, scalar_type<V> w = {}) noexcept {
    if constexpr (component_count<V> == 4) return V{x, y, z, w};
    else if constexpr (component_count<V> == 3) return V{x, y, z};
    else return V{x, y};
}

template<typename V>
constexpr bool equal(V a, V b) noexcept {
    if (a.x != b.x || a.y != b.y) return false;
    if constexpr (component_count<V> >= 3) {
        if (a.z != b.z) return false;
    }
    if constexpr (component_count<V> == 4) {
        if (a.w != b.w) return false;
    }
    return true;
}

template<typename V, typename Predicate>
constexpr bool all_components(V value, Predicate predicate) {
    if (!predicate(value.x) || !predicate(value.y)) return false;
    if constexpr (component_count<V> >= 3) {
        if (!predicate(value.z)) return false;
    }
    if constexpr (component_count<V> == 4) {
        if (!predicate(value.w)) return false;
    }
    return true;
}

// Deliberately independent of NDEBUG: Release builds execute the same checks.
struct test_runner {
    int failures = 0;

    void check(bool condition, const char* message) {
        if (!condition) {
            std::fprintf(stderr, "FAIL: %s\n", message);
            ++failures;
        }
    }

    int finish(const char* success_message) const {
        if (failures != 0) {
            std::fprintf(stderr, "%d check(s) failed.\n", failures);
            return 1;
        }
        std::puts(success_message);
        return 0;
    }
};

} // namespace test_support
