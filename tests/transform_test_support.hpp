#pragma once

#include "vector_test_support.hpp"
#include <cuda_render_math/matrix.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

namespace transform_test_support {
namespace rm = cuda_render_math;

template<rm::real_scalar T>
inline constexpr long double tolerance =
    512.0L * static_cast<long double>(
        std::numeric_limits<T>::epsilon());

template<rm::real_scalar T>
bool near(T a, T b, long double tol = tolerance<T>) {
    const long double aa = static_cast<long double>(a);
    const long double bb = static_cast<long double>(b);

    return std::isfinite(aa) && std::isfinite(bb) &&
        std::fabs(aa - bb) <= tol * (1.0L + std::fabs(bb));
}

template<rm::real_scalar T>
bool near(
    rm::vec3<T> a,
    rm::vec3<T> b,
    long double tol = tolerance<T>) {

    return near(a.x, b.x, tol) &&
           near(a.y, b.y, tol) &&
           near(a.z, b.z, tol);
}

template<rm::real_scalar T>
bool near(
    rm::mat3<T> a,
    rm::mat3<T> b,
    long double tol = tolerance<T>) {

    for (int i = 0; i < 9; ++i) {
        if (!near(a.elements[i], b.elements[i], tol))
            return false;
    }

    return true;
}
template<typename T>
bool equal(rm::mat3<T> a, rm::mat3<T> b) {
    for (int i = 0; i < 9; ++i) if (a.elements[i] != b.elements[i]) return false;
    return true;
}

// Independent accumulation in long double, not the matrix multiplication under test.
template<typename T>
long double inverse_residual(rm::mat3<T> a, rm::mat3<T> inv) {
    long double worst = 0.0L;
    for (int side = 0; side < 2; ++side) {
        const auto left = side == 0 ? a : inv;
        const auto right = side == 0 ? inv : a;
        for (int r = 0; r < 3; ++r) {
            long double row_sum = 0.0L;
            for (int c = 0; c < 3; ++c) {
                long double sum = 0.0L;
                for (int k = 0; k < 3; ++k)
                    sum += static_cast<long double>(left(r,k)) * static_cast<long double>(right(k,c));
                row_sum += std::fabs(sum - (r == c ? 1.0L : 0.0L));
            }
            if (row_sum > worst) worst = row_sum;
        }
    }
    return worst;
}

struct sequence {
    std::uint32_t state = 0xd91743b1u;
    template<typename T> T next() {
        state = state * 1664525u + 1013904223u;
        return static_cast<T>(static_cast<int>(state >> 16) - 32768) / T(32768);
    }
};
} // namespace transform_test_support
