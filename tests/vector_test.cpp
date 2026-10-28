#include <cuda_render_math/vector.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <utility>

using cuda_render_math::vec3;
using cuda_render_math::vec3d;
using cuda_render_math::vec3f;

// Defined in another translation unit to exercise normal header-only linkage.
vec3f vector_odr_sample() noexcept;

namespace {

template <typename L, typename R>
concept CanAdd = requires(L left, R right) { left + right; };

template <typename L, typename R>
concept CanSubtract = requires(L left, R right) { left - right; };

template <typename L, typename R>
concept CanMultiply = requires(L left, R right) { left * right; };

template <typename L, typename R>
concept CanDivide = requires(L left, R right) { left / right; };

template <typename L, typename R>
concept CanAddAssign = requires(L left, R right) { left += right; };

template <typename L, typename R>
concept CanSubtractAssign = requires(L left, R right) { left -= right; };

template <typename L, typename R>
concept CanMultiplyAssign = requires(L left, R right) { left *= right; };

template <typename L, typename R>
concept CanDivideAssign = requires(L left, R right) { left /= right; };

static_assert(cuda_render_math::real_scalar<float>);
static_assert(cuda_render_math::real_scalar<double>);
static_assert(!cuda_render_math::real_scalar<int>);
static_assert(!cuda_render_math::real_scalar<long double>);
static_assert(!cuda_render_math::real_scalar<const float>);
static_assert(!cuda_render_math::real_scalar<volatile double>);

template <cuda_render_math::real_scalar T>
consteval bool check_type_contracts()
{
    using V = vec3<T>;

    static_assert(std::is_aggregate_v<V>);
    static_assert(std::is_trivially_copyable_v<V>);
    static_assert(std::is_standard_layout_v<V>);
    static_assert(sizeof(V) == 3 * sizeof(T));
    static_assert(alignof(V) == alignof(T));
    static_assert(offsetof(V, x) == 0);
    static_assert(offsetof(V, y) == sizeof(T));
    static_assert(offsetof(V, z) == 2 * sizeof(T));

    static_assert(std::same_as<decltype(-std::declval<V>()), V>);
    static_assert(std::same_as<decltype(std::declval<V>() + std::declval<V>()), V>);
    static_assert(std::same_as<decltype(std::declval<V>() - std::declval<V>()), V>);
    static_assert(std::same_as<decltype(std::declval<V>() * std::declval<T>()), V>);
    static_assert(std::same_as<decltype(std::declval<T>() * std::declval<V>()), V>);
    static_assert(std::same_as<decltype(std::declval<V>() / std::declval<T>()), V>);
    static_assert(std::same_as<decltype(std::declval<V&>() += std::declval<V>()), V&>);
    static_assert(std::same_as<decltype(std::declval<V&>() -= std::declval<V>()), V&>);
    static_assert(std::same_as<decltype(std::declval<V&>() *= std::declval<T>()), V&>);
    static_assert(std::same_as<decltype(std::declval<V&>() /= std::declval<T>()), V&>);

    constexpr V default_initialized;
    constexpr V value_initialized{};
    return default_initialized.x == T(0) && default_initialized.y == T(0) &&
           default_initialized.z == T(0) && value_initialized.x == T(0) &&
           value_initialized.y == T(0) && value_initialized.z == T(0);
}

static_assert(check_type_contracts<float>());
static_assert(check_type_contracts<double>());

// Test each operation separately: rejecting one must not conceal another accepting
// an unintended conversion.
template <cuda_render_math::real_scalar T, cuda_render_math::real_scalar U>
consteval bool check_rejected_vector_operations()
{
    static_assert(!CanAdd<vec3<T>, vec3<U>>);
    static_assert(!CanSubtract<vec3<T>, vec3<U>>);
    static_assert(!CanAddAssign<vec3<T>, vec3<U>>);
    static_assert(!CanSubtractAssign<vec3<T>, vec3<U>>);
    return true;
}

static_assert(check_rejected_vector_operations<float, double>());
static_assert(check_rejected_vector_operations<double, float>());

template <cuda_render_math::real_scalar T, class Scalar>
consteval bool check_rejected_scalar_operations()
{
    static_assert(!CanMultiply<vec3<T>, Scalar>);
    static_assert(!CanMultiply<Scalar, vec3<T>>);
    static_assert(!CanDivide<vec3<T>, Scalar>);
    static_assert(!CanMultiplyAssign<vec3<T>, Scalar>);
    static_assert(!CanDivideAssign<vec3<T>, Scalar>);
    return true;
}

static_assert(check_rejected_scalar_operations<float, double>());
static_assert(check_rejected_scalar_operations<float, int>());
static_assert(check_rejected_scalar_operations<double, float>());
static_assert(check_rejected_scalar_operations<double, int>());

static_assert(!std::is_convertible_v<vec3f, vec3d>);
static_assert(!std::is_convertible_v<vec3d, vec3f>);
static_assert(std::same_as<decltype(cuda_render_math::vector_cast<double>(vec3f{})), vec3d>);
static_assert(std::same_as<decltype(cuda_render_math::vector_cast<float>(vec3d{})), vec3f>);

template <cuda_render_math::real_scalar T>
constexpr bool equal(vec3<T> actual, vec3<T> expected)
{
    return actual.x == expected.x && actual.y == expected.y && actual.z == expected.z;
}

template <cuda_render_math::real_scalar T>
consteval bool check_constexpr_arithmetic()
{
    const vec3<T> a{T(2), T(4), T(8)};
    const vec3<T> b{T(1), T(2), T(4)};
    if (!equal(-a, vec3<T>{T(-2), T(-4), T(-8)}) ||
        !equal(a + b, vec3<T>{T(3), T(6), T(12)}) ||
        !equal(a - b, b) || !equal(b * T(2), a) ||
        !equal(T(2) * b, a) || !equal(a / T(2), b)) {
        return false;
    }

    vec3<T> value = a;
    value += b;
    if (!equal(value, vec3<T>{T(3), T(6), T(12)})) {
        return false;
    }
    value -= b;
    if (!equal(value, a)) {
        return false;
    }
    value *= T(2);
    if (!equal(value, vec3<T>{T(4), T(8), T(16)})) {
        return false;
    }
    value /= T(2);
    return equal(value, a);
}

static_assert(check_constexpr_arithmetic<float>());
static_assert(check_constexpr_arithmetic<double>());
static_assert(equal(cuda_render_math::vector_cast<double>(vec3f{1.0f, -2.0f, 0.5f}),
                    vec3d{1.0, -2.0, 0.5}));
static_assert(equal(cuda_render_math::vector_cast<float>(vec3d{1.0, -2.0, 0.5}),
                    vec3f{1.0f, -2.0f, 0.5f}));

int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

template <cuda_render_math::real_scalar T>
bool finite_unit_components(vec3<T> value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && equal(value, vec3<T>{T(1), T(-1), T(1)});
}

template <cuda_render_math::real_scalar T>
void check_runtime_division(const char* denorm_message, const char* tiny_message)
{
    // Volatile reads keep these regression cases in runtime arithmetic, including
    // Release builds. No reciprocal is representable for denorm_min().
    volatile T denorm_source = std::numeric_limits<T>::denorm_min();
    const T denorm = denorm_source;
    const vec3<T> denorm_vector{denorm, -denorm, denorm};
    check(finite_unit_components(denorm_vector / denorm), denorm_message);
    vec3<T> assigned_denorm = denorm_vector;
    assigned_denorm /= denorm;
    check(finite_unit_components(assigned_denorm), denorm_message);

    volatile T tiny_source = T(1e-30);
    const T tiny = tiny_source;
    const vec3<T> tiny_vector{tiny, -tiny, tiny};
    check(finite_unit_components(tiny_vector / tiny), tiny_message);
    vec3<T> assigned_tiny = tiny_vector;
    assigned_tiny /= tiny;
    check(finite_unit_components(assigned_tiny), tiny_message);
}

} // namespace

int main()
{
    check_runtime_division<float>("float division by denorm_min", "float division by 1e-30");
    check_runtime_division<double>("double division by denorm_min", "double division by 1e-30");
    check(equal(vector_odr_sample(), vec3f{2.0f, 4.0f, 6.0f}), "second translation unit");

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return 1;
    }
    std::puts("All vector contract checks passed.");
    return 0;
}
