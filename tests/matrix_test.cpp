#include <cuda_render_math/matrix_numerics.hpp>
#include "transform_test_support.hpp"
#include <initializer_list>

#include <cfenv>
#include <concepts>
#include <type_traits>

namespace rm = cuda_render_math;
namespace ts = test_support;
namespace tx = transform_test_support;

namespace {
template<typename T> concept has_matrix = requires { typename rm::mat3<T>; };
template<typename A, typename B> concept can_multiply = requires(A a, B b) { a * b; };
static_assert(has_matrix<float> && has_matrix<double>);
static_assert(!has_matrix<int> && !has_matrix<long double>);
static_assert(!can_multiply<rm::mat3f, rm::vec3d>);
static_assert(!can_multiply<rm::mat3f, rm::mat3d>);
static_assert(!can_multiply<rm::vec3f, rm::mat3f>);
static_assert(!can_multiply<rm::mat3f, int>);
static_assert(!std::is_convertible_v<float, rm::mat3f>);
static_assert(std::is_standard_layout_v<rm::mat3f> && std::is_trivially_copyable_v<rm::mat3f>);
static_assert(std::is_standard_layout_v<rm::mat3d> && std::is_trivially_copyable_v<rm::mat3d>);
static_assert(sizeof(rm::mat3f) == 9 * sizeof(float) && alignof(rm::mat3f) == alignof(float));
static_assert(sizeof(rm::mat3d) == 9 * sizeof(double) && alignof(rm::mat3d) == alignof(double));
static_assert(noexcept(rm::try_inverse(rm::mat3f{}, std::declval<rm::mat3f&>())));

consteval bool construction() {
    const rm::mat3f z;
    const auto i = rm::mat3f::identity();
    if (z(1,1) != 0 || i(1,1) != 1 || i(2,0) != 0) return false;
    const rm::mat3f m{1,2,3,4,5,6,7,8,9};
    if (m.data()[5] != 6 || m.row(1).z != 6 || m.column(1).z != 8) return false;
    const auto t = rm::transpose(m);
    const auto r = rm::mat3f::from_rows(m.row(0),m.row(1),m.row(2));
    const auto c = rm::mat3f::from_columns(m.column(0),m.column(1),m.column(2));
    if (t(0,1) != 4 || r(2,0) != 7 || c(0,2) != 3) return false;
    auto sum = (m + m) / 2.0f;
    sum += i; sum -= i; sum *= 2.0f; sum /= 2.0f;
    return sum(2,2) == 9 && rm::matrix_cast<double>(m)(2,1) == 8.0 && (-m)(1,2) == -6;
}
static_assert(construction());

template<rm::real_scalar T>
void checks(ts::test_runner& r) {
    using M = rm::mat3<T>; using V = rm::vec3<T>;
    const int before = r.failures;
    const M identity = M::identity();
    const M a{T(1),T(2),T(3), T(0),T(1),T(4), T(5),T(6),T(0)};
    const M expected{T(-24),T(18),T(5), T(20),T(-15),T(-4), T(-5),T(4),T(1)};
    const M row_test{T(1),T(2),T(3), T(4),T(5),T(6), T(7),T(8),T(10)};
    r.check(tx::near(row_test * V{T(1),T(2),T(4)}, V{T(17),T(38),T(63)}), "row-major M*v");
    r.check(tx::near(rm::determinant(a), T(1)), "compensated determinant known value");
    M inv;
    r.check(rm::try_inverse(a, inv), "known inverse accepted");
    r.check(tx::near(inv, expected), "known inverse coefficients");
    r.check(tx::inverse_residual(a, inv) < tx::tolerance<T>, "known inverse both residuals");
    r.check(tx::near(rm::inverse(a), expected), "inverse convenience wrapper");
    M alias = a;
    r.check(rm::try_inverse(alias, alias) && tx::near(alias, expected), "inverse supports input/output aliasing");
    const M permutation{T(0),T(1),T(0), T(0),T(0),T(1), T(1),T(0),T(0)};
    r.check(rm::try_inverse(permutation, inv) && tx::equal(inv, rm::transpose(permutation)), "zero diagonal needs pivoting");
    const M scaled_permutation{T(0),T(2),T(0), T(0),T(0),T(4), T(8),T(0),T(0)};
    r.check(rm::try_inverse(scaled_permutation, inv) && tx::inverse_residual(scaled_permutation,inv) == 0.0L,
            "pivot swaps must not reorder original row scales");
    const M scaling{T(2),T(0),T(0), T(0),T(3),T(0), T(0),T(0),T(4)};
    const V v{T(1),T(2),T(3)};
    r.check(tx::near((a * scaling) * v, a * (scaling * v)), "composition applies right transform first");
    r.check(!tx::near(a * scaling, scaling * a), "matrix composition is noncommutative");
    M self = a; self *= self;
    r.check(tx::near(self,a*a), "matrix compound product self-aliasing");

    const T large = std::ldexp(T(1), std::same_as<T,float> ? 100 : 800);
    const T small = std::ldexp(T(1), std::same_as<T,float> ? -100 : -800);
    for (const T scale : {large, small, T(-1)}) {
        const M m(scale);
        r.check(rm::try_inverse(m,inv) && tx::inverse_residual(m,inv) == 0.0L,
                "inverse does not use overflowed/underflowed determinant");
    }
    const M anisotropic{large,T(0),T(0), T(0),small,T(0), T(0),T(0),T(1)};
    r.check(rm::try_inverse(anisotropic, inv) && tx::inverse_residual(anisotropic,inv) == 0.0L,
            "row equilibration preserves extreme diagonal scales");
    const T maximum = (std::numeric_limits<T>::max)();
    r.check(rm::try_inverse(M(maximum),inv) && rm::is_all_finite(inv) && inv(0,0) > T(0),
            "finite large matrix can have subnormal inverse entries");
    const T tiny = std::numeric_limits<T>::denorm_min();
    r.check(rm::detail::div_rn(tiny,tiny) == T(1), "division must not form overflowing reciprocal");
    r.check(std::signbit(rm::detail::div_rn(-T(0),T(2))), "rounded division preserves negative zero");

    const M sentinel{T(11),T(12),T(13), T(14),T(15),T(16), T(17),T(18),T(19)};
    const M singular{T(1),T(2),T(3), T(1),T(2),T(3), T(4),T(5),T(6)};
    for (const M& bad : {M{},singular,M(tiny)}) {
        inv = sentinel;
        r.check(!rm::try_inverse(bad,inv) && tx::equal(inv,sentinel), "failed inverse preserves whole output");
        const auto invalid = rm::inverse(bad);
        bool all_nan = true;
        for (const T x : invalid.elements) all_nan = all_nan && std::isnan(x);
        r.check(all_nan, "failed inverse wrapper fills every entry with NaN");
    }
    for (int index = 0; index < 9; ++index) {
        for (const T invalid : {std::numeric_limits<T>::quiet_NaN(),std::numeric_limits<T>::infinity()}) {
            M bad = identity; bad.elements[index] = invalid; inv = sentinel;
            r.check(!rm::try_inverse(bad,inv) && tx::equal(inv,sentinel), "reject nonfinite matrix entry");
        }
    }
    for (const T bad_limit : {T(-1),T(1),std::numeric_limits<T>::quiet_NaN(),std::numeric_limits<T>::infinity()}) {
        inv = sentinel;
        r.check(!rm::try_inverse(identity,inv,bad_limit) && tx::equal(inv,sentinel), "invalid residual policy preserves output");
    }
    r.check(rm::try_inverse(identity,inv,T(0)), "exact identity permits zero residual tolerance");
    // All off-diagonal entries are <= 1/4, so these matrices stay well-conditioned.
    tx::sequence rng;
    for (int trial = 0; trial < 1024; ++trial) {
        M m;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                m(row,col) = rng.next<T>() / T(4) + (row == col ? T(2) : T(0));
        const bool ok = rm::try_inverse(m,inv);
        const long double residual = ok ? tx::inverse_residual(m,inv) : 1.0L;
        r.check(ok && residual < tx::tolerance<T>, "random inverse left/right residual");
        if (!ok || residual >= tx::tolerance<T>)
            std::fprintf(stderr,"  trial=%d residual=%Lg\n",trial,residual);
    }
    if (r.failures != before) std::fprintf(stderr,"  matrix scalar=%s\n", std::same_as<T,float> ? "float" : "double");
}
} // namespace
int main() {
    ts::test_runner r;
    r.check(std::fegetround() == FE_TONEAREST,"matrix tests require round-to-nearest");
    checks<float>(r); checks<double>(r);
    return r.finish("All matrix construction, composition, and inverse checks passed.");
}
