#include <cuda_render_math/affine_transform.hpp>
#include <cuda_render_math/orthonormal_basis.hpp>
#include "transform_test_support.hpp"
#include <initializer_list>
#include <type_traits>

namespace rm = cuda_render_math;
namespace ts = test_support;
namespace tx = transform_test_support;
namespace {
template<typename A, typename V> concept can_apply = requires(A a,V v) { a*v; };
static_assert(!can_apply<rm::affine_transform3f,rm::vec3f>); // Point or vector must be explicit.
static_assert(std::is_trivially_copyable_v<rm::affine_transform3f>);
static_assert(std::is_standard_layout_v<rm::affine_transform3d>);
constexpr rm::affine_transform3f identity;
static_assert(identity.linear(0,0)==1 && identity.translation.x==0);

template<typename T>
bool equal(rm::affine_transform3<T> a,rm::affine_transform3<T> b) {
    return tx::equal(a.linear,b.linear) && ts::equal(a.translation,b.translation);
}
template<typename T>
void checks(ts::test_runner& r) {
    using A=rm::affine_transform3<T>; using M=rm::mat3<T>; using V=rm::vec3<T>;
    const int before=r.failures;
    const A a{{T(2),T(1),T(0),T(0),T(3),T(0),T(0),T(0),T(4)}, {T(5),T(-2),T(7)}};
    const A b{M(T(2)), {T(1),T(2),T(3)}};
    const V p{T(1),T(2),T(-1)}, d{T(-1),T(1),T(2)};
    r.check(tx::near(rm::transform_point(A{},p),p), "default affine is identity");
    r.check(tx::near(rm::transform_point(a,p),a.linear*p+a.translation), "point includes translation");
    r.check(tx::near(rm::transform_vector(a,d),a.linear*d), "vector ignores translation");
    r.check(tx::near(rm::transform_point(a*b,p),rm::transform_point(a,rm::transform_point(b,p))), "affine composition order");
    r.check(!tx::near(rm::transform_point(a*b,p),rm::transform_point(b*a,p)), "affine order matters");
    const T ray_parameter=T(0.75);
    r.check(tx::near(rm::transform_point(a,p+d*ray_parameter),
                    rm::transform_point(a,p)+rm::transform_vector(a,d)*ray_parameter), "ray t is unchanged (no normalization)");
    A inv;
    r.check(rm::try_inverse(a,inv), "affine inverse succeeds");
    r.check(tx::near(rm::transform_point(inv,rm::transform_point(a,p)),p), "inverse point round trip");
    r.check(tx::near(rm::transform_vector(inv,rm::transform_vector(a,d)),d), "inverse vector round trip");
    r.check(tx::near((a*inv).linear,M(T(1))) && tx::near((a*inv).translation,V{}), "compose with inverse identity");
    A alias=a;
    r.check(rm::try_inverse(alias,alias) && tx::near(alias.linear,inv.linear) &&
            tx::near(alias.translation,inv.translation), "affine inverse self-aliasing");
    alias=a; alias*=alias;
    r.check(tx::near(alias.linear,(a*a).linear) && tx::near(alias.translation,(a*a).translation), "affine product self-aliasing");
    const V tangent{T(1),T(0),T(1)}, normal{T(1),T(2),T(-1)};
    const V nt=rm::transform_normal_from_inverse(inv,normal);
    r.check(tx::near(rm::dot(nt,rm::transform_vector(a,tangent)),T(0)), "inverse transpose preserves normal orthogonality");
    r.check(tx::near(rm::normal_matrix_from_inverse(inv)*normal,nt), "cached normal matrix agrees");
    r.check(!tx::near(rm::dot(rm::transform_vector(a,normal),rm::transform_vector(a,tangent)),T(0)),
            "normal must not use ordinary vector transform under shear/scale");
    const A reflection{{T(-1),T(0),T(0), T(0),T(1),T(0), T(0),T(0),T(1)}, {}};
    const V ex{T(1),T(0),T(0)}, ey{T(0),T(1),T(0)}, ez{T(0),T(0),T(1)};
    r.check(tx::near(rm::transform_normal_from_inverse(reflection,ez),ez) &&
            tx::near(rm::cross(rm::transform_vector(reflection,ex),rm::transform_vector(reflection,ey)),-ez),
            "negative determinant distinguishes covector normal from winding orientation");
    // LTC-style chain: a view-aligned frame and an already prepared inverse
    // matrix. This is transform coverage, not a BRDF-fit or lighting-integral test.
    rm::orthonormal_basis3<T> frame;
    r.check(rm::try_make_orthonormal_basis(V{T(0),T(1),T(1)},V{T(1),T(2),T(3)},frame),
            "LTC view-aligned basis construction");
    const M warp{T(1.5),T(0),T(0.3), T(0),T(0.75),T(0), T(0.2),T(0),T(1)};
    M inverse_warp;
    const bool warp_ok=rm::try_inverse(warp,inverse_warp);
    r.check(warp_ok,"LTC linear inverse preparation");
    if (warp_ok) {
        const M world_to_base=inverse_warp*rm::transpose(rm::to_matrix(frame));
        const V offset{T(2),T(-1),T(4)};
        const V mapped=world_to_base*offset;
        r.check(tx::near(mapped,inverse_warp*rm::to_local(frame,offset)),
                "precomputed inverse and frame composition agree");
        r.check(tx::near(rm::normalize(rm::to_world(frame,warp*rm::normalize(mapped))),
                        rm::normalize(offset)),"normalized linear spherical map round trip");
    }
    const A sentinel{M(T(7)),V(T(11))};
    for (const A& bad : {A{M{},V{}},A{M(T(1)),V(std::numeric_limits<T>::infinity())},
            A{M(T(0.5)),V((std::numeric_limits<T>::max)())}}) {
        inv=sentinel;
        r.check(!rm::try_inverse(bad,inv) && equal(inv,sentinel), "failed affine inverse preserves output");
        const auto invalid=rm::inverse(bad);
        bool all_nan=ts::all_components(invalid.translation,[](T x){return std::isnan(x);});
        for (T x : invalid.linear.elements) all_nan=all_nan&&std::isnan(x);
        r.check(all_nan,"failed affine inverse fills all coefficients with NaN");
    }
    if (r.failures!=before) std::fprintf(stderr,"  affine scalar=%s\n",std::same_as<T,float> ? "float" : "double");
}
} // namespace
int main() {
    ts::test_runner r; checks<float>(r); checks<double>(r);
    return r.finish("All affine point, vector, normal, and inverse checks passed.");
}
