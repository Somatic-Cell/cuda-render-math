#include <cuda_render_math/orthonormal_basis.hpp>
#include "transform_test_support.hpp"
#include <initializer_list>
#include <type_traits>

namespace rm = cuda_render_math;
namespace ts = test_support;
namespace tx = transform_test_support;
namespace {
template<typename B, typename V> concept can_to_local = requires(B b,V v) { rm::to_local(b,v); };
static_assert(!can_to_local<rm::orthonormal_basis3f,rm::vec3d>);
static_assert(std::is_trivially_copyable_v<rm::orthonormal_basis3f>);
static_assert(std::is_standard_layout_v<rm::orthonormal_basis3d>);
static_assert(sizeof(rm::orthonormal_basis3f) == 9*sizeof(float));
constexpr rm::orthonormal_basis3f identity;
static_assert(identity.x.x == 1 && identity.y.y == 1 && identity.z.z == 1);
static_assert(rm::to_matrix(identity)(1,1) == 1);

template<typename T>
void check_normal(ts::test_runner& r, rm::vec3<T> normal) {
    rm::orthonormal_basis3<T> b;
    const bool ok = rm::try_make_orthonormal_basis(normal,b);
    r.check(ok, "finite nonzero normal accepted");
    if (!ok) return;
    r.check(rm::is_orthonormal(b), "frame is right-handed and orthonormal");
    r.check(tx::near(b.z,rm::normalize(normal)), "local z follows normalized input");
    const rm::vec3<T> v{T(0.3),T(-0.7),T(0.9)};
    r.check(tx::near(rm::to_local(b,rm::to_world(b,v)),v), "local-world-local round trip");
    r.check(tx::near(rm::to_world(b,rm::to_local(b,v)),v), "world-local-world round trip");
    r.check(tx::near(rm::to_world(b,rm::vec3<T>{T(0),T(0),T(1)}),b.z), "basis columns are world axes");
    r.check(tx::near(rm::cross(b.x,b.y),b.z), "basis handedness");
}
template<typename T>
void checks(ts::test_runner& r) {
    using V = rm::vec3<T>; using B = rm::orthonormal_basis3<T>;
    const int before = r.failures;
    const T e = std::numeric_limits<T>::epsilon();
    const T tiny = std::numeric_limits<T>::denorm_min();
    const T high = (std::numeric_limits<T>::max)();
    const V cases[] = {{T(0),T(0),T(1)}, {T(0),T(0),T(-1)},
        {T(1),T(0),T(0)}, {T(1),T(0),-T(0)}, {T(0),T(-1),T(0)},
        {e,e,T(-1)}, {T(0.0001),T(-0.0002),T(-1)},
        {tiny,T(0),T(0)}, {high,high,high}, {T(1),T(2),T(3)}};
    for (const V n : cases) check_normal(r,n);
    const B sentinel{{T(2),T(0),T(0)}, {T(0),T(3),T(0)}, {T(0),T(0),T(4)}};
    const T nan = std::numeric_limits<T>::quiet_NaN(), inf = std::numeric_limits<T>::infinity();
    for (const V bad : {V{},V{nan,T(1),T(2)},V{T(1),inf,T(2)}}) {
        B out = sentinel;
        r.check(!rm::try_make_orthonormal_basis(bad,out) && tx::equal(rm::to_matrix(out),rm::to_matrix(sentinel)),
                "invalid normal preserves output frame");
        const auto invalid = rm::make_orthonormal_basis(bad);
        r.check(ts::all_components(invalid.x,[](T x){return std::isnan(x);}) &&
                ts::all_components(invalid.y,[](T x){return std::isnan(x);}) &&
                ts::all_components(invalid.z,[](T x){return std::isnan(x);}), "failed frame is all NaN");
    }
    B tangent_frame;
    const V n{T(0),T(0),T(2)};
    r.check(rm::try_make_orthonormal_basis(n,V{T(2),T(0),T(5)},tangent_frame) &&
            tx::near(tangent_frame.x,V{T(1),T(0),T(0)}) && rm::is_orthonormal(tangent_frame),
            "tangent projection controls anisotropic x axis");
    for (const V bad : {V{},n,-n,V(inf)}) {
        tangent_frame = sentinel;
        r.check(!rm::try_make_orthonormal_basis(n,bad,tangent_frame) &&
                tx::equal(rm::to_matrix(tangent_frame),rm::to_matrix(sentinel)), "degenerate tangent rejected without fallback");
    }
    tangent_frame = B{};
    r.check(rm::try_make_orthonormal_basis(tangent_frame.z,tangent_frame), "basis accepts input/output aliasing");
    B left; left.x = -left.x;
    r.check(!rm::is_orthonormal(left) && !rm::is_orthonormal(left,T(1)),
            "reject left-handed frame even with a loose tolerance");
    r.check(!rm::is_orthonormal(sentinel), "reject nonunit frame");
    r.check(!rm::is_orthonormal(B{},T(-1)) && !rm::is_orthonormal(B{},nan), "invalid tolerance rejected");
    tx::sequence rng;
    for (int k=0;k<2048;++k) {
        const V sample{rng.next<T>(),rng.next<T>(),rng.next<T>()};
        if (rm::max_abs_component(sample) == T(0)) continue;
        check_normal(r,sample);
    }
    if (r.failures != before) std::fprintf(stderr,"  basis scalar=%s\n",std::same_as<T,float> ? "float" : "double");
}
} // namespace
int main() {
    ts::test_runner r; checks<float>(r); checks<double>(r);
    return r.finish("All orthonormal-basis construction and coordinate checks passed.");
}
