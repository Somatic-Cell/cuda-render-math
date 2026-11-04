#include <cuda_render_math/affine_transform.hpp>
#include <cuda_render_math/orthonormal_basis.hpp>

namespace rm = cuda_render_math;

template<rm::real_scalar T>
struct transform_check_result {
    rm::mat3<T> matrices[6];
    rm::affine_transform3<T> transforms[3];
    rm::orthonormal_basis3<T> basis;
    rm::vec3<T> vectors[6];
    T determinant;
    bool accepted[4];
};

template<rm::real_scalar T>
__host__ __device__ void exercise_transform(
    transform_check_result<T>& out, rm::mat3<T> a, rm::mat3<T> b,
    rm::vec3<T> point, rm::vec3<T> normal, rm::vec3<T> tangent) {
    out.matrices[0] = a*b;
    out.matrices[1] = rm::transpose(a);
    out.accepted[0] = rm::try_inverse(a,out.matrices[2]);
    out.matrices[3] = rm::inverse(b);
    out.matrices[4] = (a+b-a)*T(2)/T(2);
    out.matrices[4] *= b;
    out.determinant = rm::determinant(a);
    out.transforms[0] = rm::affine_transform3<T>{a,point};
    out.accepted[1] = rm::try_inverse(out.transforms[0],out.transforms[1]);
    out.transforms[2] = out.transforms[0] * rm::inverse(out.transforms[0]);
    out.vectors[0] = rm::transform_point(out.transforms[0],point);
    out.vectors[1] = rm::transform_vector(out.transforms[0],tangent);
    out.vectors[2] = rm::transform_normal_from_inverse(out.transforms[1],normal);
    out.matrices[5] = rm::normal_matrix_from_inverse(out.transforms[1]);
    out.basis = rm::make_orthonormal_basis(normal);
    out.accepted[2] = rm::try_make_orthonormal_basis(normal,tangent,out.basis);
    out.accepted[3] = rm::is_orthonormal(out.basis);
    out.vectors[3] = rm::to_world(out.basis,point);
    out.vectors[4] = rm::to_local(out.basis,point);
    out.vectors[5] = rm::to_matrix(out.basis) * point;
}

// Compile-only: one output record per kernel; caller provides valid inputs if run.
// No CUDA Runtime API calls or launches. Failed try_* outputs are initialized below.
extern "C" __global__ void cuda_render_math_transform_float(
    transform_check_result<float>* out, rm::mat3f a, rm::mat3f b,
    rm::vec3f p, rm::vec3f n, rm::vec3f tangent) {
    *out = transform_check_result<float>{};
    exercise_transform(*out,a,b,p,n,tangent);
}
extern "C" __global__ void cuda_render_math_transform_double(
    transform_check_result<double>* out, rm::mat3d a, rm::mat3d b,
    rm::vec3d p, rm::vec3d n, rm::vec3d tangent) {
    *out = transform_check_result<double>{};
    exercise_transform(*out,a,b,p,n,tangent);
}
__host__ void cuda_render_math_transform_host(transform_check_result<float>* out) {
    *out = transform_check_result<float>{};
    exercise_transform(*out,rm::mat3f(1.0f),rm::mat3f(2.0f),rm::vec3f(1.0f),
                       rm::vec3f{0.0f,0.0f,1.0f},rm::vec3f{1.0f,0.0f,0.0f});
}
