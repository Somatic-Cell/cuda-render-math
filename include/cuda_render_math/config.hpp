#pragma once

#if defined(__CUDACC__)
#define CUDA_RENDER_MATH_HD __host__ __device__
#else
#define CUDA_RENDER_MATH_HD
#endif