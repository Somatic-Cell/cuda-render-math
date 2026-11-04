cmake_minimum_required(VERSION 3.26)

project(cuda_render_math VERSION 0.1.0 LANGUAGES CXX)

add_library(cuda_render_math INTERFACE)
add_library(cuda_render_math::cuda_render_math ALIAS cuda_render_math)

target_compile_features(cuda_render_math INTERFACE cxx_std_20)
target_include_directories(cuda_render_math INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# ---------------------
# テスト
# ---------------------
option(CUDA_RENDER_MATH_BUILD_TESTS
    "BUild the host contract tests" ${PROJECT_IS_TOP_LEVEL}
)
option(CUDA_RENDER_MATH_BUILD_CUDA_CHECK
    "Compile the header checks to PTX without GPU execution" OFF
)

if(CUDA_RENDER_MATH_BUILD_TESTS)
    enable_testing()
endif()

if(CUDA_RENDER_MATH_BUILD_TESTS OR CUDA_RENDER_MATH_BUILD_CUDA_CHECK)
    add_subdirectory(tests)
endif()

    