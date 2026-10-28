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
    add_executable(cuda_render_math_vector_test
        tests/vector_test.cpp
        tests/vector_odr.cpp
    )
    target_link_libraries(cuda_render_math_vector_test
        PRIVATE cuda_render_math::cuda_render_math
    )
    set_target_properties(cuda_render_math_vector_test PROPERTIES
        CXX_EXTENSIONS OFF
    )

    # コンパイルオプション
    if(MSVC)
        target_compile_options(cuda_render_math_vector_test PRIVATE
            /W4 /utf-8 /permissive- /Zc:__cplusplus /Zc:preprocessor)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(cuda_render_math_vector_test PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion)
    endif()

    add_test(NAME cuda_render_math.vector COMMAND cuda_render_math_vector_test)
endif()

if(CUDA_RENDER_MATH_BUILD_CUDA_CHECK)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
            OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
        message(FATAL_ERROR "The CUDA check requires NVIDIA nvcc 12.0 or newer.")
    endif()
    add_library(cuda_render_math_cuda_check OBJECT tests/cuda_compile_check.cu)
    target_link_libraries(cuda_render_math_cuda_check
        PRIVATE cuda_render_math::cuda_render_math)
    set_target_properties(cuda_render_math_cuda_check PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS OFF
        CUDA_PTX_COMPILATION ON
        CUDA_RUNTIME_LIBRARY None
    )
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND MSVC)
        target_compile_options(cuda_render_math_cuda_check PRIVATE
            -Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:__cplusplus)
    endif()
endif()