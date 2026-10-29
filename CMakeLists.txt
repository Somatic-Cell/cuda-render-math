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

    function(cuda_render_math_configure_test target)
        target_link_libraries(${target}
        PRIVATE cuda_render_math::cuda_render_math
        )
        set_target_properties(${target} PROPERTIES
            CXX_EXTENSIONS OFF
        )
        # コンパイルオプション
        if(MSVC)
            target_compile_options(${target} PRIVATE
                /W4 /utf-8 /permissive- /Zc:__cplusplus /Zc:preprocessor)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE
                -Wall -Wextra -Wpedantic -Wconversion)
        endif()
    endfunction()

    # ベクトルの構築に関するテスト
    add_executable(vector_construction_test
        tests/vector_construction_test.cpp
    )
    cuda_render_math_configure_test(vector_construction_test)
    add_test(NAME cuda_render_math.vector_construction COMMAND vector_construction_test)

    # 演算に関するテスト
    # ベクトル
    add_executable(vector_arithmetic_test
        tests/vector_arithmetic_test.cpp 
        tests/vector_linkage.cpp
    )
    cuda_render_math_configure_test(vector_arithmetic_test)
    add_test(NAME cuda_render_math.vector_arithmetic COMMAND vector_arithmetic_test)

    add_executable(vector_geometry_test
        tests/vector_geometry_test.cpp)
    cuda_render_math_configure_test(vector_geometry_test)
    add_test(NAME cuda_render_math.vector_geometry
        COMMAND vector_geometry_test)

    add_executable(vector_component_test 
        tests/vector_component_test.cpp
    )
    cuda_render_math_configure_test(vector_component_test)
    add_test(NAME cuda_render_math.vector_component 
        COMMAND vector_component_test
    )

    add_executable(integer_vector_test tests/integer_vector_test.cpp)
    cuda_render_math_configure_test(integer_vector_test)
    add_test(NAME cuda_render_math.integer_vector COMMAND integer_vector_test)



    # スカラ
    add_executable(scalar_math_test
        tests/scalar_math_test.cpp 
        tests/scalar_math_linkage.cpp
    )
    cuda_render_math_configure_test(scalar_math_test)
    add_test(NAME cuda_render_math.scalar_math
        COMMAND scalar_math_test)

    add_executable(scalar_utilities_test 
        tests/scalar_utilities_test.cpp
    )
    cuda_render_math_configure_test(scalar_utilities_test)
    add_test(NAME cuda_render_math.scalar_utilities 
        COMMAND scalar_utilities_test
    )

    # 型のキャスト
    add_executable(numeric_cast_test tests/numeric_cast_test.cpp)
    cuda_render_math_configure_test(numeric_cast_test)
    add_test(NAME cuda_render_math.numeric_cast COMMAND numeric_cast_test)

    

    # 意図的に多重インクルードさせるテスト
    set(header_check_sources)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/header_checks")
    foreach(header IN ITEMS config scalar scalar_math vector)
        set(source "${CMAKE_CURRENT_BINARY_DIR}/${header}.cpp")
        file(WRITE "${source}" "#include <cuda_render_math/${header}.hpp>\n" "#include <cuda_render_math/${header}.hpp>\n")
        list(APPEND header_check_sources "${source}")
    endforeach()
    add_library(cuda_render_math_header_checks OBJECT 
        ${header_check_sources} 
        tests/minmax_macro_check.cpp
        tests/integer_minmax_macro_check.cpp
    )
    cuda_render_math_configure_test(cuda_render_math_header_checks)    
endif()

if(CUDA_RENDER_MATH_BUILD_CUDA_CHECK)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
            OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
        message(FATAL_ERROR "The CUDA check requires NVIDIA nvcc 12.0 or newer.")
    endif()
    add_library(cuda_render_math_cuda_check OBJECT 
        tests/cuda_compile_check.cu
        tests/cuda_component_check.cu
        tests/cuda_integer_check.cu
    )
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