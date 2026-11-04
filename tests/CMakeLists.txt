option(CUDA_RENDER_MATH_WARNINGS_AS_ERRORS
    "Treat host test and header-check warnings as errors" OFF)

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
            /W4 /utf-8 /permissive- 
            /Zc:__cplusplus /Zc:preprocessor
            /fp:precise)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -fno-fast-math -ffp-contract=off)
    endif()
    if(CUDA_RENDER_MATH_WARNINGS_AS_ERRORS)
        if(MSVC)
            target_compile_options(${target} PRIVATE /WX)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

function(cuda_render_math_add_test name)
    add_executable(${name}_test ${name}_test.cpp ${ARGN})
    cuda_render_math_configure_test(${name}_test)
    add_test(NAME cuda_render_math.${name} COMMAND ${name}_test)
endfunction()

set(cuda_render_math_check_headers
    config
    scalar
    scalar_cast
    scalar_math
    scalar_numerics
    vector
    vector_cast
    vector_math
    detail/rounded_arithmetic
    matrix
    matrix_numerics
    orthonormal_basis
    affine_transform
)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/header_checks")

if(CUDA_RENDER_MATH_BUILD_TESTS)
    cuda_render_math_add_test(vector_construction)
    cuda_render_math_add_test(vector_arithmetic vector_linkage.cpp)
    cuda_render_math_add_test(vector_geometry)
    cuda_render_math_add_test(vector_component)
    cuda_render_math_add_test(integer_vector)
    cuda_render_math_add_test(scalar_math scalar_math_linkage.cpp)
    cuda_render_math_add_test(scalar_utilities)
    cuda_render_math_add_test(scalar_numerics)
    cuda_render_math_add_test(vector_norm vector_geometry_linkage.cpp)
    cuda_render_math_add_test(vector_numerics)
    cuda_render_math_add_test(rounded_arithmetic)
    cuda_render_math_add_test(numeric_cast)
    cuda_render_math_add_test(matrix)
    cuda_render_math_add_test(orthonormal_basis)
    cuda_render_math_add_test(affine_transform)

    set(header_check_sources)
    foreach(header IN LISTS cuda_render_math_check_headers)
        string(REPLACE "/" "_" stem "${header}")
        set(source "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${stem}.cpp")
        file(WRITE "${source}"
            "#include <cuda_render_math/${header}.hpp>\n"
            "#include <cuda_render_math/${header}.hpp>\n")
        list(APPEND header_check_sources "${source}")
    endforeach()
    add_library(cuda_render_math_header_checks OBJECT
        ${header_check_sources}
        minmax_macro_check.cpp integer_minmax_macro_check.cpp
        geometry_minmax_macro_check.cpp transform_minmax_macro_check.cpp)
    cuda_render_math_configure_test(cuda_render_math_header_checks)

    add_test(NAME cuda_render_math.consumer
        COMMAND "${CMAKE_COMMAND}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/consumer"
            "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/consumer-build"
            "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DGENERATOR=${CMAKE_GENERATOR}"
            "-DGENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DGENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DCONFIG=$<CONFIG>"
            "-DCTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunConsumer.cmake")
endif()

if(CUDA_RENDER_MATH_BUILD_CUDA_CHECK)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR
       CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
        message(FATAL_ERROR "The CUDA check requires NVIDIA nvcc 12.0 or newer.")
    endif()
    set(cuda_header_sources)
    foreach(header IN LISTS cuda_render_math_check_headers)
        string(REPLACE "/" "_" stem "${header}")
        set(source "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${stem}.cu")
        file(WRITE "${source}"
            "#include <cuda_render_math/${header}.hpp>\n"
            "#include <cuda_render_math/${header}.hpp>\n")
        list(APPEND cuda_header_sources "${source}")
    endforeach()
    add_library(cuda_render_math_cuda_check OBJECT
        cuda_compile_check.cu cuda_component_check.cu cuda_integer_check.cu
        cuda_geometry_check.cu cuda_rounding_check.cu cuda_transform_check.cu
        ${cuda_header_sources})
    target_link_libraries(cuda_render_math_cuda_check PRIVATE cuda_render_math::cuda_render_math)
    set_target_properties(cuda_render_math_cuda_check PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES CUDA_EXTENSIONS OFF
        CUDA_PTX_COMPILATION ON CUDA_RUNTIME_LIBRARY None)
    target_compile_options(cuda_render_math_cuda_check PRIVATE
        --ftz=false --prec-div=true --prec-sqrt=true --fmad=true)
    if(MSVC)
        target_compile_options(cuda_render_math_cuda_check PRIVATE
            -Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:__cplusplus -Xcompiler=/fp:precise -Xcompiler=/utf-8)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(cuda_render_math_cuda_check PRIVATE
            -Xcompiler=-fno-fast-math -Xcompiler=-ffp-contract=off)
    endif()
endif()