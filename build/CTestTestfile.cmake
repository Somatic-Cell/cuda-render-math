# CMake generated Testfile for 
# Source directory: C:/Users/sy415/workspace/cuda_render_math
# Build directory: C:/Users/sy415/workspace/cuda_render_math/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[cuda_render_math.vector]=] "C:/Users/sy415/workspace/cuda_render_math/build/Debug/cuda_render_math_vector_test.exe")
  set_tests_properties([=[cuda_render_math.vector]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/sy415/workspace/cuda_render_math/CMakeLists.txt;46;add_test;C:/Users/sy415/workspace/cuda_render_math/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[cuda_render_math.vector]=] "C:/Users/sy415/workspace/cuda_render_math/build/Release/cuda_render_math_vector_test.exe")
  set_tests_properties([=[cuda_render_math.vector]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/sy415/workspace/cuda_render_math/CMakeLists.txt;46;add_test;C:/Users/sy415/workspace/cuda_render_math/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[cuda_render_math.vector]=] "C:/Users/sy415/workspace/cuda_render_math/build/MinSizeRel/cuda_render_math_vector_test.exe")
  set_tests_properties([=[cuda_render_math.vector]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/sy415/workspace/cuda_render_math/CMakeLists.txt;46;add_test;C:/Users/sy415/workspace/cuda_render_math/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[cuda_render_math.vector]=] "C:/Users/sy415/workspace/cuda_render_math/build/RelWithDebInfo/cuda_render_math_vector_test.exe")
  set_tests_properties([=[cuda_render_math.vector]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/sy415/workspace/cuda_render_math/CMakeLists.txt;46;add_test;C:/Users/sy415/workspace/cuda_render_math/CMakeLists.txt;0;")
else()
  add_test([=[cuda_render_math.vector]=] NOT_AVAILABLE)
endif()
