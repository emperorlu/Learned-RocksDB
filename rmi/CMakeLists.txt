cmake_minimum_required (VERSION 3.5)
project (linear)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(JEMALLOC_DIR "/usr/lib/x86_64-linux-gnu")
set(MKL_LINK_DIRECTORY "/opt/intel/mkl/lib/intel64")
set(MKL_INCLUDE_DIRECTORY "/opt/intel/mkl/include")

link_directories(${MKL_LINK_DIRECTORY})


include_directories(${MKL_INCLUDE_DIRECTORY})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add_compile_options(-Wall -fmax-errors=5 -march=native -mtune=native)
add_compile_options(-w -fmax-errors=5 -faligned-new -march=native -mtune=native)

# test
add_executable(test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bytes.cc
)
# string(TOLOWER ${CMAKE_BUILD_TYPE} _type)
# if (_type STREQUAL release)
#     set(default_build_type "Release")
#     message(STATUS "Setting build type to '${default_build_type}' ")
#     target_compile_definitions(test PRIVATE NDEBUGGING)
# else()
#     message(STATUS "Setting build type to '${default_build_type}' ")
# endif()
target_link_libraries(test
    PRIVATE
        mkl_rt
        -lpthread
)