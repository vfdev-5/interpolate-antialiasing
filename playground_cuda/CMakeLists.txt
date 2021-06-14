cmake_minimum_required(VERSION 3.10)
# https://devblogs.nvidia.com/building-cuda-applications-cmake/
project(playground LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

find_package(Torch REQUIRED)

add_executable(test main.cpp playground.h playground.cu)
set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET test PROPERTY CXX_STANDARD 14)
