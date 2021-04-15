cmake_minimum_required(VERSION 3.10)
project(profiling)

if (NOT TORCH_DIR)
  message(FATAL_ERROR "Please setup TORCH_DIR")
else()
  message("Found TORCH_DIR:${TORCH_DIR}")
endif()


if (NOT STEP)
  message(FATAL_ERROR "Please setup STEP, e.g. step_two")
else()
  message("Using STEP: ${STEP}")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

include_directories(${TORCH_DIR}/include ${TORCH_DIR}/include/torch/csrc/api/include)
set(TORCH_LIBRARIES
    "${TORCH_DIR}/lib/libtorch.so"
    "${TORCH_DIR}/lib/libtorch_cpu.so"
    "${TORCH_DIR}/lib/libc10.so"
)

include_directories(${CMAKE_SOURCE_DIR}/../${STEP}/)
add_executable(test main.cpp)
target_link_libraries(test PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET test PROPERTY CXX_STANDARD 14)
