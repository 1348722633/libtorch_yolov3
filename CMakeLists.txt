cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(simnet)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB INC ${CMAKE_CURRENT_LIST_DIR}/include/*.hpp)
file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(simnet ${INC} ${SRC})
target_link_libraries(simnet ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET simnet PROPERTY CXX_STANDARD 11)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
