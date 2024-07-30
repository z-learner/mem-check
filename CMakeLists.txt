cmake_minimum_required(VERSION 3.12)
project(mem-check LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)

if (NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE "Release")
    # add -g flag for release build
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")
endif()

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -Wpedantic)

add_link_options(-rdynamic)

add_library(mem-check SHARED src/mem-check.cc)
add_library(mem-plot SHARED src/mem-hook.cc src/plot-actions.cc)

find_package(Threads REQUIRED)
target_link_libraries(mem-check PUBLIC Threads::Threads)
target_link_libraries(mem-plot PUBLIC Threads::Threads)

find_package(CUDA)
if (CUDA_FOUND)
    message(STATUS "CUDA found")
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA include path: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA library path: ${CUDA_LIBRARIES}")
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    target_include_directories(mem-plot PUBLIC ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(mem-plot PUBLIC __ENABLE_CUDA__)
endif()

target_include_directories(mem-check PUBLIC include)
target_include_directories(mem-plot PUBLIC include)

add_subdirectory(test)
add_subdirectory(examples)