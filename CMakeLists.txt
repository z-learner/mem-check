cmake_minimum_required(VERSION 3.12)
project(mem-check)

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

target_include_directories(mem-check PUBLIC include)
target_include_directories(mem-plot PUBLIC include)

add_subdirectory(test)
add_subdirectory(examples)