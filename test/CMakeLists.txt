find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(. DIR_TEST)

add_executable(test_mem_check ${DIR_TEST})

target_include_directories(test_mem_check PUBLIC ../include)
target_include_directories(test_mem_check PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_mem_check PUBLIC ${GTest_INCLUDE_DIR})

target_link_libraries(test_mem_check glog::glog GTest::gtest)

set_target_properties(test_mem_check PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
# set_target_properties(test_mem_check PROPERTIES CUDA_SEPARABLE_COMPILATION ON)