include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

add_executable(test_llm ${TEST_SOURCES})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
target_link_libraries(test_llm ${link_ext_lib})
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(test_llm PUBLIC ../kuiper)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(llama absl::base re2::re2 nlohmann_json::nlohmann_json)
endif ()
target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
