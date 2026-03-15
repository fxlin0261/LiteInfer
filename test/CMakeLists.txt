if(NOT KUIPER_WITH_CUDA)
  message(STATUS "Skipping test target because the current test suite requires CUDA.")
  return()
endif()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

add_executable(test_llm ${TEST_SOURCES})
target_include_directories(test_llm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_llm PRIVATE kuiper::llama GTest::gtest)

set_target_properties(test_llm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_test(NAME test_llm COMMAND test_llm)
