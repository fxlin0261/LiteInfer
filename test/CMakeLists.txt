set(
  TEST_SOURCES
  support/test_main.cpp
  base/test_unicode.cpp
  model/test_llama_model_cpu.cpp
  model/test_model_core.cpp
  model/test_model_load.cpp
  model/test_raw_model_data.cpp
  op/cpu/test_embedding_cpu.cpp
  op/cpu/test_mha_cpu.cpp
  sampler/test_argmax_sampler.cpp
)

if(KUIPER_WITH_CUDA)
  list(
    APPEND
    TEST_SOURCES
    base/test_buffer.cpp
    base/test_buffer_core.cpp
    tensor/test_tensor.cpp
    support/test_cuda_utils.cpp
    support/cuda_test_utils.cu
    op/cuda/test_add_cuda.cpp
    op/cuda/test_embedding_cuda.cpp
    op/cuda/test_matmul_cuda.cpp
    op/cuda/test_rmsnorm_cuda.cpp
    op/cuda/test_swiglu_cuda.cpp
  )
endif()

add_executable(test_llm ${TEST_SOURCES})
target_include_directories(test_llm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_llm PRIVATE kuiper::runtime GTest::gtest)

if(KUIPER_WITH_CUDA)
  set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_test(NAME test_llm COMMAND test_llm)
