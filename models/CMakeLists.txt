function(kuiper_add_model_target target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PRIVATE kuiper::runtime)
endfunction()

kuiper_add_model_target(llama2_infer llama/infer_llama2.cpp)
kuiper_add_model_target(llama3_infer llama/infer_llama3.cpp)
kuiper_add_model_target(qwen2_infer qwen/infer_qwen2.cpp)
kuiper_add_model_target(qwen3_infer qwen/infer_qwen3.cpp)
