function(kuiper_add_model_target target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PRIVATE kuiper::llama)
endfunction()

kuiper_add_model_target(llama_infer llama3/infer_llama3.cpp)
kuiper_add_model_target(qwen_infer qwen2/infer_qwen2.cpp)
kuiper_add_model_target(qwen3_infer qwen3/infer_qwen3.cpp)
