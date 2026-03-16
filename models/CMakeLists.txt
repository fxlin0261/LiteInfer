function(liteinfer_add_model_target target_name source_file)
  add_executable(${target_name} ${source_file})
  target_include_directories(${target_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${target_name} PRIVATE liteinfer::runtime)
endfunction()

liteinfer_add_model_target(llama2_infer llama/infer_llama2.cpp)
liteinfer_add_model_target(llama3_infer llama/infer_llama3.cpp)
