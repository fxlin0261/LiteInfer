cmake_minimum_required(VERSION 3.16)

project(LiteInfer LANGUAGES CXX)

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0169 OLD)
endif()

set(LITEINFER_CUDA_MODE "AUTO" CACHE STRING "CUDA build mode: AUTO, CPU, or CUDA")
set_property(CACHE LITEINFER_CUDA_MODE PROPERTY STRINGS AUTO CPU CUDA)
string(TOUPPER "${LITEINFER_CUDA_MODE}" LITEINFER_CUDA_MODE)
set(LITEINFER_CUDA_MODE "${LITEINFER_CUDA_MODE}" CACHE STRING "CUDA build mode: AUTO, CPU, or CUDA" FORCE)

set(LITEINFER_VALID_CUDA_MODES AUTO CPU CUDA)
list(FIND LITEINFER_VALID_CUDA_MODES "${LITEINFER_CUDA_MODE}" LITEINFER_CUDA_MODE_INDEX)
if(LITEINFER_CUDA_MODE_INDEX EQUAL -1)
  message(FATAL_ERROR "Invalid LITEINFER_CUDA_MODE='${LITEINFER_CUDA_MODE}'. Expected AUTO, CPU, or CUDA.")
endif()

if(LITEINFER_CUDA_MODE STREQUAL "CPU")
  set(LITEINFER_WITH_CUDA OFF)
  message(STATUS "CUDA disabled via LITEINFER_CUDA_MODE=CPU; building CPU-only.")
else()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(LITEINFER_WITH_CUDA ON)
  elseif(LITEINFER_CUDA_MODE STREQUAL "CUDA")
    message(FATAL_ERROR "LITEINFER_CUDA_MODE=CUDA was requested, but no CUDA compiler was found.")
  else()
    set(LITEINFER_WITH_CUDA OFF)
    message(STATUS "CUDA compiler not found; building CPU-only.")
  endif()
endif()

include(CTest)
if(LITEINFER_WITH_CUDA)
  include(cmake/cuda.cmake)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(LITEINFER_WITH_CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

option(USE_CPM "Use CPM for dependency management" OFF)
set(LITEINFER_ENABLE_BPE_TOKENIZERS ON)

function(liteinfer_maybe_disable_third_party_warnings)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    return()
  endif()

  foreach(target_name IN LISTS ARGN)
    if(TARGET ${target_name})
      get_target_property(target_is_imported ${target_name} IMPORTED)
      if(NOT target_is_imported)
        target_compile_options(
          ${target_name}
          PRIVATE "-Wno-#pragma-messages" -Wno-unused-const-variable
                  -Wno-inconsistent-missing-override
        )
      endif()
    endif()
  endforeach()
endfunction()

if(USE_CPM)
  if(DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(LITEINFER_PREV_POLICY_VERSION_MINIMUM "${CMAKE_POLICY_VERSION_MINIMUM}")
    set(LITEINFER_RESTORE_POLICY_VERSION_MINIMUM ON)
  else()
    set(LITEINFER_RESTORE_POLICY_VERSION_MINIMUM OFF)
  endif()
  set(CMAKE_POLICY_VERSION_MINIMUM 3.10)

  if(DEFINED CMAKE_WARN_DEPRECATED)
    set(LITEINFER_PREV_WARN_DEPRECATED "${CMAKE_WARN_DEPRECATED}")
    set(LITEINFER_RESTORE_WARN_DEPRECATED ON)
  else()
    set(LITEINFER_RESTORE_WARN_DEPRECATED OFF)
  endif()
  set(CMAKE_WARN_DEPRECATED OFF)

  include(cmake/CPM.cmake)

  if(BUILD_TESTING)
    CPMAddPackage(
      NAME GTest
      GITHUB_REPOSITORY google/googletest
      VERSION 1.15.0
    )
  endif()

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME absl
    GITHUB_REPOSITORY abseil/abseil-cpp
    GIT_TAG 20240722.0
    OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  if(LITEINFER_ENABLE_BPE_TOKENIZERS)
    CPMAddPackage(
      NAME re2
      GITHUB_REPOSITORY google/re2
      GIT_TAG 2024-07-02
    )

    CPMAddPackage(
      NAME nlohmann_json
      GITHUB_REPOSITORY nlohmann/json
      VERSION 3.11.3
    )

    CPMAddPackage(
      NAME unordered_dense
      GITHUB_REPOSITORY martinus/unordered_dense
      GIT_TAG v4.1.2
    )
  endif()

  if(LITEINFER_RESTORE_WARN_DEPRECATED)
    set(CMAKE_WARN_DEPRECATED "${LITEINFER_PREV_WARN_DEPRECATED}")
  else()
    unset(CMAKE_WARN_DEPRECATED)
  endif()

  if(LITEINFER_RESTORE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM "${LITEINFER_PREV_POLICY_VERSION_MINIMUM}")
  else()
    unset(CMAKE_POLICY_VERSION_MINIMUM)
  endif()
endif()

if(BUILD_TESTING AND NOT TARGET GTest::gtest)
  find_package(GTest REQUIRED)
endif()

if(NOT TARGET glog::glog)
  find_package(glog REQUIRED)
endif()

if(NOT TARGET armadillo AND NOT TARGET Armadillo::Armadillo)
  find_package(Armadillo REQUIRED)
endif()

if(NOT TARGET absl::status)
  find_package(absl REQUIRED)
endif()

if(LITEINFER_ENABLE_BPE_TOKENIZERS)
  if(NOT TARGET re2::re2)
    find_package(re2 REQUIRED)
  endif()

  if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json REQUIRED)
  endif()

  if(NOT TARGET unordered_dense::unordered_dense AND NOT TARGET unordered_dense)
    find_package(unordered_dense CONFIG REQUIRED)
  endif()
endif()

liteinfer_maybe_disable_third_party_warnings(
  glog_internal
)

add_library(liteinfer_armadillo INTERFACE)
if(TARGET Armadillo::Armadillo)
  target_link_libraries(liteinfer_armadillo INTERFACE Armadillo::Armadillo)
elseif(TARGET armadillo)
  target_link_libraries(liteinfer_armadillo INTERFACE armadillo)
elseif(ARMADILLO_LIBRARIES)
  target_link_libraries(liteinfer_armadillo INTERFACE ${ARMADILLO_LIBRARIES})
  if(ARMADILLO_INCLUDE_DIRS)
    target_include_directories(liteinfer_armadillo INTERFACE ${ARMADILLO_INCLUDE_DIRS})
  elseif(ARMADILLO_INCLUDE_DIR)
    target_include_directories(liteinfer_armadillo INTERFACE ${ARMADILLO_INCLUDE_DIR})
  endif()
else()
  message(FATAL_ERROR "Unable to find a usable Armadillo target or library.")
endif()

if(LITEINFER_ENABLE_BPE_TOKENIZERS)
  add_library(liteinfer_unordered_dense INTERFACE)
  if(TARGET unordered_dense::unordered_dense)
    target_link_libraries(liteinfer_unordered_dense INTERFACE unordered_dense::unordered_dense)
  elseif(TARGET unordered_dense)
    target_link_libraries(liteinfer_unordered_dense INTERFACE unordered_dense)
  else()
    message(FATAL_ERROR "Unable to find a usable unordered_dense target.")
  endif()
endif()

set(
  LITEINFER_INFER_TARGET_SOURCES
  "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama3_infer_demo.cpp"
  "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama3_infer_prompt_demo.cpp"
)

file(GLOB_RECURSE LITEINFER_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/liteinfer/*.cpp")
if(LITEINFER_WITH_CUDA)
  file(GLOB_RECURSE LITEINFER_CUDA_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/liteinfer/*.cu")
  list(APPEND LITEINFER_SOURCES ${LITEINFER_CUDA_SOURCES})
endif()
list(REMOVE_ITEM LITEINFER_SOURCES ${LITEINFER_INFER_TARGET_SOURCES})

add_library(liteinfer_runtime SHARED ${LITEINFER_SOURCES})
add_library(liteinfer::runtime ALIAS liteinfer_runtime)

target_compile_features(liteinfer_runtime PUBLIC cxx_std_17)
target_include_directories(liteinfer_runtime PUBLIC "${PROJECT_SOURCE_DIR}/liteinfer")
target_link_libraries(liteinfer_runtime PUBLIC glog::glog absl::status liteinfer_armadillo)

if(LITEINFER_WITH_CUDA)
  target_link_libraries(liteinfer_runtime PUBLIC ${LITEINFER_CUDART_TARGET})
  target_compile_definitions(liteinfer_runtime PUBLIC LITEINFER_ENABLE_CUDA=1)
  set_target_properties(liteinfer_runtime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
  target_compile_definitions(liteinfer_runtime PUBLIC LITEINFER_ENABLE_CUDA=0)
endif()

if(LITEINFER_ENABLE_BPE_TOKENIZERS)
  target_link_libraries(
    liteinfer_runtime PUBLIC
    re2::re2
    nlohmann_json::nlohmann_json
    liteinfer_unordered_dense
  )
endif()

if(WIN32)
  set_target_properties(liteinfer_runtime PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

function(liteinfer_add_model_target target_name source_file)
  add_executable(${target_name} ${source_file})
  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  target_link_libraries(${target_name} PRIVATE liteinfer::runtime)
endfunction()

liteinfer_add_model_target(llama3_infer_demo "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama3_infer_demo.cpp")
liteinfer_add_model_target(llama3_infer_prompt_demo "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama3_infer_prompt_demo.cpp")
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
