cmake_minimum_required(VERSION 3.16)

project(KuiperLLama LANGUAGES CXX CUDA)

include(CTest)
include(cmake/cuda.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(USE_CPM "Use CPM for dependency management" OFF)
option(LLAMA3_SUPPORT "Enable Llama 3 tokenizer/model support" OFF)
option(QWEN2_SUPPORT "Enable Qwen 2 tokenizer/model support" OFF)
option(QWEN3_SUPPORT "Enable Qwen 3 tokenizer/model support" OFF)

set(KUIPER_MODEL_DEFINITIONS)
foreach(KUIPER_MODEL_OPTION IN ITEMS LLAMA3_SUPPORT QWEN2_SUPPORT QWEN3_SUPPORT)
  if(${KUIPER_MODEL_OPTION})
    list(APPEND KUIPER_MODEL_DEFINITIONS ${KUIPER_MODEL_OPTION})
  endif()
endforeach()

list(LENGTH KUIPER_MODEL_DEFINITIONS KUIPER_ENABLED_MODEL_SUPPORT_COUNT)
if(KUIPER_ENABLED_MODEL_SUPPORT_COUNT GREATER 1)
  message(FATAL_ERROR
    "Enable at most one of LLAMA3_SUPPORT, QWEN2_SUPPORT, or QWEN3_SUPPORT per build. "
    "These switches change tokenizer and model layout assumptions in shared code.")
endif()

set(KUIPER_ENABLE_BPE_TOKENIZERS OFF)
if(LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
  set(KUIPER_ENABLE_BPE_TOKENIZERS ON)
endif()

if(USE_CPM)
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
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )

  if(KUIPER_ENABLE_BPE_TOKENIZERS)
    CPMAddPackage(
      NAME absl
      GITHUB_REPOSITORY abseil/abseil-cpp
      GIT_TAG 20240722.0
      OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
    )

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

if(NOT TARGET sentencepiece::sentencepiece AND NOT TARGET sentencepiece AND NOT TARGET sentencepiece-static)
  find_package(sentencepiece REQUIRED)
endif()

if(KUIPER_ENABLE_BPE_TOKENIZERS)
  if(NOT TARGET absl::base)
    find_package(absl REQUIRED)
  endif()

  if(NOT TARGET re2::re2)
    find_package(re2 REQUIRED)
  endif()

  if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json REQUIRED)
  endif()
endif()

add_library(kuiper_sentencepiece INTERFACE)
if(TARGET sentencepiece::sentencepiece)
  target_link_libraries(kuiper_sentencepiece INTERFACE sentencepiece::sentencepiece)
elseif(TARGET sentencepiece)
  target_link_libraries(kuiper_sentencepiece INTERFACE sentencepiece)
elseif(TARGET sentencepiece-static)
  target_link_libraries(kuiper_sentencepiece INTERFACE sentencepiece-static)
else()
  message(FATAL_ERROR "Unable to find a usable sentencepiece target.")
endif()

if(DEFINED sentencepiece_SOURCE_DIR AND EXISTS "${sentencepiece_SOURCE_DIR}/src")
  target_include_directories(kuiper_sentencepiece INTERFACE "${sentencepiece_SOURCE_DIR}/src")
endif()

add_library(kuiper_armadillo INTERFACE)
if(TARGET Armadillo::Armadillo)
  target_link_libraries(kuiper_armadillo INTERFACE Armadillo::Armadillo)
elseif(TARGET armadillo)
  target_link_libraries(kuiper_armadillo INTERFACE armadillo)
elseif(ARMADILLO_LIBRARIES)
  target_link_libraries(kuiper_armadillo INTERFACE ${ARMADILLO_LIBRARIES})
  if(ARMADILLO_INCLUDE_DIRS)
    target_include_directories(kuiper_armadillo INTERFACE ${ARMADILLO_INCLUDE_DIRS})
  elseif(ARMADILLO_INCLUDE_DIR)
    target_include_directories(kuiper_armadillo INTERFACE ${ARMADILLO_INCLUDE_DIR})
  endif()
else()
  message(FATAL_ERROR "Unable to find a usable Armadillo target or library.")
endif()

file(GLOB_RECURSE LLAMA_SOURCES CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/kuiper/*.cpp"
  "${PROJECT_SOURCE_DIR}/kuiper/*.cu"
)

add_library(llama SHARED ${LLAMA_SOURCES})
add_library(kuiper::llama ALIAS llama)

target_compile_features(llama PUBLIC cxx_std_17)
target_include_directories(llama PUBLIC "${PROJECT_SOURCE_DIR}/kuiper")
target_link_libraries(llama
  PUBLIC
    ${KUIPER_CUDART_TARGET}
    glog::glog
    kuiper_armadillo
    kuiper_sentencepiece
)

if(KUIPER_MODEL_DEFINITIONS)
  target_compile_definitions(llama PUBLIC ${KUIPER_MODEL_DEFINITIONS})
endif()

if(KUIPER_ENABLE_BPE_TOKENIZERS)
  target_link_libraries(llama
    PUBLIC
      absl::base
      re2::re2
      nlohmann_json::nlohmann_json
  )
endif()

set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(WIN32)
  set_target_properties(llama PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_subdirectory(models)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
