cmake_minimum_required(VERSION 3.16)

project(LiteInfer LANGUAGES CXX)

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0169 OLD)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(LITEINFER_WITH_CUDA ON)
else()
  set(LITEINFER_WITH_CUDA OFF)
  message(STATUS "CUDA compiler not found; building CPU-only.")
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

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
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

if(NOT TARGET sentencepiece::sentencepiece
   AND NOT TARGET sentencepiece
   AND NOT TARGET sentencepiece-static
)
  find_package(sentencepiece QUIET)
endif()

if(NOT TARGET sentencepiece::sentencepiece
   AND NOT TARGET sentencepiece
   AND NOT TARGET sentencepiece-static
)
  set(LITEINFER_SENTENCEPIECE_PREFIX_HINTS
      /opt/homebrew
      /opt/homebrew/opt/sentencepiece
      /usr/local
      /usr/local/opt/sentencepiece
  )
  find_path(
    SENTENCEPIECE_INCLUDE_DIR
    sentencepiece_processor.h
    HINTS ${LITEINFER_SENTENCEPIECE_PREFIX_HINTS}
    PATH_SUFFIXES include
  )
  find_library(
    SENTENCEPIECE_LIBRARY
    NAMES sentencepiece
    HINTS ${LITEINFER_SENTENCEPIECE_PREFIX_HINTS}
    PATH_SUFFIXES lib
  )

  if(SENTENCEPIECE_INCLUDE_DIR AND SENTENCEPIECE_LIBRARY)
    add_library(sentencepiece UNKNOWN IMPORTED)
    set_target_properties(
      sentencepiece
      PROPERTIES IMPORTED_LOCATION "${SENTENCEPIECE_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${SENTENCEPIECE_INCLUDE_DIR}"
    )
  else()
    message(FATAL_ERROR "Unable to find sentencepiece headers and library.")
  endif()
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
  sentencepiece_train
  sentencepiece_train-static
)

add_library(liteinfer_sentencepiece INTERFACE)
if(TARGET sentencepiece::sentencepiece)
  target_link_libraries(liteinfer_sentencepiece INTERFACE sentencepiece::sentencepiece)
elseif(TARGET sentencepiece)
  target_link_libraries(liteinfer_sentencepiece INTERFACE sentencepiece)
elseif(TARGET sentencepiece-static)
  target_link_libraries(liteinfer_sentencepiece INTERFACE sentencepiece-static)
else()
  message(FATAL_ERROR "Unable to find a usable sentencepiece target.")
endif()

if(DEFINED sentencepiece_SOURCE_DIR AND EXISTS "${sentencepiece_SOURCE_DIR}/src")
  target_include_directories(liteinfer_sentencepiece INTERFACE "${sentencepiece_SOURCE_DIR}/src")
endif()

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
  "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama2_infer.cpp"
  "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama3_infer.cpp"
  "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama2_chat.cpp"
  "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama3_chat.cpp"
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
target_link_libraries(liteinfer_runtime PUBLIC glog::glog absl::status liteinfer_armadillo liteinfer_sentencepiece)

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
  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models")
  target_link_libraries(${target_name} PRIVATE liteinfer::runtime)
endfunction()

liteinfer_add_model_target(llama2_infer "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama2_infer.cpp")
liteinfer_add_model_target(llama3_infer "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama3_infer.cpp")
liteinfer_add_model_target(llama2_chat "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama2_chat.cpp")
liteinfer_add_model_target(llama3_chat "${PROJECT_SOURCE_DIR}/liteinfer/model/llama/llama3_chat.cpp")
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
