if(MSVC)
  # Visual Studio can attach a duplicated custom build rule to every .cu file.
  set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE
      OFF
      CACHE BOOL "CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE" FORCE
  )
endif()

if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "CUDA support is required but no CUDA compiler was configured.")
endif()

find_package(CUDAToolkit REQUIRED)
set(KUIPER_CUDART_TARGET CUDA::cudart)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(KUIPER_INSTALLED_GPU_CCS)
  string(STRIP "${KUIPER_INSTALLED_GPU_CCS}" KUIPER_INSTALLED_GPU_CCS)

  if(KUIPER_INSTALLED_GPU_CCS)
    string(REPLACE " " ";" KUIPER_INSTALLED_GPU_CCS "${KUIPER_INSTALLED_GPU_CCS}")
    string(REPLACE "." "" KUIPER_CUDA_ARCH_LIST "${KUIPER_INSTALLED_GPU_CCS}")
    set(CMAKE_CUDA_ARCHITECTURES
        "${KUIPER_CUDA_ARCH_LIST}"
        CACHE STRING "CUDA architectures" FORCE
    )
  endif()
endif()

message(STATUS "Found CUDA Toolkit v${CMAKE_CUDA_COMPILER_VERSION}")
if(CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(
    STATUS
      "CMAKE_CUDA_ARCHITECTURES was not auto-detected; use the CMake cache to set it explicitly."
  )
endif()
