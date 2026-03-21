#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
USE_CPM="${USE_CPM:-ON}"
LITEINFER_CUDA_MODE="${LITEINFER_CUDA_MODE:-AUTO}"
CLI_CUDA_MODE=""

usage() {
  echo "Usage: ./build.sh [--cpu | --cuda]"
  echo
  echo "  --cpu   Force a CPU-only build."
  echo "  --cuda  Require a CUDA build and fail if CUDA is unavailable."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      if [[ "${CLI_CUDA_MODE}" == "CUDA" ]]; then
        echo "Choose either --cpu or --cuda, not both." >&2
        exit 1
      fi
      CLI_CUDA_MODE="CPU"
      LITEINFER_CUDA_MODE="CPU"
      ;;
    --cuda)
      if [[ "${CLI_CUDA_MODE}" == "CPU" ]]; then
        echo "Choose either --cpu or --cuda, not both." >&2
        exit 1
      fi
      CLI_CUDA_MODE="CUDA"
      LITEINFER_CUDA_MODE="CUDA"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

cmake -S . -B "${BUILD_DIR}" -DUSE_CPM="${USE_CPM}" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" -DLITEINFER_CUDA_MODE="${LITEINFER_CUDA_MODE}"
cmake --build "${BUILD_DIR}" --parallel
