#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
USE_CPM="${USE_CPM:-ON}"

cmake -S . -B "${BUILD_DIR}" -DUSE_CPM="${USE_CPM}" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
cmake --build "${BUILD_DIR}" --parallel