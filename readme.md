# LiteInfer

LiteInfer is a lightweight C++ inference project for Llama-style models.

Current support:
- Llama 2 inference
- Llama 3 inference
- CPU execution
- CUDA execution when available

## Build

Run the build script from the project root:

```bash
./build.sh
```

Use `--cpu` or `--cuda` to choose the backend explicitly:

```bash
./build.sh --cpu
./build.sh --cuda
```

By default, it configures `build/` with `USE_CPM=ON`, `CMAKE_BUILD_TYPE=Release`, and auto-detects CUDA.

You can still override the defaults when needed:

```bash
USE_CPM=OFF CMAKE_BUILD_TYPE=Debug ./build.sh --cpu
```

`--cpu` forces a CPU-only build even on machines with CUDA. `--cuda` requires CUDA and fails during CMake configure if CUDA is unavailable. Without either flag, LiteInfer falls back to CPU-only when CUDA is not available.

## Run Llama 3

LiteInfer now generates automatically until it reaches EOS or fills the runtime KV cache.

```bash
./build/llama3_infer_demo <model_path> <tokenizer_path>
```

Example:

```bash
./build/llama3_infer_demo \
  local_models/llama3/Llama-3.2-1B.bin \
  local_models/llama3/Llama-3.2-1B/tokenizer.json
```

If you want a fixed local Llama 3.2 1B setup and only pass the prompt:

```bash
./build/llama3_infer_prompt_demo 你好，介绍一下你自己
```

## Tests

```bash
./build/test/test_main
ctest --test-dir build --output-on-failure -R test_main
```

## Tools

```bash
python3 tools/export_llama2.py <output_bin> --meta-llama <model_dir>
python3 tools/export_llama3.py <output_bin> --hf=<hf_model_dir>
```
