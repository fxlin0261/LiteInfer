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

By default, it configures `build/` with `USE_CPM=ON` and `CMAKE_BUILD_TYPE=Release`.

You can still override the defaults when needed:

```bash
USE_CPM=OFF CMAKE_BUILD_TYPE=Debug ./build.sh
```

If CUDA is not available, LiteInfer builds in CPU-only mode.

## Run Llama 3

LiteInfer now generates automatically until it reaches EOS or fills the runtime KV cache.

```bash
./build/models/llama3_infer <model_path> <tokenizer_path>
```

Example:

```bash
./build/models/llama3_infer \
  local_models/llama3/Llama-3.2-1B.bin \
  local_models/llama3/Llama-3.2-1B/tokenizer.json
```

## Tests

```bash
./build/test/test_main
ctest --test-dir build --output-on-failure -R '^test_main$'
```

## Tools

```bash
python3 tools/export_llama2.py <output_bin> --meta-llama <model_dir>
python3 tools/export_llama3.py <output_bin> --hf=<hf_model_dir>
```
