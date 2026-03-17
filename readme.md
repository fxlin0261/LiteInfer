# LiteInfer

LiteInfer is a lightweight C++ inference project for Llama-style models.

Current support:
- Llama 2 inference
- Llama 2 chat
- Llama 3 inference
- CPU execution
- CUDA execution when available

## Build

Use CPM to fetch dependencies automatically:

```bash
cmake -S . -B build -DUSE_CPM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Or build with system-installed dependencies:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

If CUDA is not available, LiteInfer builds in CPU-only mode.

## Run Llama 3

```bash
./build/models/llama3_infer <model_path> <tokenizer_path> [runtime_max_seq_len] [max_total_steps]
```

Example:

```bash
./build/models/llama3_infer \
  local_models/llama3/Llama-3.2-1B.bin \
  local_models/llama3/Llama-3.2-1B/tokenizer.json \
  2048 1
```

## Tests

```bash
cmake --build build --target test_llm --parallel
ctest --test-dir build --output-on-failure -R '^test_llm$'
```

## Tools

```bash
python3 tools/export_llama2.py <output_bin> --meta-llama <model_dir>
python3 tools/export_llama3.py <output_bin> --hf=<hf_model_dir>
```
