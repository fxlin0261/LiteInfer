# LiteInfer

LiteInfer is a lightweight C++ inference project for running Llama-style models.

Right now this repo mainly includes:
- Llama 2 inference
- Llama 2 chat demo
- Llama 3 inference
- CPU build by default
- CUDA build when a CUDA compiler is available

## Project Layout

```text
liteinfer/
├── base/         # allocators, buffers, common utilities
├── model/        # model core and llama implementations
├── op/           # operators and kernels
├── sampler/      # token sampling
├── tensor/       # tensor implementation
└── tokenizer/    # tokenizer helpers
```

## Build

### Option 1: build with CPM-managed dependencies

This is the easiest way if you want CMake to fetch dependencies automatically.

```bash
cmake -S . -B build -DUSE_CPM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Option 2: build with system dependencies

If your machine already has the required libraries installed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Notes

- If CUDA is not available, LiteInfer will build in CPU-only mode.
- On this repo, ARM64 + no CUDA works fine for CPU inference.

## Main Binaries

After building, the main executables are:

- `build/models/llama2_infer`
- `build/models/llama2_chat`
- `build/models/llama3_infer`

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

Arguments:
- `model_path`: exported model binary
- `tokenizer_path`: tokenizer file
- `runtime_max_seq_len`: optional runtime sequence length cap
- `max_total_steps`: optional max generation steps for smoke testing

## Run Llama 2

```bash
./build/models/llama2_infer <model_path> <tokenizer_path> [runtime_max_seq_len] [max_total_steps]
```

## Tests

Build the test target:

```bash
cmake --build build --target test_llm --parallel
```

Run tests:

```bash
ctest --test-dir build --output-on-failure -R '^test_llm$'
```

## Model Export Tools

Useful scripts live in `tools/`:

- `tools/export_llama2.py`
- `tools/export_llama3.py`
- `tools/hf_infer_llama3.py`

Examples:

```bash
python3 tools/export_llama2.py <output_bin> --meta-llama <model_dir>
python3 tools/export_llama3.py <output_bin> --hf=<hf_model_dir>
```

## Local Development Notes

Common ignored local paths:
- `build/`
- `build-*/`
- `local_models/`
- `log/`

So local build outputs and model files should not be committed.
