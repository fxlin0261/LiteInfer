# LiteInfer

LiteInfer is a lightweight C++ inference project for Llama-style models.

Current support:
- Llama 2 inference
- Llama 3 inference
- CPU execution
- CUDA execution when available

## Quick Start

### 1. Build

Run the build script from the project root:

```bash
./build.sh
```

Useful variants:

```bash
./build.sh --cpu
./build.sh --cuda
USE_CPM=OFF CMAKE_BUILD_TYPE=Debug ./build.sh --cpu
```

Notes:
- `--cpu` forces a CPU-only build.
- `--cuda` requires CUDA and fails if CUDA is unavailable.
- Without either flag, LiteInfer auto-detects CUDA and falls back to CPU-only when needed.
- By default the build uses `build/`, `USE_CPM=ON`, and `CMAKE_BUILD_TYPE=Release`.

### 2. Prepare a Local Model Directory

Place the downloaded Hugging Face model files under `local_models/llama3/`.

Recommended layout:

```text
local_models/
  llama3/
    Llama-3.2-1B/
      config.json
      generation_config.json
      model.safetensors
      special_tokens_map.json
      tokenizer.json
      tokenizer_config.json
    Llama-3.2-1B-Instruct/
      config.json
      generation_config.json
      model.safetensors
      special_tokens_map.json
      tokenizer.json
      tokenizer_config.json
```

### 3. Export LiteInfer `.bin` Weights

LiteInfer runs on its own `.bin` format, so Hugging Face weights must be exported first.

Base model:

```bash
python3 tools/export_llama3.py \
  local_models/llama3/Llama-3.2-1B.bin \
  --hf=local_models/llama3/Llama-3.2-1B
```

Instruct model:

```bash
python3 tools/export_llama3.py \
  local_models/llama3/Llama-3.2-1B-Instruct.bin \
  --hf=local_models/llama3/Llama-3.2-1B-Instruct
```

Llama 2 export is also supported:

```bash
python3 tools/export_llama2.py <output_bin> --meta-llama <model_dir>
```

### 4. Run

#### Option A: Smoke Test With Explicit Paths

`llama3_infer_demo` loads a model from explicit paths and runs a built-in prompt.

```bash
./build/llama3_infer_demo <model_path> <tokenizer_path>
```

Example:

```bash
./build/llama3_infer_demo \
  local_models/llama3/Llama-3.2-1B.bin \
  local_models/llama3/Llama-3.2-1B/tokenizer.json
```

#### Option B: Prompt Demo With Local Default Paths

`llama3_infer_prompt_demo` is the easier interactive entry point.

It:
- accepts your prompt from the command line
- prefers `Llama-3.2-1B-Instruct` when it is available
- automatically wraps Instruct prompts with the Llama 3 chat template
- prints only newly generated text instead of echoing the whole prompt back

Default behavior:

```bash
./build/llama3_infer_prompt_demo 你好，介绍一下你自己
```

Force the instruct model:

```bash
./build/llama3_infer_prompt_demo --instruct 你好，介绍一下你自己
```

Force the base model:

```bash
./build/llama3_infer_prompt_demo --base explain rotary embeddings
```

## Which Binary Should I Use?

- Use `./build/llama3_infer_prompt_demo` if you want to type your own prompt.
- Use `./build/llama3_infer_demo` if you want a simple load-and-run smoke test with explicit model paths.
- Use `./build/llama2_infer_demo` for Llama 2.

## Tests

```bash
./build/test/test_main
ctest --test-dir build --output-on-failure -R test_main
```

## Troubleshooting

- If `llama3_infer_prompt_demo` says assets are missing, make sure both the `.bin` file and the matching `tokenizer.json` exist under `local_models/llama3/`.
- If export succeeds but generation looks wrong, re-export the `.bin` from the Hugging Face directory with `tools/export_llama3.py`.
- If CMake cannot find dependencies, try building with `USE_CPM=ON` or clean `build/` and rerun `./build.sh`.
