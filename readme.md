# LiteInfer

LiteInfer is a lightweight C++ inference project for Llama-style models.

Current support:
- Llama 3
- CPU
- CUDA when available

## Build

```bash
./build.sh
```

Optional:

```bash
./build.sh --cpu
./build.sh --cuda
```

## Export

Put your Hugging Face model directory under `local_models/`, then export LiteInfer `.bin` weights.

Example:

```bash
python3 tools/export_llama3.py \
  local_models/Llama-3.2-1B-Instruct.bin \
  --hf=local_models/Llama-3.2-1B-Instruct
```

## Run

Explicit model path:

```bash
./build/llama3_infer_demo \
  local_models/Llama-3.2-1B-Instruct.bin \
  local_models/Llama-3.2-1B-Instruct/tokenizer.json
```

Prompt demo:

```bash
./build/llama3_infer_prompt_demo \
  local_models/Llama-3.2-1B-Instruct.bin \
  local_models/Llama-3.2-1B-Instruct/tokenizer.json \
  你好，介绍一下你自己
```

Prompt demo passes your prompt directly into the selected model.

For faster local CPU debugging, you can lower the runtime window:

```bash
./build/llama3_infer_prompt_demo \
  --max-seq-len 128 \
  local_models/Llama-3.2-1B-Instruct.bin \
  local_models/Llama-3.2-1B-Instruct/tokenizer.json \
  你好，介绍一下你自己
```

## Tests

```bash
./build/test/test_main
ctest --test-dir build --output-on-failure -R test_main
```
