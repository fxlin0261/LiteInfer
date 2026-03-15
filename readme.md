# KuiperLLama

一个用 C++ 实现的大模型推理项目，当前支持：

- Llama2
- Llama3
- Qwen2
- Qwen3
- CPU 推理
- CUDA 加速
- 部分量化相关能力

## 项目结构

```text
kuiper/
├── base/        # 基础设施：内存、buffer、unicode、device/cuda 配置
├── tokenizer/   # tokenizer 相关实现
├── tensor/      # 张量封装
├── op/          # 基础算子
├── sampler/     # 采样逻辑
└── model/
    ├── core/    # 模型基类、配置、权重读取
    ├── decoder/ # 通用 decoder 骨架
    ├── llama/   # Llama 系列实现
    └── qwen/    # Qwen 系列实现
```

## 依赖

- CMake
- C++ 编译器
- glog
- gtest
- sentencepiece
- armadillo
- CUDA Toolkit（如果需要 CUDA）

如果本机没有装全依赖，可以使用 `USE_CPM=ON` 让 CMake 自动拉取部分依赖。

## 编译

### 方式一：自动拉依赖

```bash
cmake -S . -B build -DUSE_CPM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 方式二：使用本机已安装依赖

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## 测试

运行当前单元测试：

```bash
ctest --test-dir build --output-on-failure -R '^test_llm$'
```

如果要查看 `test_llm` 内部包含哪些 gtest 用例：

```bash
./build/test/test_llm --gtest_list_tests
```

## 运行推理

### Llama2

```bash
./build/models/llama2_infer <model_path> <tokenizer_path>
```

### Llama3

```bash
./build/models/llama3_infer <model_path> <tokenizer_path>
```

### Qwen2

```bash
./build/models/qwen2_infer <model_path> <tokenizer_path>
```

### Qwen3

```bash
./build/models/qwen3_infer <model_path> <tokenizer_path>
```

如果不确定参数格式，可以直接运行对应可执行文件查看提示。

## 模型导出脚本

项目内提供了一些模型导出脚本，位于：

- `models/llama/`
- `models/qwen/`

例如：

```bash
python3 models/llama/export_llama2.py <output_bin> --meta-llama <model_dir>
python3 models/llama/export_llama3.py <output_bin> --hf=<hf_model_dir>
```

Qwen 相关导出脚本请看：

- `models/qwen/export_qwen2.py`
- `models/qwen/export_qwen3_pth.py`
- `models/qwen/export_qwen3_bin.py`

## 备注

- `build/` 是编译输出目录
- `models/` 下面是导出脚本和推理入口
- `test/` 下面是单元测试
- 如果只是阅读代码，建议先看：
  - `kuiper/model/core/model.h`
  - `kuiper/model/decoder/standard_decoder.h`
  - `kuiper/model/llama/llama.h`
  - `kuiper/model/qwen/qwen2.h`
