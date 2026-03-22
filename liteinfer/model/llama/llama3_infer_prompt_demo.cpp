#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <base/base.h>
#include <glog/logging.h>

#include "model/generation.h"
#include "model/llama/llama.h"

namespace {
constexpr int32_t kDefaultRuntimeMaxSeqLen = 4096;
constexpr const char* kBaseCheckpointRelativePath = "local_models/llama3/Llama-3.2-1B.bin";
constexpr const char* kBaseTokenizerRelativePath =
    "local_models/llama3/Llama-3.2-1B/tokenizer.json";
constexpr const char* kInstructCheckpointRelativePath =
    "local_models/llama3/Llama-3.2-1B-Instruct.bin";
constexpr const char* kInstructTokenizerRelativePath =
    "local_models/llama3/Llama-3.2-1B-Instruct/tokenizer.json";

enum class PromptModelPreset {
    kAuto,
    kBase,
    kInstruct,
};

struct PromptOptions {
    PromptModelPreset preset = PromptModelPreset::kAuto;
    std::string prompt;
};

struct PromptModelAssets {
    PromptModelPreset preset = PromptModelPreset::kAuto;
    std::filesystem::path checkpoint_path;
    std::filesystem::path tokenizer_path;
};

std::filesystem::path ResolveProjectRoot(const char* executable_path) {
    return std::filesystem::absolute(executable_path).parent_path().parent_path();
}

const char* PresetName(PromptModelPreset preset) {
    switch (preset) {
        case PromptModelPreset::kBase:
            return "base";
        case PromptModelPreset::kInstruct:
            return "instruct";
        case PromptModelPreset::kAuto:
            return "auto";
    }
    return "unknown";
}

PromptModelAssets BuildAssets(const std::filesystem::path& project_root, PromptModelPreset preset) {
    PromptModelAssets assets;
    assets.preset = preset;
    if (preset == PromptModelPreset::kInstruct) {
        assets.checkpoint_path = project_root / kInstructCheckpointRelativePath;
        assets.tokenizer_path = project_root / kInstructTokenizerRelativePath;
    } else {
        assets.checkpoint_path = project_root / kBaseCheckpointRelativePath;
        assets.tokenizer_path = project_root / kBaseTokenizerRelativePath;
    }
    return assets;
}

bool AssetsExist(const PromptModelAssets& assets) {
    return std::filesystem::exists(assets.checkpoint_path) &&
           std::filesystem::exists(assets.tokenizer_path);
}

base::Status ParsePromptOptions(int argc, char* argv[], PromptOptions* options) {
    if (options == nullptr) {
        return base::error::InvalidArgument("The prompt options pointer is empty.");
    }

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--base") {
            if (options->preset == PromptModelPreset::kInstruct) {
                return base::error::InvalidArgument(
                    "Only one of --base or --instruct can be provided.");
            }
            options->preset = PromptModelPreset::kBase;
            continue;
        }
        if (arg == "--instruct") {
            if (options->preset == PromptModelPreset::kBase) {
                return base::error::InvalidArgument(
                    "Only one of --base or --instruct can be provided.");
            }
            options->preset = PromptModelPreset::kInstruct;
            continue;
        }

        if (!options->prompt.empty()) {
            options->prompt += ' ';
        }
        options->prompt += arg;
    }

    if (options->prompt.empty()) {
        return base::error::InvalidArgument("The prompt is empty.");
    }
    return base::error::Success();
}

base::Status ResolvePromptModelAssets(const std::filesystem::path& project_root,
                                      PromptModelPreset requested_preset,
                                      PromptModelAssets* assets) {
    if (assets == nullptr) {
        return base::error::InvalidArgument("The prompt model assets pointer is empty.");
    }

    if (requested_preset != PromptModelPreset::kAuto) {
        *assets = BuildAssets(project_root, requested_preset);
        if (!AssetsExist(*assets)) {
            return base::error::PathNotValid("The " + std::string(PresetName(requested_preset)) +
                                             " llama3 prompt demo assets are missing.");
        }
        return base::error::Success();
    }

    const PromptModelAssets instruct_assets =
        BuildAssets(project_root, PromptModelPreset::kInstruct);
    if (AssetsExist(instruct_assets)) {
        *assets = instruct_assets;
        return base::error::Success();
    }

    const PromptModelAssets base_assets = BuildAssets(project_root, PromptModelPreset::kBase);
    if (AssetsExist(base_assets)) {
        *assets = base_assets;
        return base::error::Success();
    }

    return base::error::PathNotValid(
        "Neither base nor instruct llama3 prompt demo assets were found under local_models/llama3.");
}

std::string FormatPromptForPreset(const std::string& prompt, PromptModelPreset preset) {
    if (preset != PromptModelPreset::kInstruct) {
        return prompt;
    }

    return "<|start_header_id|>user<|end_header_id|>\n\n" + prompt +
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
}

std::vector<int32_t> GeneratedTokensOnly(const std::vector<int32_t>& all_output_tokens,
                                         int32_t prompt_token_count) {
    if (prompt_token_count >= static_cast<int32_t>(all_output_tokens.size())) {
        return {};
    }
    return std::vector<int32_t>(all_output_tokens.begin() + prompt_token_count,
                                all_output_tokens.end());
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        LOG(INFO) << "Usage: ./llama3_infer_prompt_demo [--base|--instruct] <prompt>";
        return EXIT_FAILURE;
    }

    PromptOptions options;
    const auto parse_status = ParsePromptOptions(argc, argv, &options);
    if (!parse_status.ok()) {
        LOG(ERROR) << parse_status.message();
        LOG(INFO) << "Usage: ./llama3_infer_prompt_demo [--base|--instruct] <prompt>";
        return EXIT_FAILURE;
    }

    const std::filesystem::path project_root = ResolveProjectRoot(argv[0]);
    PromptModelAssets assets;
    const auto asset_status = ResolvePromptModelAssets(project_root, options.preset, &assets);
    if (!asset_status.ok()) {
        LOG(FATAL) << asset_status.message();
    }

    const std::string formatted_prompt = FormatPromptForPreset(options.prompt, assets.preset);

    LOG(INFO) << "Prompt preset: " << PresetName(assets.preset);
    LOG(INFO) << "Prompt: " << options.prompt;
    LOG(INFO) << "Using checkpoint: " << assets.checkpoint_path.string();
    LOG(INFO) << "Using tokenizer: " << assets.tokenizer_path.string();
    if (assets.preset == PromptModelPreset::kInstruct) {
        LOG(INFO) << "Wrapped the prompt with the Llama 3 instruct chat template.";
    }

    model::Llama3Model model(assets.tokenizer_path.string(), assets.checkpoint_path.string(),
                             false);
    const auto init_status = model.init(base::DefaultDeviceType(), kDefaultRuntimeMaxSeqLen);
    if (!init_status.ok()) {
        LOG(FATAL) << "The model init failed, code: " << static_cast<int>(init_status.code())
                   << ", message: " << init_status.message();
    }

    const int32_t runtime_max_seq_len = model.max_seq_len();
    const int32_t max_context_steps = runtime_max_seq_len;
    LOG(INFO) << "Using runtime max seq len " << runtime_max_seq_len
              << ". Generation stops automatically at EOS or when the KV cache is full.";

    const auto start = std::chrono::steady_clock::now();
    std::cout << "Generating...\n" << std::flush;
    const auto tokens = model.encode(formatted_prompt);
    app::GenerationState generation_result;
    const auto generate_status =
        app::RunGeneration(model, tokens, max_context_steps, &generation_result);
    if (!generate_status.ok()) {
        LOG(FATAL) << "Text generation failed, code: "
                   << static_cast<int>(generate_status.code())
                   << ", message: " << generate_status.message();
    }

    const int32_t prompt_token_count =
        std::max<int32_t>(0, static_cast<int32_t>(tokens.size()) - 1);
    const auto generated_tokens =
        GeneratedTokensOnly(generation_result.words, prompt_token_count);
    std::cout << model.decode(generated_tokens) << ' ' << std::flush;

    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<double>(end - start).count();
    std::cout << "\nsteps/s:" << (static_cast<double>(generation_result.executed_steps) / duration)
              << '\n';
    return EXIT_SUCCESS;
}
