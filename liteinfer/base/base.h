#ifndef LITEINFER_INCLUDE_BASE_BASE_H_
#define LITEINFER_INCLUDE_BASE_BASE_H_
#include <absl/status/status.h>
#include <glog/logging.h>
#include <cstdio>
#include <cstdint>
#include <ostream>
#include <string>
#define UNUSED(expr)  \
    do {              \
        (void)(expr); \
    } while (0)

namespace model {
enum class RuntimeTensorType {
    kInputTokens = 0,
    kInputEmbeddings = 1,
    kOutputRMSNorm = 2,
    kKeyCache = 3,
    kValueCache = 4,
    kQuery = 5,
    kInputPos = 6,
    kScoreStorage = 7,
    kOutputMHA = 8,
    kAttnOutput = 9,
    kW1Output = 10,
    kW2Output = 11,
    kW3Output = 12,
    kFFNRMSNorm = 13,
    kForwardOutput = 15,
    kForwardOutputCPU = 16,

    kSinCache = 17,
    kCosCache = 18,
};

inline const char* RuntimeTensorTypeName(RuntimeTensorType buffer_type) {
    switch (buffer_type) {
        case RuntimeTensorType::kInputTokens:
            return "kInputTokens";
        case RuntimeTensorType::kInputEmbeddings:
            return "kInputEmbeddings";
        case RuntimeTensorType::kOutputRMSNorm:
            return "kOutputRMSNorm";
        case RuntimeTensorType::kKeyCache:
            return "kKeyCache";
        case RuntimeTensorType::kValueCache:
            return "kValueCache";
        case RuntimeTensorType::kQuery:
            return "kQuery";
        case RuntimeTensorType::kInputPos:
            return "kInputPos";
        case RuntimeTensorType::kScoreStorage:
            return "kScoreStorage";
        case RuntimeTensorType::kOutputMHA:
            return "kOutputMHA";
        case RuntimeTensorType::kAttnOutput:
            return "kAttnOutput";
        case RuntimeTensorType::kW1Output:
            return "kW1Output";
        case RuntimeTensorType::kW2Output:
            return "kW2Output";
        case RuntimeTensorType::kW3Output:
            return "kW3Output";
        case RuntimeTensorType::kFFNRMSNorm:
            return "kFFNRMSNorm";
        case RuntimeTensorType::kForwardOutput:
            return "kForwardOutput";
        case RuntimeTensorType::kForwardOutputCPU:
            return "kForwardOutputCPU";
        case RuntimeTensorType::kSinCache:
            return "kSinCache";
        case RuntimeTensorType::kCosCache:
            return "kCosCache";
    }
    return "kUnknownRuntimeTensorType";
}

inline std::ostream& operator<<(std::ostream& os, RuntimeTensorType buffer_type) {
    os << RuntimeTensorTypeName(buffer_type);
    return os;
}
}

namespace base {
enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

enum class ModelType : uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLlama = 1,
    kModelTypeLlama2 = kModelTypeLlama,
    kModelTypeLlama3 = 2,
};

inline size_t DataTypeSize(DataType data_type) {
    if (data_type == DataType::kDataTypeFp32) {
        return sizeof(float);
    } else if (data_type == DataType::kDataTypeInt8) {
        return sizeof(int8_t);
    } else if (data_type == DataType::kDataTypeInt32) {
        return sizeof(int32_t);
    } else {
        return 0;
    }
}

class NoCopyable {
protected:
    NoCopyable() = default;
    ~NoCopyable() = default;
    NoCopyable(const NoCopyable&) = delete;
    NoCopyable& operator=(const NoCopyable&) = delete;
};

using Status = absl::Status;
using StatusCode = absl::StatusCode;

enum class TokenizerType {
    kEncodeUnknown = -1,
    kEncodeSpe = 0,
    kEncodeBpe = 1,
};

inline bool IsLlamaModel(ModelType model_type) {
    return model_type == ModelType::kModelTypeLlama || model_type == ModelType::kModelTypeLlama2 ||
           model_type == ModelType::kModelTypeLlama3;
}

inline bool UsesLlama3RoPE(ModelType model_type) {
    return model_type == ModelType::kModelTypeLlama3;
}

inline bool UsesHalfSplitRoPE(ModelType model_type) {
    return UsesLlama3RoPE(model_type);
}

inline float RoPETheta(ModelType model_type) {
    if (UsesLlama3RoPE(model_type)) {
        return 500000.0f;
    }
    return 10000.0f;
}

inline float RmsNormEpsilon(ModelType model_type) {
    UNUSED(model_type);
    return 1e-5f;
}

namespace error {
#define STATUS_CHECK(call)                                                                       \
    do {                                                                                         \
        const base::Status status = (call);                                                      \
        if (!status.ok()) {                                                                      \
            const size_t buf_size = 512;                                                         \
            char buf[buf_size];                                                                  \
            const std::string status_message(status.message());                                  \
            snprintf(buf, buf_size - 1,                                                          \
                     "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
                     __LINE__, static_cast<int>(status.code()), status_message.c_str());         \
            LOG(FATAL) << buf;                                                                   \
        }                                                                                        \
    } while (0)

Status Success(const std::string& err_msg = "");
Status FunctionNotImplement(const std::string& err_msg = "");
Status PathNotValid(const std::string& err_msg = "");
Status ModelParseError(const std::string& err_msg = "");
Status InternalError(const std::string& err_msg = "");
Status KeyHasExits(const std::string& err_msg = "");
Status InvalidArgument(const std::string& err_msg = "");
}  // namespace error
}  // namespace base
#endif  // LITEINFER_INCLUDE_BASE_BASE_H_
