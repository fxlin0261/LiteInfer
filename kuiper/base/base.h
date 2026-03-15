#ifndef KUIPER_INCLUDE_BASE_BASE_H_
#define KUIPER_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>
#define UNUSED(expr)  \
    do {              \
        (void)(expr); \
    } while (0)

namespace model {
enum class ModelBufferType {
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
    kModelTypeQwen2 = 3,
    kModelTypeQwen3 = 4,
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

enum StatusCode : uint8_t {
    kSuccess = 0,
    kFunctionUnImplement = 1,
    kPathNotValid = 2,
    kModelParseError = 3,
    kInternalError = 5,
    kKeyValueHasExist = 6,
    kInvalidArgument = 7,
};

enum class TokenizerType {
    kEncodeUnknown = -1,
    kEncodeSpe = 0,
    kEncodeBpe = 1,
};

inline bool IsLlamaModel(ModelType model_type) {
    return model_type == ModelType::kModelTypeLlama ||
           model_type == ModelType::kModelTypeLlama2 ||
           model_type == ModelType::kModelTypeLlama3;
}

inline bool IsQwenModel(ModelType model_type) {
    return model_type == ModelType::kModelTypeQwen2 ||
           model_type == ModelType::kModelTypeQwen3;
}

inline bool UsesQwenRoPE(ModelType model_type) { return IsQwenModel(model_type); }

inline bool UsesLlama3RoPE(ModelType model_type) {
    return model_type == ModelType::kModelTypeLlama3;
}

inline bool UsesHalfSplitRoPE(ModelType model_type) {
    return UsesLlama3RoPE(model_type) || UsesQwenRoPE(model_type);
}

inline bool UsesQwen3Layout(ModelType model_type) {
    return model_type == ModelType::kModelTypeQwen3;
}

inline float RoPETheta(ModelType model_type) {
    if (UsesLlama3RoPE(model_type)) {
        return 500000.0f;
    }
    if (UsesQwenRoPE(model_type)) {
        return 1000000.0f;
    }
    return 10000.0f;
}

inline float RmsNormEpsilon(ModelType model_type) {
    return IsQwenModel(model_type) ? 1e-6f : 1e-5f;
}

class Status {
public:
    Status(int code = StatusCode::kSuccess, std::string err_message = "");

    Status(const Status& other) = default;

    Status& operator=(const Status& other) = default;

    Status& operator=(int code);

    bool operator==(int code) const;

    bool operator!=(int code) const;

    operator int() const;

    operator bool() const;

    int32_t get_err_code() const;

    const std::string& get_err_msg() const;

    void set_err_msg(const std::string& err_msg);

private:
    int code_ = StatusCode::kSuccess;
    std::string message_;
};

namespace error {
#define STATUS_CHECK(call)                                                                       \
    do {                                                                                         \
        const base::Status& status = call;                                                       \
        if (!status) {                                                                           \
            const size_t buf_size = 512;                                                         \
            char buf[buf_size];                                                                  \
            snprintf(buf, buf_size - 1,                                                          \
                     "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
                     __LINE__, int(status), status.get_err_msg().c_str());                       \
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

std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_BASE_H_
