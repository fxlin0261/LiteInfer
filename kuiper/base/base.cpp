#include "base/base.h"

namespace base {
namespace error {

Status Success(const std::string& err_msg) {
    UNUSED(err_msg);
    return absl::OkStatus();
}

Status FunctionNotImplement(const std::string& err_msg) {
    return absl::UnimplementedError(err_msg);
}

Status PathNotValid(const std::string& err_msg) { return absl::NotFoundError(err_msg); }

Status ModelParseError(const std::string& err_msg) { return absl::DataLossError(err_msg); }

Status InternalError(const std::string& err_msg) { return absl::InternalError(err_msg); }

Status InvalidArgument(const std::string& err_msg) { return absl::InvalidArgumentError(err_msg); }

Status KeyHasExits(const std::string& err_msg) { return absl::AlreadyExistsError(err_msg); }

}  // namespace error
}  // namespace base
