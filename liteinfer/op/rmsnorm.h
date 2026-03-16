#ifndef LITEINFER_INCLUDE_OP_RMSNORM_H_
#define LITEINFER_INCLUDE_OP_RMSNORM_H_
#include "layer.h"

namespace op {
class RmsNormLayer : public LayerParam {
public:
    explicit RmsNormLayer(base::DeviceType device_type, int32_t dim, float eps);
    base::Status check() const override;
    base::Status forward() override;

private:
    int32_t dim_ = 0;
    float eps_ = 1e-5f;
};
}  // namespace op
#endif  // LITEINFER_INCLUDE_OP_RMSNORM_H_
