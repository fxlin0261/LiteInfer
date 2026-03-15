# model/decoder

Shared decoder-only model skeletons.

- `standard_decoder.*`: common decoder forward structure
- `model_utils.*`: decoder support helpers

Put family-specific behavior in `model/llama` or `model/qwen`, not here.
