# model/core

Core model abstractions and shared metadata.

- `model.h/.cpp`: top-level model interface and common lifecycle
- `config.h`: model config structures
- `raw_model_data.*`: raw weight access helpers

Keep concrete model-family logic out of this directory.
