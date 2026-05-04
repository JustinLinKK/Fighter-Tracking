#pragma once

#include "inference_backend.h"

class TensorRtBackend : public InferenceBackend {
  public:
    explicit TensorRtBackend(const std::string &engine_path);
    ~TensorRtBackend() override;

    bool is_valid() const;
    const std::string &error() const;
    bool run(const float *input, InferenceOutputs outputs) override;

  private:
    struct Impl;
    Impl *impl_;
};
