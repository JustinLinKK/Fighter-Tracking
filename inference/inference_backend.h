#pragma once

#include <memory>
#include <string>

struct InferenceOutputs {
    float *response;
    float *x_max;
    float *y_max;
};

class InferenceBackend {
  public:
    virtual ~InferenceBackend() = default;
    virtual bool run(const float *input, InferenceOutputs outputs) = 0;
};

std::unique_ptr<InferenceBackend>
create_backend(const std::string &backend_name, const std::string &engine_path,
               std::string &error);
