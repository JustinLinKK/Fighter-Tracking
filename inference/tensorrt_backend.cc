#include "tensorrt_backend.h"

#include <cstring>
#include <memory>
#include <string>

#include "../utils/types.h"

#ifdef USE_TENSORRT
#include "../utils/init_engine.h"
#include "../utils/utils.h"

struct TensorRtBackend::Impl {
    explicit Impl(const std::string &engine_path) {
        engine_opt = init_engine(engine_path);
        if (!engine_opt) {
            error = "failed to initialize TensorRT engine from " + engine_path;
            return;
        }

        CHECK_CUDA(cudaMallocManaged(&input, ROI_SIZE * ROI_SIZE * sizeof(float)));
        CHECK_CUDA(
            cudaMallocManaged(&response, ROI_SIZE * ROI_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMallocManaged(&x_max, ROI_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMallocManaged(&y_max, ROI_SIZE * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(&stream));

        INIT_engine &engine_ref = *engine_opt;
        engine_ref.context->setTensorAddress(
            engine_ref.engine->getIOTensorName(0), input);
        engine_ref.context->setTensorAddress(
            engine_ref.engine->getIOTensorName(1), response);
        engine_ref.context->setTensorAddress(
            engine_ref.engine->getIOTensorName(2), x_max);
        engine_ref.context->setTensorAddress(
            engine_ref.engine->getIOTensorName(3), y_max);

        valid = true;
    }

    ~Impl() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        if (input != nullptr) {
            cudaFree(input);
        }
        if (response != nullptr) {
            cudaFree(response);
        }
        if (x_max != nullptr) {
            cudaFree(x_max);
        }
        if (y_max != nullptr) {
            cudaFree(y_max);
        }
    }

    bool run(const float *host_input, InferenceOutputs outputs) {
        if (!valid) {
            return false;
        }

        std::memcpy(input, host_input, ROI_SIZE * ROI_SIZE * sizeof(float));

        INIT_engine &engine_ref = *engine_opt;
        if (!engine_ref.context->enqueueV3(stream)) {
            error = "TensorRT enqueueV3 failed";
            return false;
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        std::memcpy(outputs.response, response,
                    ROI_SIZE * ROI_SIZE * sizeof(float));
        std::memcpy(outputs.x_max, x_max, ROI_SIZE * sizeof(float));
        std::memcpy(outputs.y_max, y_max, ROI_SIZE * sizeof(float));
        return true;
    }

    std::optional<INIT_engine> engine_opt;
    float *input = nullptr;
    float *response = nullptr;
    float *x_max = nullptr;
    float *y_max = nullptr;
    cudaStream_t stream = nullptr;
    bool valid = false;
    std::string error;
};
#else
struct TensorRtBackend::Impl {
    explicit Impl(const std::string &) {}
    bool run(const float *, InferenceOutputs) { return false; }
    bool valid = false;
    std::string error = "TensorRT backend is unavailable in this build";
};
#endif

TensorRtBackend::TensorRtBackend(const std::string &engine_path)
    : impl_(new Impl(engine_path)) {}

TensorRtBackend::~TensorRtBackend() { delete impl_; }

bool TensorRtBackend::is_valid() const { return impl_->valid; }

const std::string &TensorRtBackend::error() const { return impl_->error; }

bool TensorRtBackend::run(const float *input, InferenceOutputs outputs) {
    return impl_->run(input, outputs);
}
