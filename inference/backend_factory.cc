#include "inference_backend.h"

#include <memory>
#include <string>

#include "cpp_frangi_backend.h"
#include "tensorrt_backend.h"

std::unique_ptr<InferenceBackend>
create_backend(const std::string &backend_name, const std::string &engine_path,
               std::string &error) {
    if (backend_name == "cpp") {
        return std::make_unique<CppFrangiBackend>();
    }

    if (backend_name == "tensorrt") {
        auto backend = std::make_unique<TensorRtBackend>(engine_path);
        if (!backend->is_valid()) {
            error = backend->error();
            return nullptr;
        }
        return backend;
    }

    error = "unknown backend '" + backend_name +
            "' (expected 'cpp' or 'tensorrt')";
    return nullptr;
}
