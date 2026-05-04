#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../inference/inference_backend.h"
#include "../utils/get_roi.h"
#include "../utils/types.h"

namespace {

constexpr std::string_view ENGINE_FILE =
    "./engine_model/Norm_Grad_Response_Masked_Max_480.engine";
constexpr std::string_view VIDEO_FILE =
    "./Datasets/Anti-UAV-RGBT/test/test1/infrared.mp4";
constexpr float INIT_BOX_W = 52.0f;
constexpr float INIT_BOX_H = 39.0f;

struct Fixture {
    std::string name;
    std::vector<float> input;
};

struct DiffStats {
    float response_max_abs = 0.0f;
    float x_max_abs = 0.0f;
    float y_max_abs = 0.0f;
};

DiffStats compare_outputs(const std::vector<float> &lhs_response,
                          const std::vector<float> &lhs_x,
                          const std::vector<float> &lhs_y,
                          const std::vector<float> &rhs_response,
                          const std::vector<float> &rhs_x,
                          const std::vector<float> &rhs_y) {
    DiffStats stats;
    for (int i = 0; i < ROI_SIZE * ROI_SIZE; ++i) {
        stats.response_max_abs =
            std::max(stats.response_max_abs,
                     std::abs(lhs_response[i] - rhs_response[i]));
    }
    for (int i = 0; i < ROI_SIZE; ++i) {
        stats.x_max_abs =
            std::max(stats.x_max_abs, std::abs(lhs_x[i] - rhs_x[i]));
        stats.y_max_abs =
            std::max(stats.y_max_abs, std::abs(lhs_y[i] - rhs_y[i]));
    }
    return stats;
}

bool all_finite(const std::vector<float> &values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

void add_real_frame_fixtures(std::vector<Fixture> &fixtures) {
    cv::VideoCapture cap{std::string(VIDEO_FILE)};
    if (!cap.isOpened()) {
        return;
    }

    cv::Mat bgr;
    if (!cap.read(bgr) || bgr.empty()) {
        return;
    }

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat resized_img;
    cv::resize(gray, resized_img, cv::Size(ROI_SIZE, ROI_SIZE), 0, 0,
               cv::INTER_AREA);
    cv::Mat resized_float;
    resized_img.convertTo(resized_float, CV_32F);

    Fixture resized_fixture{"real_resized_frame",
                            std::vector<float>(ROI_SIZE * ROI_SIZE)};
    std::memcpy(resized_fixture.input.data(), resized_float.ptr<float>(),
                ROI_SIZE * ROI_SIZE * sizeof(float));
    fixtures.push_back(std::move(resized_fixture));

    std::vector<uint8_t> roi_u8(ROI_SIZE * ROI_SIZE);
    Box init_box = {0.0f, 0.0f, INIT_BOX_W, INIT_BOX_H};
    GetROI(roi_u8.data(), gray.ptr<uint8_t>(0), init_box);

    Fixture roi_fixture{"real_roi_crop", std::vector<float>(ROI_SIZE * ROI_SIZE)};
    for (int i = 0; i < ROI_SIZE * ROI_SIZE; ++i) {
        roi_fixture.input[i] = static_cast<float>(roi_u8[i]);
    }
    fixtures.push_back(std::move(roi_fixture));
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);

    std::string error;
    auto cpp_backend = create_backend("cpp", std::string(ENGINE_FILE), error);
    if (!cpp_backend) {
        std::cerr << "Failed to create cpp backend: " << error << std::endl;
        return 1;
    }

    auto trt_backend =
        create_backend("tensorrt", std::string(ENGINE_FILE), error);
    if (!trt_backend) {
        std::cerr << "Failed to create TensorRT backend: " << error
                  << std::endl;
        return 1;
    }

    std::vector<Fixture> fixtures;

    Fixture random_fixture{"random", std::vector<float>(ROI_SIZE * ROI_SIZE)};
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> pixel_dist(0.0f, 255.0f);
    for (float &value : random_fixture.input) {
        value = pixel_dist(rng);
    }
    fixtures.push_back(std::move(random_fixture));

    Fixture blank_fixture{"blank", std::vector<float>(ROI_SIZE * ROI_SIZE, 0.0f)};
    fixtures.push_back(std::move(blank_fixture));

    Fixture low_contrast_fixture{"low_contrast",
                                 std::vector<float>(ROI_SIZE * ROI_SIZE)};
    std::uniform_real_distribution<float> low_contrast_dist(120.0f, 136.0f);
    for (float &value : low_contrast_fixture.input) {
        value = low_contrast_dist(rng);
    }
    fixtures.push_back(std::move(low_contrast_fixture));

    add_real_frame_fixtures(fixtures);

    constexpr float kResponseTolerance = 0.35f;
    constexpr float kProjectionTolerance = 0.35f;

    bool all_ok = true;
    for (const Fixture &fixture : fixtures) {
        std::vector<float> cpp_response(ROI_SIZE * ROI_SIZE);
        std::vector<float> cpp_x(ROI_SIZE);
        std::vector<float> cpp_y(ROI_SIZE);
        std::vector<float> trt_response(ROI_SIZE * ROI_SIZE);
        std::vector<float> trt_x(ROI_SIZE);
        std::vector<float> trt_y(ROI_SIZE);

        if (!cpp_backend->run(
                fixture.input.data(),
                InferenceOutputs{cpp_response.data(), cpp_x.data(),
                                 cpp_y.data()})) {
            std::cerr << "cpp backend failed on fixture " << fixture.name
                      << std::endl;
            return 1;
        }

        if (!trt_backend->run(
                fixture.input.data(),
                InferenceOutputs{trt_response.data(), trt_x.data(),
                                 trt_y.data()})) {
            std::cerr << "TensorRT backend failed on fixture " << fixture.name
                      << std::endl;
            return 1;
        }

        if (!cpp_backend->run(
                fixture.input.data(),
                InferenceOutputs{cpp_response.data(), cpp_x.data(),
                                 cpp_y.data()})) {
            std::cerr << "cpp backend failed on repeated run for fixture "
                      << fixture.name << std::endl;
            return 1;
        }

        const bool finite =
            all_finite(cpp_response) && all_finite(cpp_x) && all_finite(cpp_y) &&
            all_finite(trt_response) && all_finite(trt_x) && all_finite(trt_y);
        const DiffStats stats =
            compare_outputs(cpp_response, cpp_x, cpp_y, trt_response, trt_x,
                            trt_y);

        const bool within_tolerance =
            stats.response_max_abs <= kResponseTolerance &&
            stats.x_max_abs <= kProjectionTolerance &&
            stats.y_max_abs <= kProjectionTolerance;
        all_ok = all_ok && finite && within_tolerance;

        std::cout << fixture.name << ": response max abs diff="
                  << stats.response_max_abs << ", x_max diff="
                  << stats.x_max_abs << ", y_max diff=" << stats.y_max_abs
                  << (finite ? "" : " [non-finite output]")
                  << (within_tolerance ? "" : " [out of tolerance]") << '\n';
    }

    return all_ok ? 0 : 1;
}
