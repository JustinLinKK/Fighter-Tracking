#pragma once

#include <array>
#include <vector>

#include "inference_backend.h"
#include "../utils/types.h"

class CppFrangiBackend : public InferenceBackend {
  public:
    CppFrangiBackend();
    bool run(const float *input, InferenceOutputs outputs) override;

  private:
    static constexpr int KERNEL_SIZE = 7;
    static constexpr int KERNEL_RADIUS = KERNEL_SIZE / 2;
    static constexpr int BORDER_MASK_SIZE = 10;

    static constexpr std::array<float, KERNEL_SIZE * KERNEL_SIZE> G_XX = {
        1.5713e-04f, 7.1784e-04f, 0.0000e+00f, -1.7681e-03f, 0.0000e+00f,
        7.1784e-04f, 1.5713e-04f, 1.9142e-03f, 8.7451e-03f, 0.0000e+00f,
        -2.1539e-02f, 0.0000e+00f, 8.7451e-03f, 1.9142e-03f, 8.5790e-03f,
        3.9193e-02f, 0.0000e+00f, -9.6532e-02f, 0.0000e+00f, 3.9193e-02f,
        8.5790e-03f, 1.4144e-02f, 6.4618e-02f, 0.0000e+00f, -1.5915e-01f,
        0.0000e+00f, 6.4618e-02f, 1.4144e-02f, 8.5790e-03f, 3.9193e-02f,
        0.0000e+00f, -9.6532e-02f, 0.0000e+00f, 3.9193e-02f, 8.5790e-03f,
        1.9142e-03f, 8.7451e-03f, 0.0000e+00f, -2.1539e-02f, 0.0000e+00f,
        8.7451e-03f, 1.9142e-03f, 1.5713e-04f, 7.1784e-04f, 0.0000e+00f,
        -1.7681e-03f, 0.0000e+00f, 7.1784e-04f, 1.5713e-04f};

    static constexpr std::array<float, KERNEL_SIZE * KERNEL_SIZE> G_XY = {
        2.0000e-04f, 1.4000e-03f, 3.2000e-03f, -0.0000e+00f, -3.2000e-03f,
        -1.4000e-03f, -2.0000e-04f, 1.4000e-03f, 1.1700e-02f, 2.6100e-02f,
        -0.0000e+00f, -2.6100e-02f, -1.1700e-02f, -1.4000e-03f, 3.2000e-03f,
        2.6100e-02f, 5.8500e-02f, -0.0000e+00f, -5.8500e-02f, -2.6100e-02f,
        -3.2000e-03f, -0.0000e+00f, -0.0000e+00f, -0.0000e+00f, 0.0000e+00f,
        0.0000e+00f, 0.0000e+00f, 0.0000e+00f, -3.2000e-03f, -2.6100e-02f,
        -5.8500e-02f, 0.0000e+00f, 5.8500e-02f, 2.6100e-02f, 3.2000e-03f,
        -1.4000e-03f, -1.1700e-02f, -2.6100e-02f, 0.0000e+00f, 2.6100e-02f,
        1.1700e-02f, 1.4000e-03f, -2.0000e-04f, -1.4000e-03f, -3.2000e-03f,
        0.0000e+00f, 3.2000e-03f, 1.4000e-03f, 2.0000e-04f};

    static constexpr std::array<float, KERNEL_SIZE * KERNEL_SIZE> G_YY = {
        1.5713e-04f, 1.9142e-03f, 8.5790e-03f, 1.4144e-02f, 8.5790e-03f,
        1.9142e-03f, 1.5713e-04f, 7.1784e-04f, 8.7451e-03f, 3.9193e-02f,
        6.4618e-02f, 3.9193e-02f, 8.7451e-03f, 7.1784e-04f, 0.0000e+00f,
        0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f,
        0.0000e+00f, -1.7681e-03f, -2.1539e-02f, -9.6532e-02f, -1.5915e-01f,
        -9.6532e-02f, -2.1539e-02f, -1.7681e-03f, 0.0000e+00f, 0.0000e+00f,
        0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f,
        7.1784e-04f, 8.7451e-03f, 3.9193e-02f, 6.4618e-02f, 3.9193e-02f,
        8.7451e-03f, 7.1784e-04f, 1.5713e-04f, 1.9142e-03f, 8.5790e-03f,
        1.4144e-02f, 8.5790e-03f, 1.9142e-03f, 1.5713e-04f};

    void convolve(const float *input,
                  const std::array<float, KERNEL_SIZE * KERNEL_SIZE> &kernel,
                  float *output) const;

    std::vector<float> normalized_;
    std::vector<float> dxx_;
    std::vector<float> dxy_;
    std::vector<float> dyy_;
};
