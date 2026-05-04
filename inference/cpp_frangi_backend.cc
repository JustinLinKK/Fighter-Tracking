#include "cpp_frangi_backend.h"

#include <algorithm>
#include <cmath>

CppFrangiBackend::CppFrangiBackend()
    : normalized_(ROI_SIZE * ROI_SIZE), dxx_(ROI_SIZE * ROI_SIZE),
      dxy_(ROI_SIZE * ROI_SIZE), dyy_(ROI_SIZE * ROI_SIZE) {}

void CppFrangiBackend::convolve(
    const float *input, const std::array<float, KERNEL_SIZE * KERNEL_SIZE> &kernel,
    float *output) const {
#pragma omp parallel for
    for (int y = 0; y < ROI_SIZE; ++y) {
        for (int x = 0; x < ROI_SIZE; ++x) {
            float acc = 0.0f;
            for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                const int iy = y + ky - KERNEL_RADIUS;
                if (iy < 0 || iy >= ROI_SIZE) {
                    continue;
                }
                for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                    const int ix = x + kx - KERNEL_RADIUS;
                    if (ix < 0 || ix >= ROI_SIZE) {
                        continue;
                    }
                    acc += input[iy * ROI_SIZE + ix] *
                           kernel[ky * KERNEL_SIZE + kx];
                }
            }
            output[y * ROI_SIZE + x] = acc;
        }
    }
}

bool CppFrangiBackend::run(const float *input, InferenceOutputs outputs) {
#pragma omp parallel for
    for (int i = 0; i < ROI_SIZE * ROI_SIZE; ++i) {
        normalized_[i] = input[i] / 255.0f;
    }

    convolve(normalized_.data(), G_XX, dxx_.data());
    convolve(normalized_.data(), G_XY, dxy_.data());
    convolve(normalized_.data(), G_YY, dyy_.data());

#pragma omp parallel for
    for (int y = 0; y < ROI_SIZE; ++y) {
        float row_max = 0.0f;
        for (int x = 0; x < ROI_SIZE; ++x) {
            const int idx = y * ROI_SIZE + x;
            const float trace = dxx_[idx] + dyy_[idx];
            const float det = dxx_[idx] * dyy_[idx] - dxy_[idx] * dxy_[idx];
            const float discriminant =
                std::max(trace * trace - 4.0f * det, 0.0f);
            const float a = trace + discriminant;
            const float b = trace - discriminant;
            float response =
                std::max((std::abs(a) + std::abs(b)) * (a - b), 0.0f);
            if (y < BORDER_MASK_SIZE || y >= ROI_SIZE - BORDER_MASK_SIZE ||
                x < BORDER_MASK_SIZE || x >= ROI_SIZE - BORDER_MASK_SIZE) {
                response = 0.0f;
            }
            outputs.response[idx] = response;
            row_max = std::max(row_max, response);
        }
        outputs.y_max[y] = row_max;
    }

#pragma omp parallel for
    for (int x = 0; x < ROI_SIZE; ++x) {
        float col_max = 0.0f;
        for (int y = 0; y < ROI_SIZE; ++y) {
            col_max = std::max(col_max, outputs.response[y * ROI_SIZE + x]);
        }
        outputs.x_max[x] = col_max;
    }

    return true;
}
