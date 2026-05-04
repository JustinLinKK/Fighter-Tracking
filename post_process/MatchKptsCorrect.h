#pragma once

#include "../utils/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <vector>

struct MatchKptsCorrectResult {
    Box tgt_xywh_curr_orb;
};

// Number of extremal keypoints averaged per edge. Raising this suppresses
// single-match outliers at the cost of slightly blurring the edge estimate.
inline constexpr int EDGE_K = 5;

// Correct bounding box via ORB (Oriented FAST and Rotated BRIEF) keypoint
// matching. For each of the four edges we take the top-K extremal keypoints in
// kp1 (the most left / right / top / bottom), compute the median displacement
// from kp1 to kp2 over those K pairs, and apply that displacement to the
// corresponding edge of tgt_xywh_last. Median is robust: a single outlier
// match among the K pairs cannot shift the result at all.
inline MatchKptsCorrectResult
MatchKptsCorrect(const std::vector<Point> &kp1_matched,
                 const std::vector<Point> &kp2_matched,
                 const Box &tgt_xywh_last) {
    MatchKptsCorrectResult result;
    float x = tgt_xywh_last[0];
    float y = tgt_xywh_last[1];
    float w = tgt_xywh_last[2];
    float h = tgt_xywh_last[3];

    const int n = static_cast<int>(kp1_matched.size());
    const int k = std::min(EDGE_K, n);

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    // Partition idx so its first k elements are the k src points that are
    // most extreme in the requested direction, then return the median
    // displacement along `axis` over those k pairs.
    std::array<float, EDGE_K> displacements;
    auto median_edge_displacement = [&](auto src_is_more_extreme,
                                        int axis) -> float {
        std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                         src_is_more_extreme);
        for (int i = 0; i < k; ++i) {
            int id = idx[i];
            displacements[i] =
                kp2_matched[id][axis] - kp1_matched[id][axis];
        }
        // For even k, this returns the upper median (still robust).
        std::nth_element(displacements.begin(),
                         displacements.begin() + k / 2,
                         displacements.begin() + k);
        return displacements[k / 2];
    };

    // Left edge: top-K smallest x in kp1_matched
    float dx_left = median_edge_displacement(
        [&](int a, int b) { return kp1_matched[a][0] < kp1_matched[b][0]; },
        0);
    // Right edge: top-K largest x in kp1_matched
    float dx_right = median_edge_displacement(
        [&](int a, int b) { return kp1_matched[a][0] > kp1_matched[b][0]; },
        0);
    // Top edge: top-K smallest y in kp1_matched
    float dy_top = median_edge_displacement(
        [&](int a, int b) { return kp1_matched[a][1] < kp1_matched[b][1]; },
        1);
    // Bottom edge: top-K largest y in kp1_matched
    float dy_bottom = median_edge_displacement(
        [&](int a, int b) { return kp1_matched[a][1] > kp1_matched[b][1]; },
        1);

    float current_x_min = x + dx_left;
    float current_x_max = (x + w) + dx_right;
    float current_y_min = y + dy_top;
    float current_y_max = (y + h) + dy_bottom;

    result.tgt_xywh_curr_orb[0] = current_x_min;
    result.tgt_xywh_curr_orb[1] = current_y_min;
    result.tgt_xywh_curr_orb[2] = current_x_max - current_x_min;
    result.tgt_xywh_curr_orb[3] = current_y_max - current_y_min;

    return result;
}
