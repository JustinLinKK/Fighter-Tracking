#pragma once

#include <array>
#include <cstdint>

// Shared type aliases used across the tracking pipeline
using Match = std::array<float, 3>;         // {src_idx, dst_idx, distance}
using Point = std::array<float, 2>;         // {x, y}
using Descriptor = std::array<uint64_t, 4>; // 256-bit packed binary descriptor
using Box = std::array<float, 4>;           // {x, y, w, h}

// Image and ROI (Region of Interest) constants.
// IMG_WIDTH/IMG_HEIGHT must match the resolution of frames being processed
// (see VIDEO_FILE in main.cc / main_debug.cc).
inline constexpr int ROI_SIZE = 480;
inline constexpr float IMG_WIDTH = 640.0f;
inline constexpr float IMG_HEIGHT = 512.0f;

inline constexpr int TOPK = 40;
inline constexpr int TOPK_THREADS = 4;
inline constexpr int TOPK_TOTAL = TOPK * TOPK_THREADS;

// Pixel intensity at or above this value is treated as backfire / flame
// saturation. The flame_cover_mask zeros out these pixels so they do not
// contribute to the Frangi response used for keypoint extraction.
inline constexpr uint8_t FLAME_THRESH = 250;
