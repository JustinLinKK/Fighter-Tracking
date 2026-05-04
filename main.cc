#include <algorithm>
#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "inference/inference_backend.h"
#include "post_process/CtrCorrect.h"
#include "post_process/FilterByBox.h"
#include "post_process/FilterKpts.h"
#include "post_process/MatchKptsCorrect.h"
#include "post_process/SmiTri.h"
#include "utils/box_size.h"
#include "utils/descriptor_match.h"
#include "utils/get_roi.h"
#include "utils/parallel_topk.h"
#include "utils/slice.h"
#include "utils/thresh.h"
#include "utils/types.h"
#include "utils/utils.h"

void bind_thread_to_core(int core_id) { // NOTE: Make sure that program doesn't
                                        // jump between different cores
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

constexpr int NUM_THREADS = 4;
constexpr std::string_view ENGINE_FILE =
    "./engine_model/Norm_Grad_Response_Masked_Max_480.engine";
constexpr std::string_view VIDEO_FILE =
    "./Datasets/Anti-UAV-RGBT/test/test1/infrared.mp4";
constexpr std::string_view DEFAULT_BACKEND = "cpp";
// Initial target box for the very first frame, taken from
// Datasets/Anti-UAV-RGBT/test/test1/infrared.json (gt_rect[0] = [x, y, w, h]).
constexpr float INIT_BOX_W = 52.0f;
constexpr float INIT_BOX_H = 39.0f;

int frame_count = 0;

int main(int argc, char **argv) {
    bind_thread_to_core(0);
    omp_set_num_threads(NUM_THREADS);
    std::cout << std::fixed << std::setprecision(16);

    const std::string backend_name =
        argc > 1 ? std::string(argv[1]) : std::string(DEFAULT_BACKEND);
    std::string backend_error;
    auto backend =
        create_backend(backend_name, std::string(ENGINE_FILE), backend_error);
    if (!backend) {
        std::cerr << "Fatal: failed to initialize backend '" << backend_name
                  << "': " << backend_error << std::endl;
        return 1;
    }

    FilterKptsResult kpts_result;
    FilterByBoxResult boxfil_result;
    SmiTriCheck smitri_check;
    std::array<double, 2> new_ctr_pts;
    Point ctr_pt_last;
    Box corrected_xywh;
    Box frangi_xyxy;
    MatchKptsCorrectResult OrbMatch_result;

    cv::VideoCapture cap{std::string(VIDEO_FILE)};
    if (!cap.isOpened()) {
        std::cerr << "Fatal: failed to open video " << VIDEO_FILE << std::endl;
        return 1;
    }
    // Reads one BGR frame from the video, converts it to single-channel
    // grayscale, and writes it into `out`. Returns false at EOF / on failure.
    auto read_gray = [&cap](cv::Mat &out) -> bool {
        cv::Mat bgr;
        if (!cap.read(bgr) || bgr.empty()) {
            return false;
        }
        cv::cvtColor(bgr, out, cv::COLOR_BGR2GRAY);
        return true;
    };

    std::array<float, 4> tgt_xywh_curr, tgt_xywh_last, tgt_xywh_refined_last;
    std::array<Point, 40> kpts_for_patches, kpts_curr, kpts_last,
        kpts_refined_last;
    std::array<Descriptor, 40> dscrp_curr, dscrp_last, dscrp_refined_last;

    std::array<std::array<float, 3>, 40> matches;
    std::array<std::array<float, 3>, 40> matches_refined;

    bool flame_signal_curr = true;
    bool flame_signal_last = false;

    // Only the very first invocation of the frame-0 branch uses the
    // hardcoded INIT_BOX_W/H. A later fallback (target lost → frame_count = 0)
    // reuses tgt_xywh_last[2,3], which still holds the last valid tracked size.
    bool first_init = true;

    auto roi = std::make_unique<uint8_t[]>(ROI_SIZE * ROI_SIZE);
    auto flame_cover_mask = std::make_unique<float[]>(ROI_SIZE * ROI_SIZE);
    auto orb_response = std::make_unique<float[]>(ROI_SIZE * ROI_SIZE);
    std::vector<float> img(ROI_SIZE * ROI_SIZE);
    std::vector<float> response(ROI_SIZE * ROI_SIZE);
    std::vector<float> x_max(ROI_SIZE);
    std::vector<float> y_max(ROI_SIZE);
    InferenceOutputs inference_outputs{response.data(), x_max.data(),
                                       y_max.data()};

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat eroded_mask(ROI_SIZE, ROI_SIZE, CV_32F);
    int roi_tl_x, roi_tl_y;

    std::array<std::pair<float, int>, TOPK> topk;
    std::array<std::array<std::array<float, 25>, 25>, 40> patches;

    cv::VideoWriter writer(
        "output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
        cv::Size(static_cast<int>(IMG_WIDTH), static_cast<int>(IMG_HEIGHT)));

    {
        // Double-buffered video decode: while GPU runs frame N's TRT, CPU
        // decodes frame N+1 from the video stream.
        cv::Mat img_curr_np, img_next_np;
        if (!read_gray(img_curr_np)) {
            std::cerr << "Fatal: video has no frames" << std::endl;
            return 1;
        }
        bool first_iter = true;
        bool has_next = true;
        while (true) {
            // Buffer swap: if previous iteration prefetched the next frame,
            // promote it. Otherwise (first iter or post-continue), read sync.
            if (!first_iter) {
                if (!img_next_np.empty()) {
                    img_curr_np = std::move(img_next_np);
                    img_next_np.release();
                } else {
                    if (!read_gray(img_curr_np)) {
                        break; // EOF
                    }
                }
            }
            first_iter = false;
            if (frame_count == 0) {
                // === First frame processing (also used as the fallback
                // re-init path when the target is lost) ===
                // On the very first invocation we use the hardcoded
                // INIT_BOX_W/H taken from infrared.json. On any later fallback
                // we reuse the last tracked size in tgt_xywh_last.
                float w, h;
                if (first_init) {
                    w = INIT_BOX_W;
                    h = INIT_BOX_H;
                    first_init = false;
                } else {
                    w = tgt_xywh_last[2];
                    h = tgt_xywh_last[3];
                }

                frame_count += 1;

                // NOTE: Even need to adjust its height and width
                int w_resize = static_cast<int>(w * (ROI_SIZE / IMG_WIDTH));
                int h_resize = static_cast<int>(h * (ROI_SIZE / IMG_HEIGHT));

                cv::Mat resized_img;
                cv::resize(img_curr_np, resized_img,
                           cv::Size(ROI_SIZE, ROI_SIZE), 0, 0, cv::INTER_AREA);

                cv::Mat float_img;
                resized_img.convertTo(float_img, CV_32F);
                std::memcpy(img.data(), float_img.ptr<float>(),
                            sizeof(float) * ROI_SIZE * ROI_SIZE);

                if (!backend->run(img.data(), inference_outputs)) {
                    std::cerr << "Fatal: backend run failed during first frame"
                              << std::endl;
                    return 1;
                }

                // Prefix sum on x_max and y_max
                for (int i = 1; i < ROI_SIZE; ++i) {
                    x_max[i] += x_max[i - 1];
                }
                for (int i = 1; i < ROI_SIZE; ++i) {
                    y_max[i] += y_max[i - 1];
                }

                // Shifted subtraction to find target location
                tgt_xywh_curr =
                    shift_subtract(x_max.data(), y_max.data(), w_resize,
                                   h_resize);

                // Scale back to original image coordinates
                tgt_xywh_curr[0] = static_cast<float>(
                    std::round(tgt_xywh_curr[0] * (IMG_WIDTH / ROI_SIZE)));
                tgt_xywh_curr[1] = static_cast<float>(
                    std::round(tgt_xywh_curr[1] * (IMG_HEIGHT / ROI_SIZE)));
                tgt_xywh_curr[2] = w;
                tgt_xywh_curr[3] = h;

                // Extract ROI (Region of Interest) around target
                std::tie(roi_tl_x, roi_tl_y) =
                    GetROI(roi.get(), img_curr_np.ptr<uint8_t>(0),
                           tgt_xywh_curr);

                // Threshold to detect flame and convert to float
                threshold(roi.get(), flame_cover_mask.get(), img.data(),
                          flame_signal_curr);

                if (!backend->run(img.data(), inference_outputs)) {
                    std::cerr << "Fatal: backend run failed on first ROI"
                              << std::endl;
                    return 1;
                }

                // Decode next frame after the first ROI inference
                {
                    cv::Mat tmp;
                    if (read_gray(tmp)) {
                        img_next_np = std::move(tmp);
                        has_next = true;
                    } else {
                        img_next_np.release();
                        has_next = false;
                    }
                }

                // Top-K keypoint extraction from response map
                topk = topk_sorted_parallel(response.data());

                // NOTE: Maybe can be improved here
                int i = 0;
                for (const auto &vi : topk) {
                    int index = vi.second;
                    int y = index / ROI_SIZE;
                    int x = index % ROI_SIZE;
                    kpts_for_patches[i] = {static_cast<float>(y),
                                           static_cast<float>(x)};
                    kpts_curr[i] = {static_cast<float>(x + roi_tl_x),
                                    static_cast<float>(y + roi_tl_y)};
                    ++i;
                }

                // Extract patches and compute ORB (Oriented FAST and Rotated
                // BRIEF) descriptors
                extract_all_patches(patches, img.data(), kpts_for_patches);
                dscrp_curr = extract_descriptors(patches);

                // Pass state to next frame
                tgt_xywh_last = tgt_xywh_curr;
                kpts_last = kpts_curr;
                dscrp_last = dscrp_curr;
                flame_signal_last = flame_signal_curr;

                tgt_xywh_refined_last = tgt_xywh_curr;
                kpts_refined_last = kpts_curr;
                dscrp_refined_last = dscrp_curr;

                // Write annotated frame to video
                {
                    cv::Mat vis;
                    cv::cvtColor(img_curr_np, vis, cv::COLOR_GRAY2BGR);
                    cv::rectangle(vis,
                                  cv::Point(static_cast<int>(tgt_xywh_curr[0]),
                                            static_cast<int>(tgt_xywh_curr[1])),
                                  cv::Point(static_cast<int>(tgt_xywh_curr[0] +
                                                             tgt_xywh_curr[2]),
                                            static_cast<int>(tgt_xywh_curr[1] +
                                                             tgt_xywh_curr[3])),
                                  cv::Scalar(0, 255, 0), 2);
                    writer.write(vis);
                }

            } else {
                // === Subsequent frame processing ===
                frame_count += 1;
                auto start = std::chrono::high_resolution_clock::now();

                // Extract ROI using last frame's target location
                std::tie(roi_tl_x, roi_tl_y) =
                    GetROI(roi.get(), img_curr_np.ptr<uint8_t>(0),
                           tgt_xywh_last);

                // Threshold and convert ROI
                threshold(roi.get(), flame_cover_mask.get(), img.data(),
                          flame_signal_curr);

                // If flame was present last frame but gone now, reset to
                // first-frame mode
                if (flame_signal_last && !flame_signal_curr) {
                    frame_count = 0;
                    flame_signal_last = flame_signal_curr;
                    continue;
                }

                if (!backend->run(img.data(), inference_outputs)) {
                    std::cerr << "Fatal: backend run failed during tracking"
                              << std::endl;
                    return 1;
                }

                // Erode flame mask on CPU after inference
                cv::Mat flame_cover_mask_mat(ROI_SIZE, ROI_SIZE, CV_32F,
                                             flame_cover_mask.get());
                cv::erode(flame_cover_mask_mat, eroded_mask, kernel);
                auto imread_start =
                    std::chrono::high_resolution_clock::now();
                {
                    cv::Mat tmp;
                    if (read_gray(tmp)) {
                        img_next_np = std::move(tmp);
                        has_next = true;
                    } else {
                        img_next_np.release();
                        has_next = false;
                    }
                }
                auto imread_end = std::chrono::high_resolution_clock::now();

                // Apply mask to response (suppress flame and boundary regions)
                multiply(response.data(), eroded_mask.ptr<float>(),
                         orb_response.get());

                // Top-K keypoint extraction from masked response
                topk = topk_sorted_parallel(orb_response.get());

                int i = 0;
                for (const auto &vi : topk) {
                    int index = vi.second;
                    int y = index / ROI_SIZE;
                    int x = index % ROI_SIZE;
                    kpts_for_patches[i] = {static_cast<float>(y),
                                           static_cast<float>(x)};
                    kpts_curr[i] = {static_cast<float>(x + roi_tl_x),
                                    static_cast<float>(y + roi_tl_y)};
                    ++i;
                }

                // Extract patches and compute descriptors
                extract_all_patches(patches, img.data(), kpts_for_patches);
                dscrp_curr = extract_descriptors(patches);

                // Match descriptors between consecutive frames
                matches = match_descriptors(dscrp_last, dscrp_curr);
                matches_refined =
                    match_descriptors(dscrp_refined_last, dscrp_curr);

                // Prefix sum for cumulative response
                for (int i = 1; i < ROI_SIZE; ++i) {
                    x_max[i] += x_max[i - 1];
                }
                for (int i = 1; i < ROI_SIZE; ++i) {
                    y_max[i] += y_max[i - 1];
                }

                // Shifted subtraction for target localization.
                // Use the running box size from tgt_xywh_last (updated by
                // post-processing) instead of a per-frame dictionary lookup.
                tgt_xywh_curr =
                    shift_subtract(x_max.data(), y_max.data(),
                                   tgt_xywh_last[2], tgt_xywh_last[3]);

                // Post-processing: ORB-based correction path
                kpts_result =
                    FilterKptsMode(matches, kpts_last, kpts_curr, dscrp_curr);
                boxfil_result =
                    FilterByBox(kpts_result.src_pts, kpts_result.dst_pts,
                                kpts_result.dst_dscrp, tgt_xywh_last);
                OrbMatch_result = MatchKptsCorrect(
                    boxfil_result.kp1_boxfiltered,
                    boxfil_result.kp2_boxfiltered, tgt_xywh_last);

                // Post-processing: similar triangle correction path
                kpts_result = FilterKpts(matches_refined, kpts_refined_last,
                                         kpts_curr, dscrp_curr);
                boxfil_result =
                    FilterByBox(kpts_result.src_pts, kpts_result.dst_pts,
                                kpts_result.dst_dscrp, tgt_xywh_last);
                smitri_check = CheckSmiTri(boxfil_result.kp1_boxfiltered,
                                           boxfil_result.kp2_boxfiltered);

                if (smitri_check.apply) {
                    // Convert to double precision for similar triangle
                    // computation
                    std::array<std::array<double, 2>, 3> long_src_pts;
                    std::array<std::array<double, 2>, 3> long_dst_pts;

                    for (int i = 0; i < 3; ++i) {
                        long_src_pts[i] = {
                            static_cast<double>(smitri_check.src_points[i][0]),
                            static_cast<double>(smitri_check.src_points[i][1])};
                        long_dst_pts[i] = {
                            static_cast<double>(smitri_check.dst_points[i][0]),
                            static_cast<double>(smitri_check.dst_points[i][1])};
                    }
                    // Apply similar triangle (SmiTri) correction
                    frangi_xyxy[0] = tgt_xywh_curr[0] + roi_tl_x;
                    frangi_xyxy[1] = tgt_xywh_curr[1] + roi_tl_y;
                    frangi_xyxy[2] =
                        tgt_xywh_curr[0] + roi_tl_x + tgt_xywh_curr[2];
                    frangi_xyxy[3] =
                        tgt_xywh_curr[1] + roi_tl_y + tgt_xywh_curr[3];

                    ctr_pt_last = {tgt_xywh_refined_last[0] +
                                       tgt_xywh_refined_last[2] / 2,
                                   tgt_xywh_refined_last[1] +
                                       tgt_xywh_refined_last[3] / 2};
                    std::array<double, 2> long_ctr_pt_last = {
                        static_cast<double>(ctr_pt_last[0]),
                        static_cast<double>(ctr_pt_last[1])};

                    new_ctr_pts =
                        SmiTri(long_src_pts, long_dst_pts, long_ctr_pt_last);
                    std::array<float, 2> fp_new_ctr_pts = {
                        static_cast<float>(new_ctr_pts[0]),
                        static_cast<float>(new_ctr_pts[1])};
                    corrected_xywh = CtrCorrect(fp_new_ctr_pts, frangi_xyxy,
                                                tgt_xywh_refined_last,
                                                smitri_check.dst_points);

                    tgt_xywh_refined_last = corrected_xywh;
                    kpts_refined_last = kpts_curr;
                    dscrp_refined_last = dscrp_curr;

                    tgt_xywh_last = corrected_xywh;
                    kpts_last = kpts_curr;
                    dscrp_last = dscrp_curr;
                    flame_signal_last = flame_signal_curr;
                } else {
                    // Use ORB match result without similar triangle correction
                    tgt_xywh_last = OrbMatch_result.tgt_xywh_curr_orb;
                    kpts_last = kpts_curr;
                    dscrp_last = dscrp_curr;
                    flame_signal_last = flame_signal_curr;
                }

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> whole_time =
                    end - start;
                std::chrono::duration<double, std::milli> imread_time =
                    imread_end - imread_start;
                std::chrono::duration<double, std::milli> run_time =
                    whole_time - imread_time;
                std::cout << "RUN time: " << run_time.count()
                          << " ms (imread next: " << imread_time.count()
                          << " ms)\n\n";

                // Write annotated frame to video
                {
                    cv::Mat vis;
                    cv::cvtColor(img_curr_np, vis, cv::COLOR_GRAY2BGR);
                    cv::rectangle(vis,
                                  cv::Point(static_cast<int>(tgt_xywh_last[0]),
                                            static_cast<int>(tgt_xywh_last[1])),
                                  cv::Point(static_cast<int>(tgt_xywh_last[0] +
                                                             tgt_xywh_last[2]),
                                            static_cast<int>(tgt_xywh_last[1] +
                                                             tgt_xywh_last[3])),
                                  cv::Scalar(0, 255, 0), 2);
                    writer.write(vis);
                }
            }
        }
    }

    writer.release();
}
