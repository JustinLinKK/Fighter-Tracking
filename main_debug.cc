#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "inference/inference_backend.h"
#include "opencv2/core/hal/interface.h"
#include "utils/types.h"

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
#include "utils/utils.h"

void bind_thread_to_core(int core_id) {
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
    std::cout << std::fixed << std::setprecision(6);

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
        // Frame index used as the per-frame log label (was the file stem name).
        int frame_idx = 0;
        while (true) {
            // Buffer swap: if previous iteration prefetched the next frame,
            // promote it to current. Otherwise (first iter or post-continue),
            // synchronously read.
            if (!first_iter) {
                if (!img_next_np.empty()) {
                    img_curr_np = std::move(img_next_np);
                    img_next_np.release();
                } else {
                    if (!read_gray(img_curr_np)) {
                        break; // EOF
                    }
                }
                ++frame_idx;
            }
            first_iter = false;
            using ms = std::chrono::duration<double, std::milli>;
            auto now = std::chrono::high_resolution_clock::now;

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

                std::cout << "frame " << frame_idx << std::endl;
                frame_count += 1;
                auto fstart = now();
                int w_resize = static_cast<int>(w * (ROI_SIZE / IMG_WIDTH));
                int h_resize = static_cast<int>(h * (ROI_SIZE / IMG_HEIGHT));

                auto t_after_resize_wh = now();
                cv::Mat resized_img;
                cv::resize(img_curr_np, resized_img,
                           cv::Size(ROI_SIZE, ROI_SIZE), 0, 0, cv::INTER_AREA);

                auto t_after_resize_img = now();
                cv::Mat float_img;
                resized_img.convertTo(float_img, CV_32F);
                auto t_after_convert = now();
                std::memcpy(img.data(), float_img.ptr<float>(),
                            sizeof(float) * ROI_SIZE * ROI_SIZE);
                auto t_after_memcpy = now();

                if (!backend->run(img.data(), inference_outputs)) {
                    std::cerr << "Fatal: backend run failed during first frame"
                              << std::endl;
                    return 1;
                }
                auto t_after_1st_backend = now();

                // Prefix sum on x_max and y_max
                for (int i = 1; i < ROI_SIZE; ++i) {
                    x_max[i] += x_max[i - 1];
                }
                for (int i = 1; i < ROI_SIZE; ++i) {
                    y_max[i] += y_max[i - 1];
                }
                auto t_after_cumsum = now();

                // Shifted subtraction to find target location
                tgt_xywh_curr =
                    shift_subtract(x_max.data(), y_max.data(), w_resize,
                                   h_resize);
                auto t_after_shift_sub = now();

                // Scale back to original image coordinates
                tgt_xywh_curr[0] = static_cast<float>(
                    std::round(tgt_xywh_curr[0] * (IMG_WIDTH / ROI_SIZE)));
                tgt_xywh_curr[1] = static_cast<float>(
                    std::round(tgt_xywh_curr[1] * (IMG_HEIGHT / ROI_SIZE)));
                tgt_xywh_curr[2] = w;
                tgt_xywh_curr[3] = h;
                auto t_after_rescale = now();

                // Extract ROI (Region of Interest) around target
                std::tie(roi_tl_x, roi_tl_y) =
                    GetROI(roi.get(), img_curr_np.ptr<uint8_t>(0),
                           tgt_xywh_curr);
                auto t_after_get_roi = now();

                // Threshold to detect flame and convert to float
                threshold(roi.get(), flame_cover_mask.get(), img.data(),
                          flame_signal_curr);

                auto t_after_thresh = now();

                if (!backend->run(img.data(), inference_outputs)) {
                    std::cerr << "Fatal: backend run failed on first ROI"
                              << std::endl;
                    return 1;
                }

                // Decode next frame after the first ROI inference
                double imread_next_ms = 0.0;
                {
                    auto t_imread0 = now();
                    cv::Mat tmp;
                    if (read_gray(tmp)) {
                        img_next_np = std::move(tmp);
                        has_next = true;
                    } else {
                        img_next_np.release();
                        has_next = false;
                    }
                    imread_next_ms = ms(now() - t_imread0).count();
                }
                auto t_after_2nd_backend = now();

                // Top-K keypoint extraction from response map
                topk = topk_sorted_parallel(response.data());
                auto t_after_topk = now();

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
                auto t_after_kpts = now();

                // Extract patches and compute ORB (Oriented FAST and Rotated
                // BRIEF) descriptors
                extract_all_patches(patches, img.data(), kpts_for_patches);
                dscrp_curr = extract_descriptors(patches);
                auto t_after_dscrp = now();

                // Pass state to next frame
                tgt_xywh_last = tgt_xywh_curr;
                kpts_last = kpts_curr;
                dscrp_last = dscrp_curr;
                flame_signal_last = flame_signal_curr;

                tgt_xywh_refined_last = tgt_xywh_curr;
                kpts_refined_last = kpts_curr;
                dscrp_refined_last = dscrp_curr;
                auto fend = now();

                // === Debug timing output for first frame ===
                // Subtract imread_next_ms: imread of next frame is overlapped
                // with GPU but should not count toward this frame
                ms fframe_time = fend - fstart;
                std::cout << "First frame total time  : "
                          << (fframe_time.count() - imread_next_ms)
                          << " ms (imread next: " << imread_next_ms << " ms)\n";
                std::cout << "  Resize w/h            : "
                          << ms(t_after_resize_wh - fstart).count() << " ms\n";
                std::cout << "  Resize image          : "
                          << ms(t_after_resize_img - t_after_resize_wh).count()
                          << " ms\n";
                std::cout << "  Convert to float      : "
                          << ms(t_after_convert - t_after_resize_img).count()
                          << " ms\n";
                std::cout << "  Copy to unified mem   : "
                          << ms(t_after_memcpy - t_after_convert).count()
                          << " ms\n";
                std::cout << "  1st backend run       : "
                          << ms(t_after_1st_backend - t_after_memcpy).count()
                          << " ms\n";
                std::cout << "  Cumsum                : "
                          << ms(t_after_cumsum - t_after_1st_backend).count()
                          << " ms\n";
                std::cout << "  Shift subtraction     : "
                          << ms(t_after_shift_sub - t_after_cumsum).count()
                          << " ms\n";
                std::cout << "  Rescale to orig coords: "
                          << ms(t_after_rescale - t_after_shift_sub).count()
                          << " ms\n";
                std::cout << "  GetROI                : "
                          << ms(t_after_get_roi - t_after_rescale).count()
                          << " ms\n";
                std::cout << "  Threshold + flame chk : "
                          << ms(t_after_thresh - t_after_get_roi).count()
                          << " ms\n";
                std::cout << "  2nd backend run       : "
                          << (ms(t_after_2nd_backend - t_after_thresh).count() -
                              imread_next_ms)
                          << " ms\n";
                std::cout << "  TopK                  : "
                          << ms(t_after_topk - t_after_2nd_backend).count()
                          << " ms\n";
                std::cout << "  Get keypoints         : "
                          << ms(t_after_kpts - t_after_topk).count() << " ms\n";
                std::cout << "  Patches + descriptors : "
                          << ms(t_after_dscrp - t_after_kpts).count()
                          << " ms\n";
                std::cout << "  State pass to next frm: "
                          << ms(fend - t_after_dscrp).count() << " ms\n";

                // Write annotated frame to video (not included in timing)
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
                std::cout << "frame " << frame_idx << std::endl;
                frame_count += 1;
                auto start = now();
                std::tie(roi_tl_x, roi_tl_y) =
                    GetROI(roi.get(), img_curr_np.ptr<uint8_t>(0),
                           tgt_xywh_last);

                auto t_after_roi = now();

                // Threshold and convert ROI
                threshold(roi.get(), flame_cover_mask.get(), img.data(),
                          flame_signal_curr);
                auto t_after_thresh = now();

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
                auto t_after_backend = now();

                cv::erode(cv::Mat(ROI_SIZE, ROI_SIZE, CV_32F,
                                  flame_cover_mask.get()),
                          eroded_mask, kernel);
                auto t_after_erode = now();

                double imread_next_ms = 0.0;
                {
                    auto t_imread0 = now();
                    cv::Mat tmp;
                    if (read_gray(tmp)) {
                        img_next_np = std::move(tmp);
                        has_next = true;
                    } else {
                        img_next_np.release();
                        has_next = false;
                    }
                    imread_next_ms = ms(now() - t_imread0).count();
                }

                // Apply mask to response (suppress flame and boundary regions)
                multiply(response.data(), eroded_mask.ptr<float>(),
                         orb_response.get());
                auto t_after_multiply = now();

                // Top-K keypoint extraction from masked response
                topk = topk_sorted_parallel(orb_response.get());
                auto t_after_topk = now();

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
                auto t_after_patches = now();
                dscrp_curr = extract_descriptors(patches);
                auto t_after_dscrp = now();

                // Match descriptors between consecutive frames
                matches = match_descriptors(dscrp_last, dscrp_curr);
                auto t_after_match1 = now();
                matches_refined =
                    match_descriptors(dscrp_refined_last, dscrp_curr);
                auto t_after_match2 = now();

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
                auto t_after_cumsum = now();

                // Post-processing: ORB-based correction path
                kpts_result =
                    FilterKptsMode(matches, kpts_last, kpts_curr, dscrp_curr);
                boxfil_result =
                    FilterByBox(kpts_result.src_pts, kpts_result.dst_pts,
                                kpts_result.dst_dscrp, tgt_xywh_last);
                OrbMatch_result = MatchKptsCorrect(
                    boxfil_result.kp1_boxfiltered,
                    boxfil_result.kp2_boxfiltered, tgt_xywh_last);
                auto t_after_orb = now();

                // Post-processing: similar triangle correction path
                kpts_result = FilterKpts(matches_refined, kpts_refined_last,
                                         kpts_curr, dscrp_curr);
                boxfil_result =
                    FilterByBox(kpts_result.src_pts, kpts_result.dst_pts,
                                kpts_result.dst_dscrp, tgt_xywh_last);
                smitri_check = CheckSmiTri(boxfil_result.kp1_boxfiltered,
                                           boxfil_result.kp2_boxfiltered);
                auto t_after_smitri_check = now();
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
                    std::cout << "Corrected" << std::endl;
                    std::cout << "Corrected box [" << corrected_xywh[0] << ", "
                              << corrected_xywh[1] << ", " << corrected_xywh[2]
                              << ", " << corrected_xywh[3] << "]\n";

                    tgt_xywh_refined_last = corrected_xywh;
                    kpts_refined_last = kpts_curr;
                    dscrp_refined_last = dscrp_curr;

                    tgt_xywh_last = corrected_xywh;
                    kpts_last = kpts_curr;
                    dscrp_last = dscrp_curr;
                    flame_signal_last = flame_signal_curr;
                } else {
                    // Use ORB match result without similar triangle correction
                    std::cout << "Not corrected" << std::endl;
                    std::cout << "[" << OrbMatch_result.tgt_xywh_curr_orb[0]
                              << ", " << OrbMatch_result.tgt_xywh_curr_orb[1]
                              << ", " << OrbMatch_result.tgt_xywh_curr_orb[2]
                              << ", " << OrbMatch_result.tgt_xywh_curr_orb[3]
                              << "]\n";

                    tgt_xywh_last = OrbMatch_result.tgt_xywh_curr_orb;
                    kpts_last = kpts_curr;
                    dscrp_last = dscrp_curr;
                    flame_signal_last = flame_signal_curr;
                }
                auto end = now();

                // === Debug timing output for subsequent frames ===
                // Subtract imread_next_ms: imread of next frame is overlapped
                // with GPU but should not count toward this frame's RUN time
                std::cout << "RUN time: "
                          << (ms(end - start).count() - imread_next_ms)
                          << " ms (imread next: " << imread_next_ms << " ms)\n";
                std::cout << "  GetROI                : "
                          << ms(t_after_roi - start).count() << " ms\n";
                std::cout << "  Threshold + flame chk : "
                          << ms(t_after_thresh - t_after_roi).count()
                          << " ms\n";
                std::cout << "  Backend run           : "
                          << ms(t_after_backend - t_after_thresh).count()
                          << " ms\n";
                std::cout << "  Erode mask            : "
                          << ms(t_after_erode - t_after_backend).count()
                          << " ms\n";
                std::cout << "  Decode next frame     : "
                          << imread_next_ms
                          << " ms\n";
                std::cout << "  Multiply resp * mask  : "
                          << ms(t_after_multiply - t_after_erode).count()
                          << " ms\n";
                std::cout << "  TopK                  : "
                          << ms(t_after_topk - t_after_multiply).count()
                          << " ms\n";
                std::cout << "  Extract patches       : "
                          << ms(t_after_patches - t_after_topk).count()
                          << " ms\n";
                std::cout << "  Extract descriptors   : "
                          << ms(t_after_dscrp - t_after_patches).count()
                          << " ms\n";
                std::cout << "  Match descriptors 1   : "
                          << ms(t_after_match1 - t_after_dscrp).count()
                          << " ms\n";
                std::cout << "  Match descriptors 2   : "
                          << ms(t_after_match2 - t_after_match1).count()
                          << " ms\n";
                std::cout << "  Cumsum + shift sub    : "
                          << ms(t_after_cumsum - t_after_match2).count()
                          << " ms\n";
                std::cout << "  ORB post-process      : "
                          << ms(t_after_orb - t_after_cumsum).count()
                          << " ms\n";
                std::cout << "  SmiTri check          : "
                          << ms(t_after_smitri_check - t_after_orb).count()
                          << " ms\n";
                std::cout << "  SmiTri apply + output : "
                          << ms(end - t_after_smitri_check).count() << " ms\n";
                std::cout << "\n";

                // Write annotated frame to video (not included in timing)
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
