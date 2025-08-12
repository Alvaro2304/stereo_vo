#include "feature_bucket.h"
#include <algorithm>

std::vector<cv::KeyPoint> bucketKeypoints(
    const cv::Mat &img,
    cv::Ptr<cv::FastFeatureDetector> &fast,
    int tile_h,
    int tile_w,
    int max_per_tile)
{
    std::vector<cv::KeyPoint> kept;
    const int H = img.rows, W = img.cols;
    for (int y = 0; y < H; y += tile_h) {
        for (int x = 0; x < W; x += tile_w) {
            cv::Rect roi(x, y, std::min(tile_w, W - x), std::min(tile_h, H - y));
            cv::Mat patch = img(roi);

            std::vector<cv::KeyPoint> kps_patch;
            fast->detect(patch, kps_patch);

            // adjust to image coords
            for (auto &kp : kps_patch) kp.pt += cv::Point2f((float)x, (float)y);

            if ((int)kps_patch.size() > max_per_tile) {
                // keep top responses
                std::sort(kps_patch.begin(), kps_patch.end(),
                          [](const cv::KeyPoint &a, const cv::KeyPoint &b){ return a.response > b.response; });
                kps_patch.resize(max_per_tile);
            }
            // append
            kept.insert(kept.end(), kps_patch.begin(), kps_patch.end());
        }
    }
    return kept;
}
