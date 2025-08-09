#include "feature_utils.h"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

bool matchStereoAndProject(
    const cv::Mat& left_img,
    const cv::Mat& right_img,
    const cv::Mat& left_img_next,
    const cv::Mat& K,
    double baseline,
    std::vector<cv::Point3f>& pts_3d,
    std::vector<cv::Point2f>& pts_2d
) {
    cv::Ptr<cv::ORB> detector = cv::ORB::create(3000);

    std::vector<cv::KeyPoint> kp_left, kp_right;
    cv::Mat desc_left, desc_right;
    detector->detectAndCompute(left_img, cv::Mat(), kp_left, desc_left);
    detector->detectAndCompute(right_img, cv::Mat(), kp_right, desc_right);

    if (kp_left.empty() || kp_right.empty()) return false;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(desc_left, desc_right, matches);

    std::vector<cv::Point2f> left_pts, right_pts;
    for (const auto& m : matches) {
        cv::Point2f pt_left = kp_left[m.queryIdx].pt;
        cv::Point2f pt_right = kp_right[m.trainIdx].pt;
        if (std::abs(pt_left.y - pt_right.y) < 2.0 && pt_left.x - pt_right.x > 0) {
            left_pts.push_back(pt_left);
            right_pts.push_back(pt_right);
        }
    }

    if (left_pts.size() < 500) return false;

    // Compute disparity and triangulate
    for (size_t i = 0; i < left_pts.size(); ++i) {
        float d = left_pts[i].x - right_pts[i].x;
        if (d <= 0.0) continue;
        float Z = K.at<double>(0, 0) * baseline / d;
        float X = (left_pts[i].x - K.at<double>(0, 2)) * Z / K.at<double>(0, 0);
        float Y = (left_pts[i].y - K.at<double>(1, 2)) * Z / K.at<double>(1, 1);
        pts_3d.emplace_back(X, Y, Z);
    }

    if (pts_3d.size() < 500) return false;

    // Track features to next left image
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> tracked_pts;
    cv::calcOpticalFlowPyrLK(left_img, left_img_next, left_pts, tracked_pts, status, err);

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            pts_2d.push_back(tracked_pts[i]);
        } else {
            pts_3d[i] = cv::Point3f(NAN, NAN, NAN);  // Invalidate
        }
    }

    // Remove invalid 3D points
    std::vector<cv::Point3f> pts_3d_clean;
    std::vector<cv::Point2f> pts_2d_clean;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        if (!std::isnan(pts_3d[i].x)) {
            pts_3d_clean.push_back(pts_3d[i]);
            pts_2d_clean.push_back(pts_2d[i]);
        }
    }

    pts_3d = pts_3d_clean;
    pts_2d = pts_2d_clean;

    return pts_3d.size() >= 500;
}
