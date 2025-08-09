#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

bool matchStereoAndProject(const cv::Mat& left_img, const cv::Mat& right_img,
                           const cv::Mat& left_img_next,
                           const cv::Mat& K, double baseline,
                           std::vector<cv::Point3f>& pts_3d,
                           std::vector<cv::Point2f>& pts_2d);