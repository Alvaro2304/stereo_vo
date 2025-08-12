#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

bool readCalibration(const std::string& calib_file, cv::Mat& K, double& baseline);
bool readGroundTruth(const std::string& gt_file, std::vector<cv::Mat>& poses);
