#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::KeyPoint> bucketKeypoints(const cv::Mat &img,
                                          cv::Ptr<cv::FastFeatureDetector> &fast,
                                          int tile_h, int tile_w, int max_per_tile);
