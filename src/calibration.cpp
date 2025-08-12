#include "calibration.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool readCalibration(const std::string& calib_file, cv::Mat& K, double& baseline) {
    std::ifstream file(calib_file);
    if (!file.is_open()) { std::cerr << "Cannot open calib file: " << calib_file << std::endl; return false; }
    std::string line;
    double P0[12] = {0}, P1[12] = {0};
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string tag; ss >> tag;
        if (tag == "P0:") for (int i = 0; i < 12; ++i) ss >> P0[i];
        else if (tag == "P1:") for (int i = 0; i < 12; ++i) ss >> P1[i];
    }
    K = (cv::Mat_<double>(3,3) << P0[0], P0[1], P0[2],
                                   P0[4], P0[5], P0[6],
                                   P0[8], P0[9], P0[10]);
    double fx = P0[0]; double tx0 = P0[3]; double tx1 = P1[3];
    baseline = (tx0 - tx1) / fx;
    return true;
}

bool readGroundTruth(const std::string& gt_file, std::vector<cv::Mat>& poses) {
    std::ifstream file(gt_file);
    if (!file.is_open()) { std::cerr << "Cannot open ground truth file: " << gt_file << std::endl; return false; }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        cv::Mat pose = cv::Mat::eye(4,4,CV_64F);
        for (int r=0;r<3;r++) for (int c=0;c<4;c++) ss >> pose.at<double>(r,c);
        poses.push_back(pose);
    }
    return !poses.empty();
}
