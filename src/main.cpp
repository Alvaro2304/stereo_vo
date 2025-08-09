#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

// Read KITTI calib.txt
bool readCalibration(const string& calib_file, Mat& K, double& baseline) {
    ifstream file(calib_file);
    if (!file.is_open()) {
        cerr << "Cannot open calib file: " << calib_file << endl;
        return false;
    }
    string line;
    double P0[12], P1[12];
    while (getline(file, line)) {
        stringstream ss(line);
        string tag;
        ss >> tag;
        if (tag == "P0:") {
            for (int i = 0; i < 12; ++i) ss >> P0[i];
        } else if (tag == "P1:") {
            for (int i = 0; i < 12; ++i) ss >> P1[i];
        }
    }
    K = (Mat_<double>(3,3) << P0[0], P0[1], P0[2],
                               P0[4], P0[5], P0[6],
                               P0[8], P0[9], P0[10]);
    baseline = (P0[3] - P1[3]) / P0[0];
    return true;
}

// Read KITTI ground truth pose
bool readGroundTruth(const string& gt_file, vector<Mat>& poses) {
    ifstream file(gt_file);
    if (!file.is_open()) {
        cerr << "Cannot open ground truth file: " << gt_file << endl;
        return false;
    }
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        Mat pose = Mat::eye(4, 4, CV_64F);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                ss >> pose.at<double>(r, c);
        poses.push_back(pose);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./stereo_vo <sequence_number>" << endl;
        return 1;
    }

    string seq = argv[1];
    string base = "../../kitti_dataset/data_odometry_gray/dataset/sequences/" + seq + "/";
    string left_dir = base + "image_0/";
    string right_dir = base + "image_1/";
    string calib_path = base + "calib.txt";
    string gt_path = "../../kitti_dataset/data_odometry_poses/dataset/poses/" + seq + ".txt";

    // Load calibration
    Mat K;
    double baseline;
    if (!readCalibration(calib_path, K, baseline)) return 1;

    // Load ground truth
    vector<Mat> gt_poses;
    if (!readGroundTruth(gt_path, gt_poses)) return 1;

    // ORB feature detector
    Ptr<ORB> orb = ORB::create(2000);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // Trajectory visualization
    Mat traj = Mat::zeros(800, 800, CV_8UC3);
    double traj_scale = 0.5; // Adjust for visibility

    // Initialize pose
    Mat curr_pose = gt_poses[0].clone();

    for (int frame_id = 0; frame_id < (int)gt_poses.size()-1; frame_id++) {
        char fname[256];
        sprintf(fname, "%06d.png", frame_id);
        Mat imgL = imread(left_dir + fname, IMREAD_GRAYSCALE);
        Mat imgR = imread(right_dir + fname, IMREAD_GRAYSCALE);
        if (imgL.empty() || imgR.empty()) break;

        // Detect features in left and right
        vector<KeyPoint> kptsL, kptsR;
        Mat descL, descR;
        orb->detectAndCompute(imgL, noArray(), kptsL, descL);
        orb->detectAndCompute(imgR, noArray(), kptsR, descR);

        // Match left-right
        vector<DMatch> matches_lr;
        matcher->match(descL, descR, matches_lr);

        // Filter good matches
        double max_dist = 0; double min_dist = 100;
        for (auto &m : matches_lr) {
            double dist = m.distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }
        vector<Point2f> ptsL, ptsR;
        for (auto &m : matches_lr) {
            if (m.distance <= max(2*min_dist, 30.0)) {
                ptsL.push_back(kptsL[m.queryIdx].pt);
                ptsR.push_back(kptsR[m.trainIdx].pt);
            }
        }

        // Triangulate 3D points
        Mat projL = K * Mat::eye(3, 4, CV_64F);
        Mat projR = K * (Mat_<double>(3,4) << 1, 0, 0, -baseline, 0, 1, 0, 0, 0, 0, 1, 0);
        Mat pts4D;
        triangulatePoints(projL, projR, ptsL, ptsR, pts4D);

        // Project to 3D
        vector<Point3f> points3D;
        for (int i = 0; i < pts4D.cols; i++) {
            Mat x = pts4D.col(i);
            x /= x.at<float>(3);
            points3D.push_back(Point3f(x.at<float>(0), x.at<float>(1), x.at<float>(2)));
        }

        // Load next left image and match
        sprintf(fname, "%06d.png", frame_id+1);
        Mat imgL_next = imread(left_dir + fname, IMREAD_GRAYSCALE);
        vector<KeyPoint> kptsL_next;
        Mat descL_next;
        orb->detectAndCompute(imgL_next, noArray(), kptsL_next, descL_next);

        vector<DMatch> matches_ll;
        matcher->match(descL, descL_next, matches_ll);

        vector<Point2f> pts_curr, pts_next;
        for (auto &m : matches_ll) {
            pts_curr.push_back(kptsL[m.queryIdx].pt);
            pts_next.push_back(kptsL_next[m.trainIdx].pt);
        }

        // Essential matrix and pose
        Mat E, R, t, mask;
        E = findEssentialMat(pts_next, pts_curr, K, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, pts_next, pts_curr, K, R, t, mask);

        // Update pose
        Mat Rt = Mat::eye(4, 4, CV_64F);
        R.copyTo(Rt(Range(0, 3), Range(0, 3)));
        t.copyTo(Rt(Range(0, 3), Range(3, 4)));
        curr_pose = curr_pose * Rt.inv();

        // Draw trajectories
        Point gt_pos(gt_poses[frame_id].at<double>(0,3)*traj_scale + 400,
                     gt_poses[frame_id].at<double>(2,3)*traj_scale + 400);
        Point est_pos(curr_pose.at<double>(0,3)*traj_scale + 400,
                      curr_pose.at<double>(2,3)*traj_scale + 400);
        circle(traj, gt_pos, 1, Scalar(0,255,0), 2); // green GT
        circle(traj, est_pos, 1, Scalar(0,0,255), 2); // red estimated

        imshow("Trajectory", traj);
        if (waitKey(1) == 27) break; // ESC
    }
    return 0;
}
