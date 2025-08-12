// klt_stereo_vo.cpp  (replace your main implementation with this)
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "calibration.h"
#include "feature_bucket.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc < 2) { cerr << "Usage: ./stereo_vo <sequence_number>" << endl; return 1; }
    string seq = argv[1];
    string base = "../../kitti_dataset/data_odometry_gray/dataset/sequences/" + seq + "/";
    string left_dir = base + "image_0/";
    string right_dir = base + "image_1/";
    string calib_path = base + "calib.txt";
    string gt_path = "../../kitti_dataset/data_odometry_poses/dataset/poses/" + seq + ".txt";

    // load calibration & GT
    Mat K; double baseline;
    if (!readCalibration(calib_path, K, baseline)) return -1;
    vector<Mat> gt_poses;
    if (!readGroundTruth(gt_path, gt_poses)) {
        cerr << "No GT loaded; still running but no GT visualization." << endl;
    }

    double fx = K.at<double>(0,0), fy = K.at<double>(1,1), cx = K.at<double>(0,2), cy = K.at<double>(1,2);
    cout << "K:\n" << K << "\nbaseline = " << baseline << endl;

    // FAST detector for per-tile detection (fast)
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(20, true);

    // bucketing params - tune as needed
    const int TILE_H = 20;
    const int TILE_W = 40;
    const int MAX_PER_TILE = 10;

    // KLT params (pyrLK)
    vector<int> lk_win = {21,21}; // window size
    TermCriteria lk_crit(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
    int lk_max_level = 3;
    double lk_max_error = 12.0; // pixel error threshold (tweak)

    // Trajectory visual
    Mat traj = Mat::zeros(800, 800, CV_8UC3);
    double traj_scale = 0.1;
    int origin_x = traj.cols/2, origin_y = traj.rows/4;

    // initialize curr_pose with first GT pose if available, else identity
    Mat curr_pose = Mat::eye(4,4,CV_64F);
    if (!gt_poses.empty()) curr_pose = gt_poses[0].clone();

    // ----- PREPARE FRAME 0 (prev_*) -----
    auto load_gray = [](const string &path)->Mat { return imread(path, IMREAD_GRAYSCALE); };
    char buf[256];
    sprintf(buf, "%06d.png", 0);
    Mat prevL = load_gray(left_dir + buf), prevR = load_gray(right_dir + buf);
    if (prevL.empty() || prevR.empty()) { cerr << "Cannot read frame 0 images." << endl; return -1; }

    // detect bucketed keypoints in left and right
    vector<KeyPoint> prev_kpL = bucketKeypoints(prevL, fast, TILE_H, TILE_W, MAX_PER_TILE);
    vector<KeyPoint> prev_kpR = bucketKeypoints(prevR, fast, TILE_H, TILE_W, MAX_PER_TILE);

    // convert keypoints -> Point2f for LK
    vector<Point2f> prev_ptsL, prev_ptsR;
    KeyPoint::convert(prev_kpL, prev_ptsL);
    KeyPoint::convert(prev_kpR, prev_ptsR);

    // --- stereo matching using KLT: prevL -> prevR (tracks left points to right image) ---
    vector<uchar> status_lr;
    vector<float> err_lr;
    vector<Point2f> prev_ptsL_tracked_to_R;
    if (!prev_ptsL.empty()) {
        calcOpticalFlowPyrLK(prevL, prevR, prev_ptsL, prev_ptsL_tracked_to_R, status_lr, err_lr,
                             Size(lk_win[0], lk_win[1]), lk_max_level, lk_crit, 0, 0.001);
    }

    // compute prev_3d_per_kp (size = prev_kpL.size()), set invalid z<=0
    vector<Point3f> prev_3d_per_kp(prev_kpL.size(), Point3f(0,0,-1));
    for (size_t i = 0; i < prev_ptsL.size(); ++i) {
        if (i >= status_lr.size() || !status_lr[i]) continue;
        if (err_lr[i] > lk_max_error) continue;
        Point2f pl = prev_ptsL[i];
        Point2f pr = prev_ptsL_tracked_to_R[i];
        // ensure in bounds
        if (pr.x < 0 || pr.x >= prevR.cols || pr.y < 0 || pr.y >= prevR.rows) continue;
        float disparity = pl.x - pr.x;
        if (disparity <= 0.0f) continue;
        double Z = fx * baseline / disparity;
        if (!isfinite(Z) || Z <= 0) continue;
        double X = (pl.x - cx) * Z / fx;
        double Y = (pl.y - cy) * Z / fy;
        prev_3d_per_kp[i] = Point3f((float)X, (float)Y, (float)Z);
    }

    // ----- LOOP frames from 1..N-1 -----
    int N = (int)gt_poses.size();
    if (N == 0) N = 500; // fallback if no GT file length available

    for (int frame = 1; frame < N; ++frame) {
        // load current frame
        sprintf(buf, "%06d.png", frame);
        Mat curL = load_gray(left_dir + buf), curR = load_gray(right_dir + buf);
        if (curL.empty() || curR.empty()) break;

        // ---------- Temporal tracking (prev left -> cur left) using KLT ----------
        vector<Point2f> prev_pts_for_tracking;
        KeyPoint::convert(prev_kpL, prev_pts_for_tracking);
        vector<Point2f> tracked_pts; vector<uchar> status_tc; vector<float> err_tc;
        if (!prev_pts_for_tracking.empty()) {
            calcOpticalFlowPyrLK(prevL, curL, prev_pts_for_tracking, tracked_pts, status_tc, err_tc,
                                 Size(lk_win[0], lk_win[1]), lk_max_level, lk_crit, 0, 0.001);
        }

        // build 3D-2D correspondences: 3D from prev_3d_per_kp, 2D = tracked_pts in current left image
        vector<Point3f> objPoints;
        vector<Point2f> imgPoints;
        for (size_t i = 0; i < prev_pts_for_tracking.size(); ++i) {
            if (i >= status_tc.size() || !status_tc[i]) continue;
            if (err_tc[i] > lk_max_error) continue;
            Point3f P = prev_3d_per_kp[i]; // 3D from prev frame
            if (!(P.z > 0 && isfinite(P.z))) continue;
            Point2f p_cur = tracked_pts[i];
            // ensure in image bounds
            if (p_cur.x < 0 || p_cur.x >= curL.cols || p_cur.y < 0 || p_cur.y >= curL.rows) continue;
            objPoints.push_back(P);
            imgPoints.push_back(p_cur);
        }

        // run solvePnP if enough correspondences
        if (objPoints.size() >= 6) {
            Mat rvec, tvec, inliers;
            Mat Kmat = (Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
            bool ok = solvePnPRansac(objPoints, imgPoints, Kmat, noArray(),
                                     rvec, tvec, false, 100, 2.0, 0.99, inliers, SOLVEPNP_ITERATIVE);
            if (ok) {
                Mat R;
                Rodrigues(rvec, R);
                Mat T = Mat::eye(4,4,CV_64F);
                R.convertTo(T(Range(0,3), Range(0,3)), CV_64F);
                T.at<double>(0,3) = tvec.at<double>(0);
                T.at<double>(1,3) = tvec.at<double>(1);
                T.at<double>(2,3) = tvec.at<double>(2);

                // update global pose
                curr_pose = curr_pose * T;
            }
        }

        // ---------- For next iteration: detect new bucketed keypoints in current frame ----------
        vector<KeyPoint> cur_kpL = bucketKeypoints(curL, fast, TILE_H, TILE_W, MAX_PER_TILE);
        vector<KeyPoint> cur_kpR = bucketKeypoints(curR, fast, TILE_H, TILE_W, MAX_PER_TILE);

        // compute cur left->right stereo matches via KLT
        vector<Point2f> cur_ptsL; KeyPoint::convert(cur_kpL, cur_ptsL);
        vector<Point2f> cur_ptsL_tracked_to_R; vector<uchar> status_cur_lr; vector<float> err_cur_lr;
        if (!cur_ptsL.empty()) {
            calcOpticalFlowPyrLK(curL, curR, cur_ptsL, cur_ptsL_tracked_to_R, status_cur_lr, err_cur_lr,
                                 Size(lk_win[0], lk_win[1]), lk_max_level, lk_crit, 0, 0.001);
        }
        // recompute prev_3d_per_kp for the (now) prev frame (the current frame just processed)
        prev_3d_per_kp.assign(cur_kpL.size(), Point3f(0,0,-1));
        for (size_t i = 0; i < cur_ptsL.size(); ++i) {
            if (i >= status_cur_lr.size() || !status_cur_lr[i]) continue;
            if (err_cur_lr[i] > lk_max_error) continue;
            Point2f pl = cur_ptsL[i];
            Point2f pr = cur_ptsL_tracked_to_R[i];
            if (pr.x < 0 || pr.x >= curR.cols || pr.y < 0 || pr.y >= curR.rows) continue;
            float disparity = pl.x - pr.x;
            if (disparity <= 0.0f) continue;
            double Z = fx * baseline / disparity;
            if (!isfinite(Z) || Z <= 0) continue;
            double X = (pl.x - cx) * Z / fx;
            double Y = (pl.y - cy) * Z / fy;
            prev_3d_per_kp[i] = Point3f((float)X, (float)Y, (float)Z);
        }

        // update prev frame references for next iter
        prevL = curL.clone();
        prevR = curR.clone();
        prev_kpL = std::move(cur_kpL);
        prev_kpR = std::move(cur_kpR);

        // draw trajectories
        int px = int(curr_pose.at<double>(0,3) * traj_scale) + origin_x;
        int py = int(-curr_pose.at<double>(2,3) * traj_scale) + origin_y; // display Z inverted if you want
        circle(traj, Point(px, py), 2, Scalar(0,255,0), -1);

        if (!gt_poses.empty() && frame < (int)gt_poses.size()) {
            Mat GT = gt_poses[frame];
            int gx = int(GT.at<double>(0,3) * traj_scale) + origin_x;
            int gy = int(GT.at<double>(2,3) * traj_scale) + origin_y;
            circle(traj, Point(gx, gy), 2, Scalar(0,0,255), -1);
        }

        imshow("Trajectory", traj);
        imshow("Left", curL);
        if (waitKey(1) == 27) break;
    }

    cout << "Finished." << endl;
    return 0;
}
