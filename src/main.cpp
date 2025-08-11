#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

// Read KITTI calib.txt (same as your earlier helper)
bool readCalibration(const string& calib_file, Mat& K, double& baseline) {
    ifstream file(calib_file);
    if (!file.is_open()) {
        cerr << "Cannot open calib file: " << calib_file << endl;
        return false;
    }
    string line;
    double P0[12] = {0}, P1[12] = {0};
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
    // intrinsics
    K = (Mat_<double>(3,3) << P0[0], P0[1], P0[2],
                               P0[4], P0[5], P0[6],
                               P0[8], P0[9], P0[10]);
    // baseline estimate (KITTI P0 often has Tx=0, P1 has Tx)
    double fx = P0[0];
    double tx0 = P0[3];
    double tx1 = P1[3];
    baseline = (tx0 - tx1) / fx; // baseline positive
    return true;
}

// Read KITTI ground truth poses
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
    return !poses.empty();
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

    // load calibration & GT
    Mat K; double baseline;
    if (!readCalibration(calib_path, K, baseline)) return -1;
    vector<Mat> gt_poses;
    if (!readGroundTruth(gt_path, gt_poses)) {
        cerr << "No GT loaded; still running but no GT visualization." << endl;
    }

    double fx = K.at<double>(0,0), fy = K.at<double>(1,1), cx = K.at<double>(0,2), cy = K.at<double>(1,2);
    cout << "K:\n" << K << "\nbaseline = " << baseline << endl;

    // ORB + matcher
    Ptr<ORB> orb = ORB::create(5000);
    BFMatcher matcher(NORM_HAMMING, true);

    // Trajectory visual
    Mat traj = Mat::zeros(800, 800, CV_8UC3);
    double traj_scale = 0.5;
    int origin_x = traj.cols/2, origin_y = traj.rows/4;

    // initialize curr_pose with first GT pose if available, else identity
    Mat curr_pose = Mat::eye(4,4,CV_64F);
    if (!gt_poses.empty()) curr_pose = gt_poses[0].clone();

    // ----- PREPARE FRAME 0 (prev_*) -----
    auto load_gray = [](const string &path)->Mat { return imread(path, IMREAD_GRAYSCALE); };
    char buf[256];
    sprintf(buf, "%06d.png", 0);
    Mat prevL = load_gray(left_dir + buf), prevR = load_gray(right_dir + buf);
    if (prevL.empty() || prevR.empty()) {
        cerr << "Cannot read frame 0 images." << endl;
        return -1;
    }

    vector<KeyPoint> prev_kpL, prev_kpR;
    Mat prev_descL, prev_descR;
    orb->detectAndCompute(prevL, noArray(), prev_kpL, prev_descL);
    orb->detectAndCompute(prevR, noArray(), prev_kpR, prev_descR);

    // left->right matches for frame 0 to compute per-left-keypoint 3D
    vector<DMatch> matches_lr0;
    if (!prev_descL.empty() && !prev_descR.empty())
        matcher.match(prev_descL, prev_descR, matches_lr0);

    // compute prev_3d_per_kp (size = prev_kpL.size()), set z<=0 as invalid
    vector<Point3f> prev_3d_per_kp(prev_kpL.size(), Point3f(0,0,-1));
    vector<Point2f> ptsL_lr, ptsR_lr;
    for (auto &m : matches_lr0) {
        int iL = m.queryIdx, iR = m.trainIdx;
        float xl = prev_kpL[iL].pt.x, yl = prev_kpL[iL].pt.y;
        float xr = prev_kpR[iR].pt.x;
        float disparity = xl - xr;
        if (disparity <= 0) continue;
        double Z = fx * baseline / disparity;
        if (!isfinite(Z) || Z <= 0) continue;
        double X = (xl - cx) * Z / fx;
        double Y = (yl - cy) * Z / fy;
        prev_3d_per_kp[iL] = Point3f((float)X, (float)Y, (float)Z);
    }

    // ----- LOOP frames from 1..N-1 -----
    int N = (int)gt_poses.size();
    if (N == 0) N = 500; // fallback if no GT file length available

    for (int frame = 1; frame < N; ++frame) {
        // load current frame
        sprintf(buf, "%06d.png", frame);
        Mat curL = load_gray(left_dir + buf), curR = load_gray(right_dir + buf);
        if (curL.empty() || curR.empty()) break;

        // detect descriptors current
        vector<KeyPoint> cur_kpL, cur_kpR;
        Mat cur_descL, cur_descR;
        orb->detectAndCompute(curL, noArray(), cur_kpL, cur_descL);
        orb->detectAndCompute(curR, noArray(), cur_kpR, cur_descR);

        // match previous-left -> current-left (temporal)
        vector<DMatch> matches_prev_curr;
        if (!prev_descL.empty() && !cur_descL.empty())
            matcher.match(prev_descL, cur_descL, matches_prev_curr);

        // build 3D-2D correspondences: 3D from prev_3d_per_kp, 2D = cur keypoints
        vector<Point3f> objPoints;
        vector<Point2f> imgPoints;
        objPoints.reserve(matches_prev_curr.size());
        imgPoints.reserve(matches_prev_curr.size());
        for (auto &m : matches_prev_curr) {
            int idxPrev = m.queryIdx;
            int idxCurr = m.trainIdx;
            if (idxPrev < (int)prev_3d_per_kp.size()) {
                Point3f P = prev_3d_per_kp[idxPrev];
                if (P.z > 0 && isfinite(P.z)) {
                    objPoints.push_back(P);
                    imgPoints.push_back(cur_kpL[idxCurr].pt);
                }
            }
        }

        // run solvePnP if enough correspondences
        if (objPoints.size() >= 6) {
            Mat rvec, tvec, inliers;
            Mat Kmat = (Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
            bool ok = solvePnPRansac(objPoints, imgPoints, Kmat, noArray(),
                                     rvec, tvec, false, 100, 8.0, 0.99, inliers, SOLVEPNP_ITERATIVE);
            if (ok) {
                Mat R;
                Rodrigues(rvec, R);
                Mat T = Mat::eye(4,4,CV_64F);
                R.convertTo(T(Range(0,3), Range(0,3)), CV_64F);
                T.at<double>(0,3) = tvec.at<double>(0);
                T.at<double>(1,3) = tvec.at<double>(1);
                T.at<double>(2,3) = tvec.at<double>(2);

                // T maps points from prev -> cur (X_cur = R*X_prev + t)
                // update global pose: curr_pose = curr_pose * T
                curr_pose = curr_pose * T;
            } else {
                // solvePnP failed; skip update
            }
        } else {
            // not enough correspondences
        }

        // draw trajectories
        int px = int(curr_pose.at<double>(0,3) * traj_scale) + origin_x;
        int py = int(curr_pose.at<double>(2,3) * traj_scale * -1) + origin_y;
        circle(traj, Point(px, py), 2, Scalar(0,0,255), -1);

        if (!gt_poses.empty() && frame < (int)gt_poses.size()) {
            Mat GT = gt_poses[frame];
            int gx = int(GT.at<double>(0,3) * traj_scale) + origin_x;
            int gy = int(GT.at<double>(2,3) * traj_scale) + origin_y;
            circle(traj, Point(gx, gy), 2, Scalar(255,0,0), -1);
        }

        imshow("Trajectory", traj);
        imshow("Left", curL);
        if (waitKey(1) == 27) break;

        // prepare prev_* for next iteration:
        prevL = curL.clone();
        prevR = curR.clone();
        prev_kpL = cur_kpL;
        prev_kpR = cur_kpR;
        prev_descL = cur_descL.clone();
        prev_descR = cur_descR.clone();

        // recompute prev_3d_per_kp for the (now) prev frame (i.e. the current frame we just processed)
        // use left<->right matches of this frame
        vector<DMatch> matches_lr;
        if (!prev_descL.empty() && !prev_descR.empty()) matcher.match(prev_descL, prev_descR, matches_lr);
        prev_3d_per_kp.assign(prev_kpL.size(), Point3f(0,0,-1));
        for (auto &m : matches_lr) {
            int iL = m.queryIdx, iR = m.trainIdx;
            float xl = prev_kpL[iL].pt.x, yl = prev_kpL[iL].pt.y;
            float xr = prev_kpR[iR].pt.x;
            float disparity = xl - xr;
            if (disparity <= 0) continue;
            double Z = fx * baseline / disparity;
            if (!isfinite(Z) || Z <= 0) continue;
            double X = (xl - cx) * Z / fx;
            double Y = (yl - cy) * Z / fy;
            prev_3d_per_kp[iL] = Point3f((float)X, (float)Y, (float)Z);
        }
    }

    cout << "Finished." << endl;
    return 0;
}
