#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include <random>
#include <string>
#include <experimental/filesystem>

cv::Mat H_transform(cv::Mat& moveImg, cv::Mat& H);
cv::Mat get_H(std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ptsTM);
std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> get_pts(cv::Mat& targetImg, cv::Mat& moveImg);
cv::Mat DLT_H(std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ptsTM, int& max_num, double& resThr);
cv::Mat RANSAC(std::vector<cv::Point2d> ptsT, std::vector<cv::Point2d> ptsM, int& max_iter, double& resThr);
std::vector<int> get_sampleIdx(std::uniform_int_distribution<int>& uid, std::mt19937& gen);
cv::Mat get_Hk(std::vector<cv::Point2d> ptsT, std::vector<cv::Point2d> ptsM, std::vector<int> index);
int countInlier(std::vector<cv::Point2d> ptsT, std::vector<cv::Point2d> ptsM, cv::Mat& Hk, double& resThr);
