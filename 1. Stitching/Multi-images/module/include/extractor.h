#include "opencv2/opencv.hpp"
#include "opencv2/stitching.hpp"
#include <set>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <experimental/filesystem>

std::vector<std::string> sorted_path(std::experimental::filesystem::path &path);
std::vector<cv::Mat> set_imgs(std::vector<std::string> &file_paths);

void FindMatches(cv::Mat &BaseImage, cv::Mat &SecImage, std::vector<cv::DMatch>& GoodMatches, std::vector<cv::KeyPoint>& BaseImage_kp, std::vector<cv::KeyPoint>& SecImage_kp);
void FindHomography(std::vector<cv::DMatch> &Matches, std::vector<cv::KeyPoint> &BaseImage_kp, std::vector<cv::KeyPoint> &SecImage_kp, cv::Mat& HomographyMatrix);
void GetNewFrameSizeAndMatrix(cv::Mat &HomographyMatrix, int* Sec_ImageShape, int* Base_ImageShape, int* NewFrameSize, int* Correction);
cv::Mat StitchImages(cv::Mat &BaseImage, cv::Mat &SecImage);
cv::Mat get_mask(cv::Mat img0, cv::Mat img1);