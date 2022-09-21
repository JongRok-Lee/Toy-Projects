//
// Created by jr
//
#include "module1/Class.hpp"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

int main()
{
  cv::Mat _src1 = cv::imread("sources/img1.jpg", cv::IMREAD_COLOR);
  cv::Mat _src2 = cv::imread("sources/img2.jpg", cv::IMREAD_COLOR);

  cv::Mat src1, src2;
  cv::cvtColor(_src1, src1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(_src2, src2, cv::COLOR_BGR2GRAY);

  cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();
  cv::Mat desc1, desc2;
  std::vector<cv::KeyPoint> keypoints1, keypoints2;

  detector->detectAndCompute(src1, cv::Mat(), keypoints1, desc1);
  detector->detectAndCompute(src2, cv::Mat(), keypoints2, desc2);

  desc1.convertTo(desc1, CV_32F);
  desc2.convertTo(desc2, CV_32F);

  cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
  std::vector<cv::DMatch> matches;
  matcher->match(desc1, desc2, matches);

  std::sort(matches.begin(), matches.end());
  std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + 20);

  std::vector<cv::Point2f> pts1, pts2;
  for (size_t i = 0; i < good_matches.size(); i++) {
    pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
    pts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
  }
  
  cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC);
  cv::Mat src_2;
  cv::warpPerspective(_src2, src_2, H, cv::Size(src1.cols*1.8, src1.rows*1.5), cv::INTER_CUBIC);

  cv::Mat roi(src_2,cv::Rect(0, 0, src1.cols, src1.rows));
  _src1.copyTo(roi);

  cv::Mat dst;
  cv::drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
                  cv::INTER_CUBIC, cv::Scalar::all(-1), cv::Scalar::all(-1),
                  std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  while (cv::waitKey(0) != 27)
  {
    cv::imshow("match", dst);
    cv::imshow("src_2", src_2);
  }

  return 0;
}
