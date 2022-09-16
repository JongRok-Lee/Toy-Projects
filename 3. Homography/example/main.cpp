#include "module.h"

namespace fs = std::experimental::filesystem;

int main(int argc, char** argv)
{   
    cv::Mat targetImg = cv::imread("sources/01.jpg", cv::IMREAD_ANYCOLOR);
    cv::Mat moveImg = cv::imread("sources/02.jpg", cv::IMREAD_ANYCOLOR);

    int max_iter = atoi(argv[1]);
    double resThr = atof(argv[2]);
    std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ptsTM = get_pts(targetImg, moveImg);
    cv::Mat H1 = get_H(ptsTM);
    cv::Mat H2 = DLT_H(ptsTM, max_iter, resThr);
    cv::Mat dst1 = H_transform(moveImg, H1);
    cv::Mat dst2 = H_transform(moveImg, H2);

    std::cout<< "Homography (DLT + RANSAC): " << std::endl << H2 << std::endl;
    std::cout<< "Homography (OpenCV): " << std::endl << H1 << std::endl;

    cv::imshow("targetImg", targetImg);
    cv::imshow("moveImg", moveImg);
    cv::imshow("OpenCV", dst1);
    cv::imshow("JongRok", dst2);
    cv::waitKey(0);
    return 0;
}