#include "module.h"

int main(int argc, char** argv)
{   
    cv::Mat targetImg = cv::imread("sources/01.jpg", cv::IMREAD_ANYCOLOR);
    cv::Mat moveImg = cv::imread("sources/02.jpg", cv::IMREAD_ANYCOLOR);

    const int max_iter = atoi(argv[1]);
    const double resThr = atof(argv[2]);
    std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ptsTM = get_pts(targetImg, moveImg);
    cv::Mat H1 = get_H(ptsTM);
    std::pair<cv::Mat, cv::Mat> Hs = DLT_H(ptsTM, max_iter, resThr);
    cv::Mat H2 = Hs.first, H3 = Hs.second;
    cv::Mat dst1 = H_transform(moveImg, H1);
    cv::Mat dst2 = H_transform(moveImg, H2);
    cv::Mat dst3 = H_transform(moveImg, H3);

    std::cout<< "Homography (DLT + RANSAC): " << std::endl << H2 << std::endl;
    std::cout<< "Homography (DLT + PROSAC): " << std::endl << H3 << std::endl;
    std::cout<< "Homography (OpenCV): " << std::endl << H1 << std::endl;
    

    cv::imshow("targetImg", targetImg);
    cv::imshow("moveImg", moveImg);
    cv::imshow("OpenCV", dst1);
    cv::imshow("RANSAC", dst2);
    cv::imshow("PROSAC", dst3);
    cv::waitKey(0);
    return 0;
}
