#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

class Corner
{
    public:
        Corner(const std::string &path, double T1_, double T2_, std::string method_);
        ~Corner();
    public:
        const std::string IA_method;
        const double T1, T2;
        cv::Mat img, src_img, resized_img;
        std::vector<std::vector<uchar>> src;
        std::vector<std::vector<uchar>> resized_src;
        std::vector<std::vector<double>> corners;
    public:
        void show();
        std::vector<int> size();
        std::vector<std::vector<double>> step1();
        std::vector<std::vector<double>> step2(std::vector<std::vector<double>>& candidates_);
        std::vector<std::vector<double>> compute_R1();
        std::vector<std::vector<double>> linear_compute_R2(std::vector<std::vector<double>> &candidates_);
        std::vector<std::vector<double>> circular_compute_R2(std::vector<std::vector<double>> &candidates_);

};