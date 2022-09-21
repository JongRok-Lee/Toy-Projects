#include "extractor.h"

namespace fs = std::experimental::filesystem;

int main()
{   fs::path dir_path("sources/opencv_data");
    if (!fs::exists(dir_path)) {
        throw std::runtime_error("No directory") ;
    }
    
    std::vector<std::string> file_paths = sorted_path(dir_path);
    std::vector<cv::Mat> imgs = set_imgs(file_paths);

    cv::Mat pano;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
    cv::Stitcher::Status status = stitcher->stitch(imgs, pano);

    if (status != cv::Stitcher::OK) {
        std::cout << "No Run, Error code: " << int(status) << "\n";
    }

    cv::imwrite("sources/result.png", pano);
    return 0;
}