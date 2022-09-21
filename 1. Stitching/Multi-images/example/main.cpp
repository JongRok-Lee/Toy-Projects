#include "extractor.h"

namespace fs = std::experimental::filesystem;

int main()
{   fs::path dir_path("sources/opencv_data");
    if (!fs::exists(dir_path)) {
        throw std::runtime_error("No directory") ;
    }
    
    std::vector<std::string> file_paths = sorted_path(dir_path);
    
    int img_num = file_paths.size();
    int iter_num = static_cast<int>(ceil((static_cast<float>(img_num) - 1) / 2));
    int base_num = img_num / 2;
    std::vector<int> left_imgs, right_imgs;
    for (int i = base_num-1; i >= 0; --i) {
        left_imgs.push_back(i);
    }
    for (int i = base_num+1; i < img_num; ++i) {
        right_imgs.push_back(i);
    }
    
    cv::Mat base_img = cv::imread(file_paths[base_num]), img1, img2, stitchedImage;
    std::cout << file_paths[base_num] << "\n";
    img1 = base_img;

    int cnt = 0;
    for (int i=0; i <iter_num; ++i) {
        if (i <= right_imgs.size() - 1) {
            img2 = cv::imread(file_paths[right_imgs[i]]);
            img1 = StitchImages(img1, img2);
            cnt++;
            cv::imwrite("./result"+std::to_string(cnt)+".jpg", img1);
        }
        if (i <= left_imgs.size() - 1) {
            img2 = cv::imread(file_paths[left_imgs[i]]);
            img1 = StitchImages(img1, img2);
            cnt++;
            cv::imwrite("./result"+std::to_string(cnt)+".jpg", img1);
        }
    }

    return 0;
}