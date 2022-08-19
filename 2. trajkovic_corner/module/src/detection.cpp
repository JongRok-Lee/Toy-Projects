#include <detection.h>



Corner::Corner(const std::string &path, double resize_param_, double T1_, double T2_, std::string method_) : resize_param(resize_param_),T1(T1_), T2(T2_), IA_method(method_)
{
	src_img = cv::imread(path, cv::IMREAD_UNCHANGED);
	cv::Mat img_;
	cv::cvtColor(src_img, img_, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(img_, img, cv::Size(3,3), 0);
	cv::resize(img, resized_img, cv::Size(), resize_param, resize_param, cv::INTER_AREA);

	std::vector<uchar> temp;
	for (int y = 0; y < img.rows; ++y){
		for (int x = 0; x < img.cols; ++x){
			temp.push_back(img.at<uchar>(y, x));
		}
		src.push_back(temp);
		temp.clear();
	}

	for (int y = 0; y < resized_img.rows; ++y){
		for (int x = 0; x < resized_img.cols; ++x){
			temp.push_back(resized_img.at<uchar>(y, x));
		}
		resized_src.push_back(temp);
		temp.clear();
	}

	std::vector<std::vector<double>> candidates1 = step1();
	// for (auto i:candidates1) std::cout << "x: " << i[0] << ",\ty: " << i[1] << ",\tR: " << i[2] << "\n";  
	corners = step2(candidates1);

	// for (auto i:corners) {
	// 	std::cout << "x: " << i[0] << ",\ty: " << i[1] << ",\tR: " << i[2] << "\n";
	// }
	std::cout << "Step 1. Number of corner candidates : "<< candidates1.size() <<"\n";
	std::cout << "Step 2. Number of corner candidates : "<< corners.size() <<"\n";
}
Corner::~Corner() {}

void Corner::show()
{	
	std::vector<cv::KeyPoint> kp;
	cv::Mat fast_img = src_img.clone();

    cv::FAST(img, kp, 60, false);
    for (auto k : kp) {
        cv::circle(fast_img, cv::Point(cvRound(k.pt.x), cvRound(k.pt.y)), 5, cv::Scalar(0, 255, 0), -1);
    }
	for (auto i : corners) {
		cv::circle(src_img, cv::Point(static_cast<int>(i[0]) , static_cast<int>(i[1])), 5, cv::Scalar(0, 255, 0), -1);
	}
	cv::Mat show;
	cv::hconcat(fast_img, src_img, show);
	cv::namedWindow("show", cv::WINDOW_NORMAL);
	cv::imshow("show", show);
	while(cv::waitKey(0) != 27) {
    	continue;
	}
}


std::vector<std::vector<double>> Corner::step2(std::vector<std::vector<double>> &candidates_)
{
	if (IA_method == "linear") return linear_compute_R(candidates_);
	if (IA_method == "circular") return circular_compute_R(candidates_);
}

std::vector<std::vector<double>> Corner::step1()
{
	std::vector<std::vector<double>> coordinates;
	std::vector<double> coordinate;
	double f_C, f_A, f_B, f_A_dot, f_B_dot;
	double rA, rB, R;
	for (int y = 1; y < resized_src.size()-1; y++) {
		for (int x = 1; x < resized_src[0].size()-1; x++) {
			f_C 	= static_cast<double>(resized_src[y][x]);
			f_A 	= static_cast<double>(resized_src[y][x+1]);
			f_B 	= static_cast<double>(resized_src[y-1][x]);
			f_A_dot = static_cast<double>(resized_src[y][x-1]);
			f_B_dot = static_cast<double>(resized_src[y+1][x]);

			rA = (f_A-f_C)*(f_A-f_C) + (f_A_dot-f_C)*(f_A_dot-f_C);
			rB = (f_B-f_C)*(f_B-f_C) + (f_B_dot-f_C)*(f_B_dot-f_C);
			R = std::min(rA, rB);
			
			if (R > T1) {
				coordinate.push_back(x);
				coordinate.push_back(y);
				coordinate.push_back(R);
				coordinates.push_back(coordinate);
				coordinate.clear();
			}

		}
	}
	return coordinates;
}

std::vector<std::vector<double>> Corner::linear_compute_R(std::vector<std::vector<double>> &candidates_)
{
	std::vector<std::vector<double>> coordinates;
	std::vector<double> coordinate;
	int x, y;
	double f_C, f_A, f_B, f_A_dot, f_B_dot;
	
	double rA, rB, r1, r2;
	double R, B, B1, B2, A, A1, A2, C;
	for (auto xyr: candidates_) {
		x = static_cast<int>(1/resize_param*xyr[0]);
		y = static_cast<int>(1/resize_param*xyr[1]);
		
		f_C 	= static_cast<double>(src[y][x]);
		f_A 	= static_cast<double>(src[y][x+1]);
		f_B 	= static_cast<double>(src[y-1][x]);
		f_A_dot = static_cast<double>(src[y][x-1]);
		f_B_dot = static_cast<double>(src[y+1][x]);

		rA = (f_A-f_C)*(f_A-f_C) + (f_A_dot-f_C)*(f_A_dot-f_C);
		rB = (f_B-f_C)*(f_B-f_C) + (f_B_dot-f_C)*(f_B_dot-f_C);
		R = std::min(rA, rB);

		if (R > T2) {
			B1 = (f_B-f_A)*(f_A-f_C) + (f_B_dot-f_A_dot)*(f_A_dot-f_C);
			B2 = (f_B-f_A_dot)*(f_A_dot-f_C) + (f_B_dot-f_A)*(f_A-f_C);
			C = rA;
			B = std::min(B1, B2);
			A = rB - rA - 2*B;

			if (B < 0.0 && A+B > 0.0) {
				if (A != 0.0) {
					R = C - B*B/A;
					// std::cout << "con1) A: " << A << "\tB: " << B << "\tR: " << R << "\n";
				}
				else {
					R = B + C;
					// std::cout << "A: " << A << "\tB: " << B << "\tR: " << R << "\n";
				}
			}
			else {
				R = R;
				// std::cout << "con2) A: " << A << "\tB: " << B << "\tR: " << R << "\n";
			}
			
			if (R > T2) {
				coordinate.push_back(x);
				coordinate.push_back(y);
				coordinate.push_back(R);
				coordinates.push_back(coordinate);
				coordinate.clear();
			}
		}
	}
	return coordinates;
}

std::vector<std::vector<double>> Corner::circular_compute_R(std::vector<std::vector<double>> &candidates_)
{
	std::vector<std::vector<double>> coordinates;
	std::vector<double> coordinate;
	int x, y;
	double f_C, f_A, f_B, f_A_dot, f_B_dot;
	
	double rA, rB, r1, r2;
	double R, B, B1, B2, A, C;
	for (auto xyr: candidates_) {
		x = static_cast<int>(1/resize_param*xyr[0]);
		y = static_cast<int>(1/resize_param*xyr[1]);
		
		f_C 	= src[y][x];
		f_A 	= src[y][x+1];
		f_B 	= src[y-1][x];
		f_A_dot = src[y][x-1];
		f_B_dot = src[y+1][x];

		rA = (f_A-f_C)*(f_A-f_C) + (f_A_dot-f_C)*(f_A_dot-f_C);
		rB = (f_B-f_C)*(f_B-f_C) + (f_B_dot-f_C)*(f_B_dot-f_C);
		R = std::min(rA, rB);

		if (R > T2) {
			A = (rA - rB) / 2;
			C = (rA + rB) / 2;
			B1 = (f_A-f_C)*(f_B-f_C) + (f_A_dot-f_C)*(f_B_dot-f_C);
			B2 = (f_A_dot-f_C)*(f_B-f_C) + (f_A-f_C)*(f_B_dot-f_C);
			B = std::min(B1, B2);

			if (B < 0.0) {
				R = C - std::sqrt(A*A + B*B);
			}
			if (R > T2) {
				coordinate.push_back(x);
				coordinate.push_back(y);
				coordinate.push_back(R);
				coordinates.push_back(coordinate);
				coordinate.clear();
			}
		}
	}
	return coordinates;
}