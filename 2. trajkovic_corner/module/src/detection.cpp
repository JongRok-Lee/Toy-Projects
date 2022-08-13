#include <detection.h>



Corner::Corner(const std::string &path, double T1_, double T2_, std::string method_) : T1(T1_), T2(T2_), IA_method(method_)
{
	src_img = cv::imread(path, cv::IMREAD_UNCHANGED);
	cv::cvtColor(src_img, img, cv::COLOR_BGR2GRAY);
	cv::resize(img, resized_img, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

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
	for (auto i:candidates1) std::cout << "x: " << i[0] << ",\ty: " << i[1] << ",\tR: " << i[2] << "\n";  
	corners = step2(candidates1);
	// for (auto i:corners) std::cout << "x: " << i[0] << ",\ty: " << i[1] << ",\tR: " << i[2] << "\n";
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

std::vector<int> Corner::size()
{	std::vector<int> temp;
	temp.push_back(src.size());
	temp.push_back(src[0].size());
	return temp;
}

std::vector<std::vector<double>> Corner::step1()
{
	return compute_R1();
}

std::vector<std::vector<double>> Corner::step2(std::vector<std::vector<double>> &candidates_)
{
	if (IA_method == "linear") return linear_compute_R2(candidates_);
	if (IA_method == "circular") return circular_compute_R2(candidates_);
}

std::vector<std::vector<double>> Corner::compute_R1()
{
	std::vector<std::vector<double>> coordinates;
	std::vector<double> coordinate;
	double f_C, f_A, f_B, f_A_dot, f_B_dot;
	double rA, rB;
	double R, B, B1, B2, A, A1, A2;
	for (int y = 1; y < resized_src.size()-1; y++) {
		for (int x = 1; x < resized_src[0].size()-1; x++) {
			f_C 	= resized_src[y][x];
			f_A 	= resized_src[y][x+1];
			f_B 	= resized_src[y-1][x];
			f_A_dot = resized_src[y][x-1];
			f_B_dot = resized_src[y+1][x];

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

std::vector<std::vector<double>> Corner::linear_compute_R2(std::vector<std::vector<double>> &candidates_)
{
	std::vector<std::vector<double>> coordinates;
	std::vector<double> coordinate;
	int x, y;
	double f_C, f_A, f_B, f_A_dot, f_B_dot;
	
	double rA, rB, r1, r2;
	double R, B, B1, B2, A, A1, A2;
	for (auto xyr: candidates_) {
		x = 2*xyr[0];
		y = 2*xyr[1];
		
		f_C 	= src[y][x];
		f_A 	= src[y][x+1];
		f_B 	= src[y-1][x];
		f_A_dot = src[y][x-1];
		f_B_dot = src[y+1][x];

		rA = (f_A-f_C)*(f_A-f_C) + (f_A_dot-f_C)*(f_A_dot-f_C);
		rB = (f_B-f_C)*(f_B-f_C) + (f_B_dot-f_C)*(f_B_dot-f_C);
		R = std::min(rA, rB);

		if (R > T2) {
			B1 = (f_B-f_A)*(f_A-f_C) + (f_B_dot-f_A_dot)*(f_A_dot-f_C);
			B2 = (f_B-f_A_dot)*(f_A_dot-f_C) + (f_B_dot-f_A)*(f_A-f_C);
			A1 = rB - rA - 2*B1;
			A2 = rB - rA - 2*B2;

			B = std::min(B1, B2);
			A = rB - rA - 2*B;

			if (B < 0 && (A+B) > 0) {
				R = rA - B*B/A;
			}
			if (R > T2) {
					coordinate.push_back(x);
					coordinate.push_back(y);
					coordinate.push_back(R);
					coordinates.push_back(coordinate);
					coordinate.clear();
			}
			// else {
			// 	R = std::numeric_limits<double>::max();
			// 	for (double linear_x = 0.01; linear_x <1.0; linear_x += 0.01) {
			// 		r1 = A1*linear_x*linear_x + 2*B1*linear_x + rA;
			// 		r2 = A2*linear_x*linear_x + 2*B2*linear_x + rA;

			// 		R = std::min(R, std::min(r1, r2));
			// 	}
			// 	if (R > T2) {
			// 		coordinate.push_back(x);
			// 		coordinate.push_back(y);
			// 		coordinate.push_back(R);
			// 		coordinates.push_back(coordinate);
			// 		coordinate.clear();
			// 	}
			// }
		}
	}
	return coordinates;
}

std::vector<std::vector<double>> Corner::circular_compute_R2(std::vector<std::vector<double>> &candidates_)
{
	std::vector<std::vector<double>> coordinates;
	std::vector<double> coordinate;
	int x, y;
	double f_C, f_A, f_B, f_A_dot, f_B_dot;
	
	double rA, rB, r1, r2;
	double R, B, B1, B2, A;
	for (auto xyr: candidates_) {
		x = 2*xyr[0];
		y = 2*xyr[1];
		
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
			B = (rA + rB) / 2;
			B1 = (f_A-f_C)*(f_B-f_C) + (f_A_dot-f_C)*(f_B_dot-f_C);
			B2 = (f_A_dot-f_C)*(f_B-f_C) + (f_A-f_C)*(f_B_dot-f_C);
			
			if (B < 0) {
				R = rA - std::sqrt(A*A + B*B);
			}
			if (R > T2) {
				coordinate.push_back(x);
				coordinate.push_back(y);
				coordinate.push_back(R);
				coordinates.push_back(coordinate);
				coordinate.clear();
			}
			// else {
			// 	R = std::numeric_limits<double>::max();
			// 	for (double alpha = 0.017; alpha <M_PI/2; alpha += 0.017) {
			// 		r1 = A*std::cos(2*alpha) + B1*std::sin(2*alpha) + rA;
			// 		r2 = A*std::cos(2*alpha) + B2*std::sin(2*alpha) + rA;

			// 		R = std::min(R, std::min(r1, r2));
			// 	}
			// 	if (R > T2) {
			// 		coordinate.push_back(x);
			// 		coordinate.push_back(y);
			// 		coordinate.push_back(R);
			// 		coordinates.push_back(coordinate);
			// 		coordinate.clear();
			// 	}
			// }
		}
	}
	return coordinates;
}