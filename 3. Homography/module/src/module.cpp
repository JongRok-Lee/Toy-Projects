#include "module.h"

cv::Mat RANSAC(std::vector<cv::Point2d> ptsT, std::vector<cv::Point2d> ptsM, int& max_iter, double& resThr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> uid(0, ptsT.size() - 1);
    int cnt = 0, inlierNum, maxInlierNum = 0;
    std::vector<int> index;
    cv::Mat Hk, H;
    std::cout << "Total num: " << ptsM.size() << std::endl;
    while (cnt <= max_iter) {
        index = get_sampleIdx(uid, gen);
        Hk = get_Hk(ptsT, ptsM, index);
        inlierNum = countInlier(ptsT, ptsM, Hk, resThr);
        maxInlierNum = std::max(maxInlierNum, inlierNum);
        if (maxInlierNum == inlierNum) {
            H = Hk;
            std::cout << "i: " << cnt << "\t" << "Num: " <<  inlierNum << std::endl;
            if (maxInlierNum >= static_cast<int>(ptsT.size()* 0.95)) {
                std::cout << "Early Stop!" << std::endl;
                return H;
            }
        }
        cnt++;
    }
    std::cout << "Max iteration!" << std::endl;
    return H;

}

int countInlier(std::vector<cv::Point2d> ptsT, std::vector<cv::Point2d> ptsM, cv::Mat& Hk, double& resThr) {
    Eigen::Vector3d ptsMe, ptsTe, M2Te, errorV;
    Eigen::MatrixXd He;
    double residual;
    int inlierCnt = 0;
    cv::cv2eigen(Hk, He);
    for (int i = 0; i < ptsT.size(); ++i) {
        ptsMe << ptsM[i].x, ptsM[i].y, 1;
        ptsTe << ptsT[i].x, ptsT[i].y, 1;
        M2Te = He * ptsMe;
        M2Te = M2Te / M2Te(2);

        errorV = ptsTe - M2Te;

        residual = std::sqrt(errorV(0)*errorV(0) + errorV(1)*errorV(0));
        if (residual < resThr) {
            inlierCnt++;
        }


    }
    return inlierCnt;
}

cv::Mat get_Hk(std::vector<cv::Point2d> ptsT, std::vector<cv::Point2d> ptsM, std::vector<int> index) {
    Eigen::MatrixXd B(8, 9); 
    Eigen::Matrix<double, 1, 9> temp1, temp2;
    double xM, yM, xT, yT;

    for (int i = 0; i < 4; ++i) {
        xM = ptsM[index[i]].x;
        yM = ptsM[index[i]].y;
        xT = ptsT[index[i]].x;
        yT = ptsT[index[i]].y;

        temp1 << xM, yM, 1, 0, 0, 0, -xT*xM, -xT*yM, -xT;
        temp2 << 0, 0, 0, xM, yM, 1, -yT*xM, -yT*yM, -yT;
        B.row(2*i) = temp1;
        B.row(2*i + 1) = temp2;

    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd p = V.col(V.cols()-1);
    Eigen::Matrix3d H = Eigen::MatrixXd::Identity(3, 3);
    H.block<1, 3>(0, 0) = p.block<3, 1>(0, 0);
    H.block<1, 3>(1, 0) = p.block<3, 1>(3, 0);
    H.block<1, 3>(2, 0) = p.block<3, 1>(6, 0);

    H = H / H(2, 2);
    cv::Mat Hcv;
    cv::eigen2cv(H, Hcv);
    return Hcv;
}

cv::Mat DLT_H(std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ptsTM, int& max_iter, double& resThr) {
    std::vector<cv::Point2d> ptsT = ptsTM.first;
    std::vector<cv::Point2d> ptsM = ptsTM.second;
    cv::Mat H = RANSAC(ptsT, ptsM, max_iter, resThr);
    
    return H;
}

std::vector<int> get_sampleIdx(std::uniform_int_distribution<int>& uid, std::mt19937& gen) {
    int samplingNum = 0, randIdx = 0, beforeIdx = - 1;
    std::vector<int> index;

    while (samplingNum != 4) {
        randIdx = uid(gen);
        if (randIdx == beforeIdx) {
            continue;
        }
        else {
            index.push_back(randIdx);
            beforeIdx = randIdx;
            samplingNum ++;
        }
    }
    return index;
}

std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> get_pts(cv::Mat& targetImg, cv::Mat& moveImg) {
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    cv::Mat descT, descM;
    std::vector<cv::KeyPoint> kpT, kpM;
    detector ->detectAndCompute(targetImg, cv::Mat(), kpT, descT);
    detector ->detectAndCompute(moveImg, cv::Mat(), kpM, descM);

    cv::Ptr<cv::DescriptorMatcher> mathcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> goodMatches;
    mathcher ->knnMatch(descT, descM, matches, 2);
    for (int i = 0; i < matches.size(); ++i)
	{
		if (matches[i][0].distance < 0.75 * matches[i][1].distance)
		{
			goodMatches.push_back(matches[i][0]);
		}
	}

    std::vector<cv::Point2d> ptsT, ptsM;
	for (int i = 0 ; i < goodMatches.size() ; i++)
	{
		ptsT.push_back(kpT[goodMatches[i].queryIdx].pt);
		ptsM.push_back(kpM[goodMatches[i].trainIdx].pt);
	}
    std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ptsTM;
    ptsTM.first = ptsT;
    ptsTM.second = ptsM;
    return ptsTM;

}

cv::Mat get_H(std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> ptsTM) {
    std::vector<cv::Point2d> ptsT = ptsTM.first;
    std::vector<cv::Point2d> ptsM = ptsTM.second;
    cv::Mat H = cv::findHomography(ptsM, ptsT, cv::RANSAC);
    
    return H;
}

cv::Mat H_transform(cv::Mat& moveImg, cv::Mat& H) {
    
    cv::Mat dst;
    cv::warpPerspective(moveImg, dst, H, cv::Size());
    return dst;
}