#include "color_moments.h"

std::vector<cv::Scalar> calculateMoments(const cv::Mat& image) {
    std::vector<cv::Scalar> imageMoments(COLOR_CHANNEL_COUNT);

    cv::Scalar mean, stdDev;
    cv::meanStdDev(image, mean, stdDev);

    imageMoments[0] = mean;
    imageMoments[1] = stdDev;

    cv::Mat tmpMat;
    for (auto i = 0; i < COLOR_CHANNEL_COUNT; ++i) {
        cv::pow(image - mean[i], 3, tmpMat);
        cv::Scalar skewness = cv::mean(tmpMat);
        imageMoments[2][i] = skewness[i];
    }

    return imageMoments;
}

bool areDuplicatesUsingMoments(const std::vector<cv::Scalar>& firstImageMoments,
                   const std::vector<cv::Scalar>& secondImageMoments,
                   double threshold) {
    for (int i = 0; i < COLOR_CHANNEL_COUNT; ++i) {
        for (int j = 0; j < COLOR_CHANNEL_COUNT; ++j) {
            if (cv::abs(firstImageMoments[i][j] - secondImageMoments[i][j]) > threshold) {
                return false;
            }
        }
    }

    return true;
}