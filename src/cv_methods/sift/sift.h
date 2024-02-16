#pragma once

#include "utils.h"

const double RATIO_TEST_FILTER = 0.7;

namespace sift {
    int calculateMatches(const cv::Mat& firstImage, const cv::Mat& secondImage, std::vector<std::vector<cv::DMatch>>& matches,
                         const cv::Ptr<cv::SIFT>& siftInstance, const cv::Ptr<cv::FlannBasedMatcher>& matcher);

    bool areDuplicates(int matchSize, int minKpSize, int threshold);
}