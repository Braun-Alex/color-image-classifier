#pragma once

#include "utils.h"

const double RATIO_TEST_FILTER = 0.6;

namespace sift {
    void calculateMatches(const cv::Mat& firstImage, const cv::Mat& secondImage, std::vector<std::vector<cv::DMatch>>& matches,
                          const cv::Ptr<cv::SIFT>& siftInstance, const cv::Ptr<cv::FlannBasedMatcher>& matcher);

    bool areDuplicates(const std::vector<std::vector<cv::DMatch>>& matches, int threshold);
}