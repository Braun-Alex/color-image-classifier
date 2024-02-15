#pragma once

#include "utils.h"

const int COLOR_CHANNEL_COUNT = 3;

namespace moments {
    std::vector<cv::Scalar> calculateMoments(const cv::Mat &image);

    bool areDuplicates(const std::vector<cv::Scalar> &firstImageMoments,
                       const std::vector<cv::Scalar> &secondImageMoments,
                       int threshold);
}