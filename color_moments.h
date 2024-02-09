#pragma once

#include "utils.h"

const int COLOR_CHANNEL_COUNT = 3;

std::vector<cv::Scalar> calculateMoments(const cv::Mat& image);
bool areDuplicatesUsingMoments(const std::vector<cv::Scalar>& firstImageMoments,
                   const std::vector<cv::Scalar>& secondImageMoments,
                   double threshold);