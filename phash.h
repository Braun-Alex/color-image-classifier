#pragma once

#include "utils.h"
#include <opencv2/img_hash.hpp>

bool areDuplicatesUsingPHashes(const cv::Mat& firstImage, const cv::Mat& secondImage, double threshold);
double calculateHammingDistance(const cv::Mat& firstImage, const cv::Mat& secondImage);