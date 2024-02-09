#include "phash.h"

bool areDuplicatesUsingPHashes(const cv::Mat& firstImage, const cv::Mat& secondImage, double threshold) {
    cv::Mat firstImageHash, secondImageHash;
    cv::img_hash::pHash(firstImage, firstImageHash);
    cv::img_hash::pHash(secondImage, secondImageHash);

    return cv::norm(firstImageHash, secondImageHash, cv::NORM_HAMMING) <= threshold;
}

double calculateHammingDistance(const cv::Mat& firstImage, const cv::Mat& secondImage) {
    cv::Mat firstImageHash, secondImageHash;
    cv::img_hash::pHash(firstImage, firstImageHash);
    cv::img_hash::pHash(secondImage, secondImageHash);

    return cv::norm(firstImageHash, secondImageHash, cv::NORM_HAMMING);
}