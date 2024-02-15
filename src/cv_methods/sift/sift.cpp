#include "sift.h"

void sift::calculateMatches(const cv::Mat& firstImage, const cv::Mat& secondImage, std::vector<std::vector<cv::DMatch>>& matches,
                      const cv::Ptr<cv::SIFT>& siftInstance, const cv::Ptr<cv::FlannBasedMatcher>& matcher) {
    std::vector<cv::KeyPoint> firstImKp, secondImKp;
    cv::Mat firstImDesc, secondImDesc;

    siftInstance->detectAndCompute(firstImage, cv::Mat(), firstImKp, firstImDesc);
    siftInstance->detectAndCompute(secondImage, cv::Mat(), secondImKp, secondImDesc);

    matcher->knnMatch(firstImDesc, secondImDesc, matches, 2);

    matches.erase(std::remove_if(matches.begin(), matches.end(), [](const std::vector<cv::DMatch>& matchPair) {
        return matchPair[0].distance >= RATIO_TEST_FILTER * matchPair[1].distance;
    }), matches.end());
}

bool sift::areDuplicates(const std::vector<std::vector<cv::DMatch>>& matches, int threshold) {
    return matches.size() > threshold;
}