#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <curl/curl.h>
#include <opencv2/opencv.hpp>

std::string fileToString(const std::string& jsonFileName);
size_t writeCallback(void* data, size_t size, size_t nMemBytes, std::vector<unsigned char>* buffer);
cv::Mat downloadImage(const std::string& url, cv::ImreadModes mode);
void saveFile(const std::string& fileName, const char* data);
void repeatEntering(const std::string& reason);
void resizeImage(const cv::Mat& srcImage, cv::Mat& dstImage, int maxPixels);