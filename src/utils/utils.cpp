#include "utils.h"

std::string fileToString(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file) {
        throw std::runtime_error("File \"" + fileName + "\" could not be opened!");
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    return buffer.str();
}

size_t writeCallback(void* data, size_t size, size_t nMemBytes, std::vector<unsigned char>* buffer) {
    const size_t totalSize = size * nMemBytes;
    const auto* castedData = static_cast<unsigned char*>(data);
    const size_t currentSize = buffer->size();
    buffer->resize(buffer->size() + totalSize);
    std::copy(castedData, castedData + totalSize, buffer->begin() + currentSize);
    return totalSize;
}

cv::Mat downloadImage(const std::string& url) {
    CURL* curlInstance;
    CURLcode result;
    std::vector<unsigned char> buffer;

    curl_global_init(CURL_GLOBAL_ALL);
    curlInstance = curl_easy_init();

    if (curlInstance) {
        curl_easy_setopt(curlInstance, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curlInstance, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curlInstance, CURLOPT_WRITEDATA, &buffer);

        result = curl_easy_perform(curlInstance);
        curl_easy_cleanup(curlInstance);

        if (result != CURLE_OK) {
            throw std::runtime_error("Image \"" + url + "\" could not be loaded!");
        }
        curl_global_cleanup();

        if (buffer.empty()) {
            throw std::runtime_error("Image \"" + url + "\" contains no data!");
        }

        return cv::imdecode(cv::Mat(1, buffer.size(), CV_8UC1, buffer.data()), cv::IMREAD_COLOR);
    }

    throw std::runtime_error("Could not initialize CURL instance");
}

void saveFile(const std::string& fileName, const char* data) {
    std::ofstream file(fileName);

    if (file) {
        file << data;
        file.close();
    }
}

void repeatEntering(const std::string& reason) {
    std::cout << reason;
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}