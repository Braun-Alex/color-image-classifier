#include "utils.h"

#include <rapidjson/document.h>

const std::string TRAIN_DATA = "train_data.json";

int main() {
    std::string jsonObject = fileToString(TRAIN_DATA);

    rapidjson::Document document;
    document.Parse(jsonObject.c_str());

    if (document.IsObject()) {
        const rapidjson::Value& data = document["data"];
        const rapidjson::Value& results = data["results"];

        for (rapidjson::SizeType imagePairIndex = 0; imagePairIndex < results.Size(); ++imagePairIndex) {
            const rapidjson::Value& imagePair = results[imagePairIndex];
            const rapidjson::Value& representativeData = imagePair["representativeData"];

            const std::string firstImageUrl = representativeData["image1"]["imageUrl"].GetString();
            const std::string secondImageUrl = representativeData["image2"]["imageUrl"].GetString();

            cv::Mat firstImage = downloadImage(firstImageUrl);
            cv::Mat secondImage = downloadImage(secondImageUrl);
            std::cout << "Image " << imagePairIndex << "\n";
        }
    }
}
