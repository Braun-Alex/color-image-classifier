#include "utils.h"

#include <rapidjson/document.h>
#include "mpi.h"

const std::string TRAIN_DATA = "train_data.json";

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    std::string jsonObject = fileToString(TRAIN_DATA);

    rapidjson::Document document;
    document.Parse(jsonObject.c_str());

    if (document.IsObject()) {
        const rapidjson::Value& data = document["data"];
        const rapidjson::Value& results = data["results"];

        auto imagesPerProcess = results.Size() / worldSize;
        auto startIndex = worldRank * imagesPerProcess;
        auto endIndex = (worldRank + 1) * imagesPerProcess;

        if (worldRank == worldSize - 1) {
            endIndex = results.Size();
        }

        for (rapidjson::SizeType imagePairIndex = startIndex; imagePairIndex < endIndex; ++imagePairIndex) {
            const rapidjson::Value& imagePair = results[imagePairIndex];
            const rapidjson::Value& representativeData = imagePair["representativeData"];

            const std::string firstImageUrl = representativeData["image1"]["imageUrl"].GetString();
            const std::string secondImageUrl = representativeData["image2"]["imageUrl"].GetString();

            cv::Mat firstImage = downloadImage(firstImageUrl);
            cv::Mat secondImage = downloadImage(secondImageUrl);
            std::cout << "Image " << imagePairIndex << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
