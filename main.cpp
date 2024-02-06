#include "color_moments.h"

#include <rapidjson/document.h>
#include "mpi.h"

const std::string TRAIN_DATA = "test_data.json";

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    std::string jsonObject = fileToString(TRAIN_DATA);

    rapidjson::Document document;
    document.Parse(jsonObject.c_str());

    int localDuplicates = 0;
    int localRecognizedDuplicates = 0;

    int localTruePositives = 0;
    int localFalsePositives = 0;
    int localFalseNegatives = 0;

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

            const std::string answer = imagePair["answers"][0]["answer"][0]["id"].GetString();
            const int areExpectedDuplicates = std::stoi(answer);
            localDuplicates += areExpectedDuplicates;

            const rapidjson::Value& representativeData = imagePair["representativeData"];

            const std::string firstImageUrl = representativeData["image1"]["imageUrl"].GetString();
            const std::string secondImageUrl = representativeData["image2"]["imageUrl"].GetString();

            const cv::Mat firstImage = downloadImage(firstImageUrl);
            const cv::Mat secondImage = downloadImage(secondImageUrl);

            const std::vector<cv::Scalar> firstImageMoments = calculateMoments(firstImage);
            const std::vector<cv::Scalar> secondImageMoments = calculateMoments(secondImage);

            const bool isRecognized = areDuplicates(firstImageMoments, secondImageMoments, 29);

            if (isRecognized) {
                localRecognizedDuplicates += 1;
            }

            if (isRecognized && areExpectedDuplicates) {
                localTruePositives += 1;
            } else if (isRecognized) {
                localFalsePositives += 1;
            } else if (areExpectedDuplicates) {
                localFalseNegatives += 1;
            }

            if (worldRank == 0) {
                double estimatedProgress = 100 * static_cast<double>(imagePairIndex - startIndex + 1) / static_cast<double>(endIndex - startIndex);
                std::string progress = std::to_string(estimatedProgress);
                progress = std::string(progress.begin(), std::find(progress.begin(), progress.end(), '.'));
                std::cout << "\rProgress: " << progress + "%" << std::flush;
            }
        }

        int totalDuplicates, totalRecognizedDuplicates, totalTruePositives, totalFalsePositives, totalFalseNegatives;
        MPI_Reduce(&localDuplicates, &totalDuplicates, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localRecognizedDuplicates, &totalRecognizedDuplicates, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localTruePositives, &totalTruePositives, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFalsePositives, &totalFalsePositives, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFalseNegatives, &totalFalseNegatives, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (worldRank == 0) {
            const double precision = static_cast<double>(totalTruePositives) / static_cast<double>(totalTruePositives + totalFalsePositives);
            const double recall = static_cast<double>(totalTruePositives) / static_cast<double>(totalTruePositives + totalFalseNegatives);
            const double f1Score = 2 * precision * recall / (precision + recall);

            std::cout << "\n1) Total images: " << results.Size()
                      << "\n2) Total duplicates: " << totalDuplicates
                      << "\n3) Total recognized duplicates: " << totalRecognizedDuplicates
                      << "\n4) Total TP: " << totalTruePositives
                      << "\n5) Total FP: " << totalFalsePositives
                      << "\n6) Total FN: " << totalFalseNegatives
                      << "\n7) Precision: " << precision
                      << "\n8) Recall: " << recall
                      << "\n9) F1 score: " << f1Score << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
