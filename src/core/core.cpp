#include "metrics.h"
#include "core.h"

void testModel(int worldSize, int worldRank, double threshold, const std::string& datasetPath) {
    std::string jsonObject = fileToString(datasetPath);

    rapidjson::Document document;
    document.Parse(jsonObject.c_str());

    int localTP = 0;
    int localFP = 0;
    int localFN = 0;

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

            const rapidjson::Value& representativeData = imagePair["representativeData"];

            const std::string firstImageUrl = representativeData["image1"]["imageUrl"].GetString();
            const std::string secondImageUrl = representativeData["image2"]["imageUrl"].GetString();

            const cv::Mat firstImage = downloadImage(firstImageUrl);
            const cv::Mat secondImage = downloadImage(secondImageUrl);

            const std::vector<cv::Scalar> firstImageMoments = calculateMoments(firstImage);
            const std::vector<cv::Scalar> secondImageMoments = calculateMoments(secondImage);

            const bool isRecognized = areDuplicates(firstImageMoments, secondImageMoments, threshold);

            if (isRecognized && areExpectedDuplicates) {
                localTP += 1;
            } else if (isRecognized) {
                localFP += 1;
            } else if (areExpectedDuplicates) {
                localFN += 1;
            }

            if (worldRank == ROOT_RANK) {
                double estimatedProgress = 100 * static_cast<double>(imagePairIndex - startIndex + 1) / static_cast<double>(endIndex - startIndex);
                std::string progress = std::to_string(estimatedProgress);
                progress = std::string(progress.begin(), std::find(progress.begin(), progress.end(), '.'));
                std::cout << "\rTesting recognizing model... Progress: " << progress + "%" << std::flush;
            }
        }

        int totalTP, totalFP, totalFN;
        MPI_Reduce(&localTP, &totalTP, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFP, &totalFP, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFN, &totalFN, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (worldRank == ROOT_RANK) {
            const auto datasetSize = static_cast<int>(results.Size());

            int totalTN = datasetSize - totalTP - totalFP - totalFN;

            const std::vector<double> keyMetrics = calculateKeyMetrics(totalTP, totalFP,totalTN, totalFN, datasetSize);

            std::cout << "\n1) Total images: " << datasetSize
                      << "\n2) Total TP: " << totalTP
                      << "\n3) Total FP: " << totalFP
                      << "\n4) Total TN: " << totalTN
                      << "\n5) Total FN: " << totalFN
                      << "\n6) Weighted average precision: " << keyMetrics[0]
                      << "\n7) Weighted average recall: " << keyMetrics[1]
                      << "\n8) Weighted average F1-score: " << keyMetrics[2] << "\n";
        }
    }
}

void useModel(int worldSize, int worldRank, double threshold, const std::string& datasetPath,
             const std::string& modelAnswerFileName) {
    std::string jsonObject = fileToString(datasetPath);

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

        std::string localAnswer;

        if (worldRank == 0) {
            localAnswer += "taskId,answer\n";
        }

        localAnswer.reserve(imagesPerProcess * CSV_ROW_LENGTH);

        for (rapidjson::SizeType imagePairIndex = startIndex; imagePairIndex < endIndex; ++imagePairIndex) {
            const rapidjson::Value& imagePair = results[imagePairIndex];

            const rapidjson::Value& representativeData = imagePair["representativeData"];

            const std::string firstImageUrl = representativeData["image1"]["imageUrl"].GetString();
            const std::string secondImageUrl = representativeData["image2"]["imageUrl"].GetString();

            const cv::Mat firstImage = downloadImage(firstImageUrl);
            const cv::Mat secondImage = downloadImage(secondImageUrl);

            const std::vector<cv::Scalar> firstImageMoments = calculateMoments(firstImage);
            const std::vector<cv::Scalar> secondImageMoments = calculateMoments(secondImage);

            const bool isRecognized = areDuplicates(firstImageMoments, secondImageMoments, threshold);

            localAnswer += imagePair["taskId"].GetString();
            localAnswer += ",";
            localAnswer += std::to_string(static_cast<int>(isRecognized));
            if (imagePairIndex != endIndex - 1) {
                localAnswer += "\n";
            }

            if (worldRank == ROOT_RANK) {
                double estimatedProgress = 100 * static_cast<double>(imagePairIndex - startIndex + 1) / static_cast<double>(endIndex - startIndex);
                std::string progress = std::to_string(estimatedProgress);
                progress = std::string(progress.begin(), std::find(progress.begin(), progress.end(), '.'));
                std::cout << "\rRecognizing duplicates of images... Progress: " << progress + "%" << std::flush;
            }
        }

        char* processAnswer = localAnswer.data();
        int answerLength = static_cast<int>(localAnswer.size());

        int* recvcounts = NULL;

        if (worldRank == ROOT_RANK) {
            recvcounts = static_cast<int*>(malloc(worldSize * sizeof(int)));
        }

        MPI_Gather(&answerLength, 1, MPI_INT, recvcounts, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

        int totalAnswerLength;
        int* displs = NULL;
        char* totalAnswer = NULL;

        if (worldRank == ROOT_RANK) {
            displs = static_cast<int*>(malloc(worldSize * sizeof(int)));
            displs[0] = 0;
            totalAnswerLength = recvcounts[0] + 1;

            for (int i = 1; i < worldSize; ++i) {
                totalAnswerLength += recvcounts[i] + 1;
                displs[i] = displs[i - 1] + recvcounts[i - 1] + 1;
            }

            totalAnswer = static_cast<char*>(malloc(totalAnswerLength * sizeof(char)));
            for (int i = 0; i < totalAnswerLength - 1; ++i) {
                totalAnswer[i] = '\n';
            }
            totalAnswer[totalAnswerLength - 1] = '\0';
        }

        MPI_Gatherv(processAnswer, answerLength, MPI_CHAR, totalAnswer, recvcounts, displs, MPI_CHAR, ROOT_RANK, MPI_COMM_WORLD);

        if (worldRank == ROOT_RANK) {
            std::cout << "\n";

            if (modelAnswerFileName.empty()) {
                std::cout << totalAnswer << "\n";
            } else {
                saveFile(modelAnswerFileName, totalAnswer);
                std::cout << "Model answer has been saved in CMake build directory as " << modelAnswerFileName << " file\n";
            }

            free(totalAnswer);
            free(displs);
            free(recvcounts);
        }
    }
}