#include "core.h"

int testModel(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    std::string jsonObject = fileToString(TEST_DATA);

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

            const bool isRecognized = areDuplicates(firstImageMoments, secondImageMoments, THRESHOLD);

            localRecognizedDuplicates += static_cast<int>(isRecognized);

            if (isRecognized && areExpectedDuplicates) {
                localTruePositives += 1;
            } else if (isRecognized) {
                localFalsePositives += 1;
            } else if (areExpectedDuplicates) {
                localFalseNegatives += 1;
            }

            if (worldRank == ROOT_RANK) {
                double estimatedProgress = 100 * static_cast<double>(imagePairIndex - startIndex + 1) / static_cast<double>(endIndex - startIndex);
                std::string progress = std::to_string(estimatedProgress);
                progress = std::string(progress.begin(), std::find(progress.begin(), progress.end(), '.'));
                std::cout << "\rTesting recognizing model... Progress: " << progress + "%" << std::flush;
            }
        }

        int totalDuplicates, totalRecognizedDuplicates, totalTruePositives, totalFalsePositives, totalFalseNegatives;
        MPI_Reduce(&localDuplicates, &totalDuplicates, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localRecognizedDuplicates, &totalRecognizedDuplicates, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localTruePositives, &totalTruePositives, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFalsePositives, &totalFalsePositives, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFalseNegatives, &totalFalseNegatives, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (worldRank == ROOT_RANK) {
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

int useModel(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    std::string jsonObject = fileToString(VALIDATION_DATA);

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

            const bool isRecognized = areDuplicates(firstImageMoments, secondImageMoments, THRESHOLD);

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

            saveFile(CSV_DST_FILE, totalAnswer);

            free(totalAnswer);
            free(displs);
            free(recvcounts);
        }
    }

    MPI_Finalize();
    return 0;
}