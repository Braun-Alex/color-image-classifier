#include "metrics.h"
#include "core.h"
#include <omp.h>

void testModel(int cvMethod, int threshold, const std::string& datasetPath) {
    std::string jsonObject = fileToString(datasetPath);

    rapidjson::Document document;
    document.Parse(jsonObject.c_str());

    int totalTP = 0;
    int totalFP = 0;
    int totalFN = 0;

    if (document.IsObject()) {
        const rapidjson::Value& data = document["data"];
        const rapidjson::Value& results = data["results"];

        const auto datasetSize = static_cast<int>(results.Size());

        const double startMoment = omp_get_wtime();

#pragma omp parallel reduction(+:totalTP, totalFP, totalFN)
        {
            int threadNumber = omp_get_thread_num();

            if (cvMethod == 1) {
#pragma omp for schedule(static, 1)
                for (rapidjson::SizeType imagePairIndex = 0; imagePairIndex < datasetSize; ++imagePairIndex) {
                    const rapidjson::Value& imagePair = results[imagePairIndex];

                    const std::string answer = imagePair["answers"][0]["answer"][0]["id"].GetString();
                    const int areExpectedDuplicates = std::stoi(answer);

                    const rapidjson::Value& representativeData = imagePair["representativeData"];

                    const std::string firstImageUrl = representativeData["image1"]["imageUrl"].GetString();
                    const std::string secondImageUrl = representativeData["image2"]["imageUrl"].GetString();

                    cv::Mat firstImage, secondImage;

                    resizeImage(downloadImage(firstImageUrl, cv::IMREAD_COLOR), firstImage, MAX_D);
                    resizeImage(downloadImage(secondImageUrl, cv::IMREAD_COLOR), secondImage, MAX_D);

                    const std::vector<cv::Scalar> firstImageMoments = moments::calculateMoments(firstImage);
                    const std::vector<cv::Scalar> secondImageMoments = moments::calculateMoments(secondImage);

                    const bool isRecognized = moments::areDuplicates(firstImageMoments, secondImageMoments, threshold);

                    if (isRecognized && areExpectedDuplicates) {
                        totalTP += 1;
                    } else if (isRecognized) {
                        totalFP += 1;
                    } else if (areExpectedDuplicates) {
                        totalFN += 1;
                    }

                    if (threadNumber == 0) {
                        double estimatedProgress = 100 * static_cast<double>(imagePairIndex + 1) / static_cast<double>(datasetSize);
                        std::string progress = std::to_string(estimatedProgress);
                        progress = std::string(progress.begin(), std::find(progress.begin(), progress.end(), '.'));
                        std::cout << "\rTesting recognizing model... Progress: " << progress + "%" << std::flush;
                    }
                }
            } else if (cvMethod == 2) {
                std::vector<std::vector<cv::DMatch>> imageMatches;
                cv::Ptr<cv::SIFT> siftInstance = cv::SIFT::create();

                cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
                indexParams->setAlgorithm(cvflann::FLANN_INDEX_KDTREE);
                indexParams->setInt("trees", 5);

                cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
                searchParams->setInt("checks", 50);

                cv::Ptr<cv::FlannBasedMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);

#pragma omp for schedule(static, 1)
                for (rapidjson::SizeType imagePairIndex = 0; imagePairIndex < datasetSize; ++imagePairIndex) {
                    const rapidjson::Value &imagePair = results[imagePairIndex];

                    const std::string answer = imagePair["answers"][0]["answer"][0]["id"].GetString();
                    const int areExpectedDuplicates = std::stoi(answer);

                    const rapidjson::Value& representativeData = imagePair["representativeData"];

                    const std::string firstImageUrl = representativeData["image1"]["imageUrl"].GetString();
                    const std::string secondImageUrl = representativeData["image2"]["imageUrl"].GetString();

                    cv::Mat firstImage, secondImage;

                    resizeImage(downloadImage(firstImageUrl, cv::IMREAD_GRAYSCALE), firstImage, MAX_D);
                    resizeImage(downloadImage(secondImageUrl, cv::IMREAD_GRAYSCALE), secondImage, MAX_D);

                    int minImageKeypointSize = sift::calculateMatches(firstImage, secondImage, imageMatches, siftInstance, matcher);

                    const bool isRecognized = sift::areDuplicates(static_cast<int>(imageMatches.size()), minImageKeypointSize, threshold);

                    if (isRecognized && areExpectedDuplicates) {
                        totalTP += 1;
                    } else if (isRecognized) {
                        totalFP += 1;
                    } else if (areExpectedDuplicates) {
                        totalFN += 1;
                    }

                    if (threadNumber == 0) {
                        double estimatedProgress = 100 * static_cast<double>(imagePairIndex + 1) / static_cast<double>(datasetSize);
                        std::string progress = std::to_string(estimatedProgress);
                        progress = std::string(progress.begin(), std::find(progress.begin(), progress.end(), '.'));
                        std::cout << "\rTesting recognizing model... Progress: " << progress + "%" << std::flush;
                    }
                }
            }
        }

        const double endMoment = omp_get_wtime();
        const double duration = endMoment - startMoment;

        int totalTN = datasetSize - totalTP - totalFP - totalFN;

        const std::vector<double> keyMetrics = calculateKeyMetrics(totalTP, totalFP, totalTN, totalFN, datasetSize);

        std::cout << "\n1) Total images: " << datasetSize
                  << "\n2) Total TP: " << totalTP
                  << "\n3) Total FP: " << totalFP
                  << "\n4) Total TN: " << totalTN
                  << "\n5) Total FN: " << totalFN
                  << "\n6) Weighted average precision: " << keyMetrics[0]
                  << "\n7) Weighted average recall: " << keyMetrics[1]
                  << "\n8) Weighted average F1-score: " << keyMetrics[2]
                  << "\n9) Execution time: " << duration << " seconds\n";
    }
}

void useModel(int cvMethod, int threshold, const std::string& datasetPath,
              const std::string& modelAnswerFileName) {
    std::string jsonObject = fileToString(datasetPath);

    rapidjson::Document document;
    document.Parse(jsonObject.c_str());

    if (document.IsObject()) {
        const rapidjson::Value& data = document["data"];
        const rapidjson::Value& results = data["results"];

        const auto datasetSize = static_cast<int>(results.Size());
        std::vector<std::string> localAnswers(omp_get_max_threads());
        localAnswers[0] += "taskId,answer\n";

        const double startMoment = omp_get_wtime();

#pragma omp parallel
        {
            int threadNumber = omp_get_thread_num();
            std::string& localAnswer = localAnswers[threadNumber];
            localAnswer.reserve(datasetSize * CSV_ROW_LENGTH / omp_get_num_threads());

            if (cvMethod == 1) {
#pragma omp for schedule(static, 1)
                for (rapidjson::SizeType imagePairIndex = 0; imagePairIndex < datasetSize; ++imagePairIndex) {
                    const rapidjson::Value& imagePair = results[imagePairIndex];

                    const rapidjson::Value& representativeData = imagePair["representativeData"];

                    const std::string firstImageUrl = representativeData["image1"]["imageUrl"].GetString();
                    const std::string secondImageUrl = representativeData["image2"]["imageUrl"].GetString();

                    cv::Mat firstImage, secondImage;

                    resizeImage(downloadImage(firstImageUrl, cv::IMREAD_COLOR), firstImage, MAX_D);
                    resizeImage(downloadImage(secondImageUrl, cv::IMREAD_COLOR), secondImage, MAX_D);

                    const std::vector<cv::Scalar> firstImageMoments = moments::calculateMoments(firstImage);
                    const std::vector<cv::Scalar> secondImageMoments = moments::calculateMoments(secondImage);

                    const bool isRecognized = moments::areDuplicates(firstImageMoments, secondImageMoments, threshold);

                    localAnswer += imagePair["taskId"].GetString();
                    localAnswer += ",";
                    localAnswer += std::to_string(static_cast<int>(isRecognized));
                    localAnswer += "\n";

                    if (threadNumber == 0) {
                        double estimatedProgress = 100 * static_cast<double>(imagePairIndex + 1) / static_cast<double>(datasetSize);
                        std::string progress = std::to_string(estimatedProgress);
                        progress = std::string(progress.begin(), std::find(progress.begin(), progress.end(), '.'));
                        std::cout << "\rRecognizing duplicates of images... Progress: " << progress + "%" << std::flush;
                    }
                }
            } else if (cvMethod == 2) {
                std::vector<std::vector<cv::DMatch>> imageMatches;
                cv::Ptr<cv::SIFT> siftInstance = cv::SIFT::create();

                cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
                indexParams->setAlgorithm(cvflann::FLANN_INDEX_KDTREE);
                indexParams->setInt("trees", 5);

                cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
                searchParams->setInt("checks", 50);

                cv::Ptr<cv::FlannBasedMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);

#pragma omp for schedule(static, 1)
                for (rapidjson::SizeType imagePairIndex = 0; imagePairIndex < datasetSize; ++imagePairIndex) {
                    const rapidjson::Value& imagePair = results[imagePairIndex];

                    const rapidjson::Value& representativeData = imagePair["representativeData"];

                    const std::string firstImageUrl = representativeData["image1"]["imageUrl"].GetString();
                    const std::string secondImageUrl = representativeData["image2"]["imageUrl"].GetString();

                    cv::Mat firstImage, secondImage;

                    resizeImage(downloadImage(firstImageUrl, cv::IMREAD_GRAYSCALE), firstImage, MAX_D);
                    resizeImage(downloadImage(secondImageUrl, cv::IMREAD_GRAYSCALE), secondImage, MAX_D);

                    int minImageKeypointSize = sift::calculateMatches(firstImage, secondImage, imageMatches, siftInstance, matcher);

                    const bool isRecognized = sift::areDuplicates(static_cast<int>(imageMatches.size()), minImageKeypointSize, threshold);

                    localAnswer += imagePair["taskId"].GetString();
                    localAnswer += ",";
                    localAnswer += std::to_string(static_cast<int>(isRecognized));
                    localAnswer += "\n";

                    if (threadNumber == 0) {
                        double estimatedProgress = 100 * static_cast<double>(imagePairIndex + 1) / static_cast<double>(datasetSize);
                        std::string progress = std::to_string(estimatedProgress);
                        progress = std::string(progress.begin(), std::find(progress.begin(), progress.end(), '.'));
                        std::cout << "\rRecognizing duplicates of images... Progress: " << progress + "%" << std::flush;
                    }
                }
            }
        }

        const double endMoment = omp_get_wtime();
        const double duration = endMoment - startMoment;

        std::string totalAnswer;
        totalAnswer.reserve(datasetSize * CSV_ROW_LENGTH);
        for (const auto& localAnswer: localAnswers) {
            totalAnswer += localAnswer;
        }

        std::cout << "\n";

        if (modelAnswerFileName.empty()) {
            std::cout << totalAnswer << "Execution time: " << duration << " seconds\n";
        } else {
            saveFile(modelAnswerFileName, totalAnswer.c_str());
            std::cout << "Model answer has been saved in CMake build directory as " << modelAnswerFileName << " file\n";
        }
    }
}