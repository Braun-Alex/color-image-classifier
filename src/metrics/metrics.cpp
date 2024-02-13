#include "metrics.h"

std::vector<double> calculateKeyMetrics(int TP, int FP, int TN, int FN, int datasetSize) {
    auto nonDuplicateSupport = static_cast<double>(TN + FP);
    auto duplicateSupport = static_cast<double>(TP + FN);

    double nonDuplicateWeight = nonDuplicateSupport / static_cast<double>(datasetSize);
    double duplicateWeight = duplicateSupport / static_cast<double>(datasetSize);

    double nonDuplicatePrecision = static_cast<double>(TN) / static_cast<double>(TN + FN);
    double nonDuplicateRecall = static_cast<double>(TN) / static_cast<double>(TN + FP);

    double duplicatePrecision = static_cast<double>(TP) / static_cast<double>(TP + FP);
    double duplicateRecall = static_cast<double>(TP) / static_cast<double>(TP + FN);

    double nonDuplicateF1Score = 2 * nonDuplicatePrecision * nonDuplicateRecall / (nonDuplicatePrecision + nonDuplicateRecall);
    double duplicateF1Score = 2 * duplicatePrecision * duplicateRecall / (duplicatePrecision + duplicateRecall);

    double weightedAveragePrecision = nonDuplicateWeight * nonDuplicatePrecision + duplicateWeight * duplicatePrecision;
    double weightedAverageRecall = nonDuplicateWeight * nonDuplicateRecall + duplicateWeight * duplicateRecall;
    double weightedAverageF1Score = nonDuplicateWeight * nonDuplicateF1Score + duplicateWeight * duplicateF1Score;

    std::vector<double> keyMetrics(3);

    keyMetrics[0] = weightedAveragePrecision;
    keyMetrics[1] = weightedAverageRecall;
    keyMetrics[2] = weightedAverageF1Score;

    return keyMetrics;
}