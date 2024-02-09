#pragma once

#include "phash.h"
#include "color_moments.h"

#include <rapidjson/document.h>
#include <mpi.h>

const int CSV_ROW_LENGTH = 22,
          ROOT_RANK = 0;
const double THRESHOLD = 39;
const std::string TRAIN_DATA = "train_data.json",
                  TEST_DATA = "test_data.json",
                  VALIDATION_DATA = "validation_data.json",
                  CSV_DST_FILE = "anotherResult.csv";

int testModel(int argc, char* argv[]);
int useModel(int argc, char* argv[]);
int getOptimalHammingThreshold(int argc, char* argv[]);
double findOptimalThreshold(const std::vector<double>& distances, const std::vector<int>& labels, double start, double end, double step);