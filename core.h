#pragma once

#include "color_moments.h"

#include <rapidjson/document.h>
#include <mpi.h>

const int THRESHOLD = 29,
          CSV_ROW_LENGTH = 22,
          ROOT_RANK = 0;
const std::string TRAIN_DATA = "train_data.json",
                  TEST_DATA = "test_data.json",
                  VALIDATION_DATA = "validation_data.json",
                  CSV_DST_FILE = "result.csv";

int testModel(int argc, char* argv[]);
int useModel(int argc, char* argv[]);