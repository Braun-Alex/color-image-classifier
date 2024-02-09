#pragma once

#include "color_moments.h"

#include <filesystem>
#include <rapidjson/document.h>
#include <mpi.h>

const int CSV_ROW_LENGTH = 22,
          ROOT_RANK = 0;

void testModel(int worldSize, int worldRank, double threshold, const std::string& datasetPath);
void useModel(int worldSize, int worldRank, double threshold, const std::string& datasetPath,
             const std::string& modelAnswerFileName);