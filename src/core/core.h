#pragma once

#include "color_moments.h"
#include "sift.h"

#include <rapidjson/document.h>
#include <mpi.h>

const int CSV_ROW_LENGTH = 22,
          ROOT_RANK = 0,
          MAX_D = 1024;

void testModel(int worldSize, int worldRank, int cvMethod, int threshold, const std::string& datasetPath);
void useModel(int worldSize, int worldRank, int cvMethod, int threshold, const std::string& datasetPath,
             const std::string& modelAnswerFileName);