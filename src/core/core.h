#pragma once

#include "color_moments.h"
#include "sift.h"

#include <rapidjson/document.h>

const int CSV_ROW_LENGTH = 22,
          MAX_D = 1024;

void testModel(int cvMethod, int threshold, const std::string& datasetPath);
void useModel(int cvMethod, int threshold, const std::string& datasetPath,
             const std::string& modelAnswerFileName);