#include "utils.h"

#include <fstream>
#include <sstream>

std::string fileToString(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file) {
        throw std::runtime_error("File \"" + fileName + "\" could not be opened!");
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    return buffer.str();
}