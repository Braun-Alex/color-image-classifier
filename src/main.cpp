#include "core.h"

#include <filesystem>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int launchTypeChoice, cvMethodChoice, threshold, answer, datasetLength;
    std::string dataType, cvMethodName, datasetPath, modelAnswerFileName;

    if (worldRank == ROOT_RANK) {
        std::cout << "Choose the program launch type:\n"
                  << "1) Model testing\n"
                  << "2) Model application\n";

        while (!(std::cin >> launchTypeChoice) || (launchTypeChoice < 1 || launchTypeChoice > 2)) {
            repeatEntering("Entered invalid choice. Please enter 1 or 2\n");
        }

        std::cout << "Choose the computer vision method:\n"
                  << "1) Color Moments\n"
                  << "2) Scale Invariant Feature Transform\n";

        while (!(std::cin >> cvMethodChoice) || (cvMethodChoice < 1 || cvMethodChoice > 2)) {
            repeatEntering("Entered invalid choice. Please enter 1 or 2\n");
        }

        if (cvMethodChoice == 1) {
            cvMethodName = "Color Moments";
        } else if (cvMethodChoice == 2) {
            cvMethodName = "Scale Invariant Feature Transform";
        }

        std::cout << "Enter a natural number to use a threshold based on " << cvMethodName << " method:\n";
        while (!(std::cin >> threshold) || threshold <= 0) {
            repeatEntering("Entered invalid number. Please enter the positive number:\n");
        }

        if (launchTypeChoice == 1) {
            dataType = "train or test";
        } else if (launchTypeChoice == 2) {
            dataType = "validation";
        }

        std::cout << "Enter " << dataType << " JSON file as data for the model:\n";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::getline(std::cin, datasetPath);

        while (!std::filesystem::exists(datasetPath)) {
            std::cout << "File does not exist. Please check the path and enter " << dataType
                      << " JSON file as data for the model again:\n";
            std::getline(std::cin, datasetPath);
        }
        datasetLength = static_cast<int>(datasetPath.size());

        if (launchTypeChoice == 2) {
            std::cout << "Did you use Docker launching program? If yes, model answer will be printed in console, but not CSV file\n"
                      << "1) Yes\n"
                      << "2) No\n";

            while (!(std::cin >> answer) || (answer < 1 || answer > 2)) {
                repeatEntering("Entered invalid answer. Please enter 1 or 2\n");
            }

            if (answer == 2) {
                std::cout << "Enter the name of the CSV file where you want to save the model answer in the CMake build directory:\n";
                std::getline(std::cin, modelAnswerFileName);

                std::string tmpFileName = modelAnswerFileName;
                tmpFileName.erase(std::remove_if(tmpFileName.begin(), tmpFileName.end(), isspace), tmpFileName.end());
                while (tmpFileName.size() < 5 || tmpFileName.substr(tmpFileName.size() - 4) != ".csv") {
                    std::cout << "File does not have name. Please enter the name of the CSV file where you want to save the model answer in the CMake build directory again:\n";
                    std::getline(std::cin, modelAnswerFileName);
                    tmpFileName = modelAnswerFileName;
                    tmpFileName.erase(std::remove_if(tmpFileName.begin(), tmpFileName.end(), isspace),
                                      tmpFileName.end());
                }
            }
        }
    }

    MPI_Bcast(&launchTypeChoice, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&cvMethodChoice, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&datasetLength, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    datasetPath.resize(datasetLength);
    MPI_Bcast(datasetPath.data(), datasetLength, MPI_CHAR, ROOT_RANK, MPI_COMM_WORLD);

    if (launchTypeChoice == 1) {
        testModel(worldSize, worldRank, cvMethodChoice, threshold, datasetPath);
    } else if (launchTypeChoice == 2) {
        useModel(worldSize, worldRank, cvMethodChoice, threshold, datasetPath, modelAnswerFileName);
    }

    MPI_Finalize();
    return 0;
}
