# Computer Vision Algorithms: Color Moments and SIFT methods 🎨

## Overview
This project implements and integrates the computer vision algorithms, specifically the "Color Moments Method" and SIFT method, using C++ as the primary programming language. It leverages key technologies such as curl for downloading images into memory, CMake for compilation, RapidJSON for processing JSON objects, MPI for parallelizing the program on multi-core clusters using distributed memory, OpenCV for computer vision tasks, and Docker for building and running the program across different operating systems and devices.

### Key Technologies 🛠️
- **curl** 🌐. For image downloading.
- **CMake** 🏗. For project compilation.
- **RapidJSON** 📄. For JSON object processing.
- **MPI** 💻. For program parallelization on multi-core clusters with distributed memory.
- **OpenCV** 📸. The most widely used computer vision library.
- **Docker** 🐳. For building and running the program on various platforms.

### External Libraries Used 📚
- `libcurl4-openssl-dev`
- `cmake`
- `g++`
- `rapidjson-dev`
- `openmpi-bin`
- `openmpi-common`
- `libopenmpi-dev`
- `libopencv-dev`

## Program Behavior 🤖
The program operates in an interactive mode. Users can specify the execution type of the program: model testing (which performs image duplicate recognition and outputs statistics of its operation in the form of metrics) or model application (where recognition is performed, and the result is output to a CSV file).

### Features 🔍
- Users can define the thresholds used by the methods and the path to the file depending on the execution type (training, testing, or validation data).
- If the execution type is model application, users need to specify if the program is run using Docker. If so, the result is output to the console; otherwise, users must enter the path to a CSV file where the result will be saved.
- The program displays the percentage of the dataset processed in real-time, allowing users to estimate the execution speed and the approximate time required for processing the entire dataset.

## Program Acceleration ⚡
Using MPI has significantly accelerated the program, proportional to the number of logical cores of the system or cluster. Loading images takes considerable time, so all allocated processes (equal to the number of logical cores, which can be found using `nproc`) divide the entire dataset into equal parts, after which the results from these datasets are aggregated into a single result.

## Building and Using the Program 🏗️
To download, compile, and run the program, you need to have Git and Docker installed.

### Launch Instructions 🚀
1. Clone the repository: `git clone https://github.com/Braun-Alex/color-image-classifier`
2. Navigate to the project directory: `cd color-image-classifier`
3. Build the Docker image: `docker build -t color-image-classifier .`
4. Run the Docker container interactively: `docker run -it --rm color-image-classifier`
