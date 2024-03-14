FROM ubuntu:latest

WORKDIR /classifier

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libcurl4-openssl-dev \
    cmake \
    g++ \
    rapidjson-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /classifier/

RUN cmake . && make

CMD ./color_image_classifier