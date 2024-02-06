FROM ubuntu:latest

WORKDIR /classifier

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libcurl4-openssl-dev \
    cmake \
    g++ \
    rapidjson-dev \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /classifier/

RUN cmake . && make

RUN chmod +x /classifier/entrypoint.sh

ENTRYPOINT ["/classifier/entrypoint.sh"]