FROM ubuntu:latest

WORKDIR /classifier

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    cmake \
    g++ \
    rapidjson-dev \
    libopencv-dev \
    mpich \
    && rm -rf /var/lib/apt/lists/*

COPY . /classifier/

RUN cmake . && make

RUN chmod +x /classifier/entrypoint.sh

ENTRYPOINT ["/classifier/entrypoint.sh"]