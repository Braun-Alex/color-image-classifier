FROM ubuntu:latest

WORKDIR /classifier

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake g++ rapidjson-dev libopencv-dev

COPY . /classifier/

RUN cmake . && make

CMD ./color_image_classifier