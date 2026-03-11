FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    tzdata

WORKDIR /app

COPY . .

RUN mkdir build && cd build && \
    cmake -G Ninja .. && \
    ninja

CMD ["./build/sudoku_server"]
