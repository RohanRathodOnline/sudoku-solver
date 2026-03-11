FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build

WORKDIR /app
COPY . .

RUN mkdir build && cd build && \
    cmake -G Ninja .. && \
    ninja

CMD ["./build/server"]
