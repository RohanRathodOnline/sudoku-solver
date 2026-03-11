FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    git

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Build project
RUN mkdir build && cd build && \
    cmake -G Ninja .. && \
    ninja

# Run server
CMD ["./build/sudoku_server"]
