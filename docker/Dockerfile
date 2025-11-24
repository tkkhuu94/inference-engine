# Base Image: NVIDIA CUDA 13.0 Development on Ubuntu 24.04
# This gives 'nvcc', CUDA headers, and GPU drivers pre-configured.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    vim \
    python3 \
    python3-pip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Libraries to display with opencv highgui
RUN apt-get update && apt-get install -y \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0

# Bazelisk
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# 4. Build OpenCV 4.11.0 from Source (WITH CONTRIB)
WORKDIR /tmp
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.11.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.11.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mkdir -p build && cd build && \
    cmake ../opencv-4.11.0 \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.11.0/modules \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D WITH_NVCUVENC=OFF \
    -D WITH_NVCUVID=OFF \
    -D ENABLE_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
    && make -j1 && make install && ldconfig && \
    cd /tmp && rm -rf opencv-4.11.0 opencv_contrib-4.11.0 opencv.zip opencv_contrib.zip build

# 5. Set up workspace
WORKDIR /workspace

CMD ["/bin/bash"]
