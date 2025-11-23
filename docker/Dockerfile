# Base Image: NVIDIA CUDA 13.0 Development on Ubuntu 24.04
# This gives 'nvcc', CUDA headers, and GPU drivers pre-configured.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# 'libopencv-dev' includes Core, DNN, and HighGUI.
# Also install 'libgtk-3-dev' to ensure HighGUI has window support (even if headless).
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    wget \
    unzip \
    vim \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Bazelisk
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Set up workspace
WORKDIR /workspace

CMD ["/bin/bash"]
