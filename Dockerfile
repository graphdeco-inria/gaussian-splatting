ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
ARG USER_ID=1000

# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION
ARG USER_ID

# metainformation
LABEL org.opencontainers.image.version="0.1.18"
LABEL org.opencontainers.image.licenses="Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}"

ARG CUDA_ARCHITECTURES=70

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV CUDA_HOME="/usr/local/cuda"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-distutils \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install GLOG (required by ceres).
RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j "$(nproc)" && \
    make install && \
    cd ../.. && \
    rm -rf glog

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Install Ceres-solver (required by colmap).
RUN git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch && \
    cd ceres-solver && \
    git checkout "$(git describe --tags)" && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j "$(nproc)" && \
    make install && \
    cd ../.. && \
    rm -rf ceres-solver

# Install colmap.
RUN git clone --branch 3.8 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    make -j "$(nproc)" && \
    make install && \
    cd ../.. && \
    rm -rf colmap

# Create non-root user and set up environment.
RUN useradd -m -d /home/user -g root -G sudo -u ${USER_ID} user
RUN usermod -aG sudo user
RUN echo "user:user" | chpasswd
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${USER_ID}
WORKDIR /home/user

ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

RUN python3 -m pip install --upgrade pip setuptools pathtools promise pybind11

USER root
RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python3 -m pip install \
    torch==2.0.0+cu${CUDA_VER} \
    torchvision==0.15.0+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}
USER ${USER_ID}

RUN git clone --branch v0.4.0 --recursive https://github.com/colmap/pycolmap.git && \
    cd pycolmap && \
    python3 -m pip install . && \
    cd ..

RUN git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git && \
    cd Hierarchical-Localization && \
    python3 -m pip install -e . && \
    cd ..

RUN python3 -m pip install omegaconf

USER root
RUN chsh -s /bin/bash user
USER ${USER_ID}