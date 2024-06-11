FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

RUN apt update -y
RUN apt install -y wget

RUN apt install -y build-essential
RUN apt-get install -y cmake
RUN apt install -y python3 python3-pip python-is-python3

## Miniconda
RUN mkdir -p /miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh
RUN bash /miniconda3/miniconda.sh -b -u -p /miniconda3
RUN /miniconda3/bin/conda init bash


WORKDIR /root/gaussian-splatting
COPY environment.yml /root/gaussian-splatting/environment.yml
COPY submodules /root/gaussian-splatting/submodules
RUN /miniconda3/bin/conda env create -f environment.yml
# conda env createがが終わったら不要
RUN rm -rf /root/gaussian-splatting 

RUN apt-get install -y git

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y --fix-missing
RUN apt-get install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev
RUN apt-get install -y libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev ninja-build

# SIBR viewer
COPY SIBR_viewers/CMakeLists.txt /root/SIBR_viewers/CMakeLists.txt
COPY SIBR_viewers/cmake /root/SIBR_viewers/cmake
COPY SIBR_viewers/src /root/SIBR_viewers/src
RUN cd /root/SIBR_viewers && \
    cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j24 --target install

ENV LIBGL_ALWAYS_INDIRECT=1

RUN apt-get install -y mesa-utils
RUN apt-get install -y libglew-dev

CMD ["/bin/bash"]