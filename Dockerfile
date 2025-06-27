# Gaussian Splatting Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    ninja-build \
    pkg-config \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Minicondaのインストール
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# 作業ディレクトリの設定
WORKDIR /app

# リポジトリのコピー
COPY . .

# Conda環境の作成とアクティベート
RUN conda env create --file environment.yml
SHELL ["conda", "run", "-n", "gaussian_splatting", "/bin/bash", "-c"]

# サブモジュールのビルド
RUN conda run -n gaussian_splatting pip install ./submodules/diff-gaussian-rasterization
RUN conda run -n gaussian_splatting pip install ./submodules/simple-knn

# SIBR Viewersのビルド（オプション）
RUN cd SIBR_viewers && \
    conda run -n gaussian_splatting cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && \
    conda run -n gaussian_splatting cmake --build build -j$(nproc) --target install

# 実行時のデフォルトコマンド
CMD ["conda", "run", "-n", "gaussian_splatting", "python", "train.py", "--help"]
