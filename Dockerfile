FROM nvcr.io/nvidia/pytorch:24.03-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget git \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    cmake ninja-build libpng-dev libjpeg-dev libpython3-dev python3-distutils
ADD submodules/ ./
RUN pip install simple-knn/
RUN pip install diff-gaussian-rasterization/
RUN pip install plyfile tqdm
# RUN pip install \
#     --extra-index-url=https://pypi.nvidia.com \
#     cudf-cu12==24.4.* dask-cudf-cu12==24.4.* cuml-cu12==24.4.* \
#     cugraph-cu12==24.4.* cuspatial-cu12==24.4.* cuproj-cu12==24.4.* \
#     cuxfilter-cu12==24.4.* cucim-cu12==24.4.* pylibraft-cu12==24.4.* \
#     raft-dask-cu12==24.4.* cuvs-cu12==24.4.*
ARG username=yoshimura
ARG UID=1158
# RUN useradd -u $UID $username
RUN useradd -m -u 1158 -U -s /bin/bash $username
RUN gpasswd -a $username sudo
RUN groupadd -g 300 ldapuser
