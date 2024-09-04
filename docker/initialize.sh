#!/bin/bash


#git clone https://github.com/NVIDIA/cuda-samples #<- To Test docker Cuda image..
#cd cuda-samples


#also get add-ons
git clone https://github.com/antimatter15/splat
git clone https://github.com/ReshotAI/gaussian-splatting-blender-addon/
git clone https://github.com/francescofugazzi/3dgsconverter #needs scikit-learn




git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
#git checkout $(git describe --tags) # Checkout the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES="60;70;80" #-DUSE_CUDA=OFF 
make -j8
sudo make install
cd ..
cd ..


git clone https://github.com/colmap/colmap
cd colmap
#git checkout dev
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="60;70;80" #-DCUDA_ENABLED=OFF 
make -j8
sudo make install
cd ..
cd ..



python3.10 -m pip install plyfile tqdm scikit-learn
python3.10 -m pip install  https://huggingface.co/camenduru/gaussian-splatting/resolve/main/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl
python3.10 -m pip install  https://huggingface.co/camenduru/gaussian-splatting/resolve/main/simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl
python3.10 -m pip install torchvision

ln -s docker/run.sh ./run.sh

#Build viewer
#sudo apt install -y libimgui-dev libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
#git clone https://github.com/JayFoxRox/SIBR_viewers
#cd SIBR_viewers
#cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release -DASSIMP_LIBRARY=/usr/lib/x86_64-linux-gnu/libassimp.so
#cmake --build build -j24 --target install
#cd ..

#sudo apt-get -y install cuda
sudo apt -y install nvidia-cuda-toolkit

exit 0
