# 3D Gaussian Splatting for Real-Time Radiance Field Rendering
Bernhard Kerbl*, Georgios Kopanas*, Thomas Leimkühler, George Drettakis (* indicates equal contribution)<br>
| [Webpage](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [Full Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) |
[Video](https://youtu.be/T_kXY43VZnk) | [Other GRAPHDECO Publications](http://www-sop.inria.fr/reves/publis/gdindex.php) | [FUNGRAPH project page](https://fungraph.inria.fr)

[T&T+DB Datasets](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) | [Pre-trained Models TODO](TODO)| [Evaluation Renderings TODO](TODO)|  <br>
![Teaser image](assets/teaser.png)

This repository contains the code associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). We further provide the reference images used to create the error metrics reported in the paper, as well as recently created, pre-trained models. 

<a href="https://www.inria.fr/"><img height="100" src="assets/logo_inria.png"> </a>
<a href="https://univ-cotedazur.eu/"><img height="100" src="assets/logo_uca.png"> </a>
<a href="https://www.mpi-inf.mpg.de"><img height="100" src="assets/logo_mpi.png"> </a> 
<a href="https://team.inria.fr/graphdeco/"> <img style="width:100%;" src="assets/logo_graphdeco.png"></a>

Abstract: *Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (≥ 30 fps) novel-view synthesis at 1080p resolution. First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space; Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene; Third, we develop a fast visibility-aware rendering algorithm that supports anisotropic splatting and both accelerates training and allows realtime rendering. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>


## Funding and Acknowledgments

This research was funded by the ERC Advanced grant FUNGRAPH No 788065. The authors are grateful to Adobe for generous donations, the OPAL infrastructure from Université Côte d’Azur and for the HPC resources from GENCI–IDRIS (Grant 2022-AD011013409). The authors thank the anonymous reviewers for their valuable feedback, P. Hedman and A. Tewari for proofreading earlier drafts also T. Müller, A. Yu and S. Fridovich-Keil for helping with the comparisons.

## Cloning the Repository

The repository contains submodules, thus please check it out with
```shell
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

## Overview

The codebase has 4 main components:
- A PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs
- A network viewer that allows to connect to and visualize the optimization process
- An OpenGL-based real-time viewer to render trained models in real-time.
- A script to help you turn your own images into optimization-ready SfM data sets

The components have different requirements w.r.t. both hardware and software. They have been tested on Windows 10 and Linux Ubuntu 22. Instructions for setting up and running each of them are found in the sections below.

## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM to train the largest scenes in our test suite

### Software Requirements
- C++ Compiler (Visual Studio 2019 for Windows)
- CUDA 11 SDK for PyTorch extensions (we used 11.8)
- Conda (recommended for easy setup)

### Setup

Our provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```
Note for Windows: if setup stops and you are prompted to set ```DISTUTILS_USE_SDK=1```, please complete the environment setup with:
```shell
SET DISTUTILS_USE_SDK=1
conda env update --file environment.yml
```

Tip: Downloading packages and creating a new environment with Conda can require a significant amount of disk space. By default, Conda will use the main system hard drive. You can avoid this by specifying a different package download location and an environment on a different drive:

```shell
conda config --add pkgs_dirs <Drive>/<pkg_path>
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting
```

#### Custom Install

If you can afford the disk space, we recommend using our environment files for setting up a training environment identical to ours. If you want to make changes, please note that major version changes might affect the results of our method. However, our (limited) experiments suggest that the codebase works just fine inside a more up-to-date environment (Python 3.8, PyTorch 2.0.0, CUDA 11.8).

### Running

To run the optimizer, simply use

```shell
python train.py -s <path to dataset>
```

The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). You can find our SfM data sets for Tanks&Temples and Deep Blending [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt+db.zip). If you do not provide an output model directory (```-m```), trained models are written to folders with randomized unique names inside the ```output``` directory. At this point, the trained models may be viewed with the real-time viewer (see further below).

### Evaluation
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python train.py -s <path to dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

We further provide the ```full_eval.py``` script. This script specifies the routine used in our evaluation and demonstrates the use of some additional parameters, e.g., ```--images (-i)``` to define alternative image directories within COLMAP data sets. If you have downloaded and extracted all the training data, you can run it like this:
```
python full_eval.py --m360 <mipnerf360 folder> --tat <tanks and temples folder> --db <deep blending folder>
```
In the current version, this process takes about 7h on our reference machine containing an A6000.

## Network Viewer

The Network Viewer can be used to observe the training process and watch the model as it forms. It is not required for the basic workflow, but it is automatically set up when preparing SIBR for the Real-Time Viewer.  

### Hardware Requirements

- OpenGL 4.5-ready GPU
- 8 GB VRAM

### Software Requirements
- C++ Compiler (Visual Studio 2019 for Windows)
- CUDA 11 Developer SDK (we used 11.8)
- CMake (recent version, we used 3.24)
- 7zip (Windows)

### Setup

If you cloned with submodules (e.g., using ```--recursive```), the source code for the viewers is found in ```SIBR_viewers_(windows|linux)``` (choose whichever fits your OS). The network viewer runs within the SIBR framework for Image-based Rendering applications.

#### Windows
On Windows, CMake should take care of your dependencies.
```shell
cd SIBR_viewers_windows
cmake -Bbuild .
cmake --build build --target install --config RelWithDebInfo
```
You may specify a different configuration, e.g. ```Debug``` if you need more control during development.

#### Ubuntu
For Ubuntu, you will need to install a few dependencies before running the project setup.
```shell
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# Project setup
cd SIBR_viewers_linux
cmake -Bbuild .
cmake --build build --target install
```
If you receive a build error related to ```libglfw```, locate the library directory and set up a symbolic link there ```libglfw3.so``` &rarr; ```<your actual liblgfw lib>```. 

### Running
You may run the compiled ```SIBR_remoteGaussian_app_<config>``` either by opening the build in your C++ development IDE or by running the installed app in ```install/bin```, e.g.: 
```shell
./SIBR_viewers_windows/install/bin/SIBR_remoteGaussian_app_rwdi.exe
```

The network viewer allows you to connect to a running training process on the same or a different machine. If you are training on the same machine and OS, no command line parameters should be required: the optimizer communicates the location of the training data to the network viewer. By default, optimizer and network viewer will try to establish a connection on **localhost** on port **6009**. You can change this behavior by providing matching ```--ip``` and ```--port``` parameters to both the optimizer and the network viewer. If for some reason the path used by the optimizer to find the training data is not reachable by the network viewer (e.g., due to them running on different (virtual) machines), you may specify an override location to the viewer by using ```--path <source path>```. 

### Navigation

The SIBR interface provides several methods of navigating the scene. By default, you will be started with an FPS navigator, which you can control with ```W, A, S, D``` for camera translation and ```Q, E, I, K, J, L``` for rotation. Alternatively, you may want to use a Trackball-style navigator (select from the floating menu). You can also snap to a camera from the data set with the ```Snap to``` button or find the closest camera with ```Snap to closest```. The floating menues also allow you to change the navigation speed. You can use the ```Scaling Modifier``` to control the size of the displayed Gaussians, or show the initial point cloud.

## Real-Time Viewer

The Real-Time Viewer can be used to render trained models with real-time frame rates.

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- OpenGL 4.5-ready GPU
- 8 GB VRAM

### Software Requirements
- C++ Compiler (Visual Studio 2019 for Windows)
- CUDA 11 Developer SDK
- CMake (recent version)
- 7zip (Windows)

### Setup

The setup is the same as for the remote viewer.

### Running
You may run the compiled ```SIBR_gaussianViewer_app_<config>``` either by opening the build in your C++ development IDE or by running the installed app in ```install/bin```, e.g.: 
```shell
./SIBR_viewers_windows/install/bin/SIBR_gaussianViewer_app_rwdi.exe --model-path <path to trained model>
```

It should suffice to provide the ```--model-path``` parameter pointing to a trained model directory. Alternatively, you can specify an override location for training input data using ```--path```. To use a specific resolution other than the auto-chosen one, specify ```--rendering-size <width> <height>```. To unlock the full frame rate, please disable V-Sync on your machine and enter full-screen mode (Menu &rarr; Display).

### Navigation

Navigation works exactly as it does in the network viewer. However, you also have the option to visualize the Gaussians by rendering them as ellipsoids from the floating menu.

## Converting your own Scenes

We provide a converter script ```convert.py```, which uses COLMAP to extract SfM information. Optionally, you can use ImageMagick to resize the undistorted images. This rescaling is similar to MipNeRF360, i.e., it creates images with 1/2, 1/4 and 1/8 the original resolution in corresponding folders. To use them, please first install a recent version of COLMAP (ideally CUDA-powered) and ImageMagick. Put the images you want to use in a directory ```<location>/input```. If you have COLMAP and ImageMagick on your system path, you can simply run 
```shell
python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed
```
Alternatively, you can use the optional parameters ```--colmap_executable``` and ```--magick_executable``` to point to the respective paths. Please not that on Windows, the executable should point to the COLMAP ```.bat``` file that takes care of setting the execution environment. Once done, ```<location>``` will contain the expected COLMAP data set structure with undistorted, differently sized input images, in addition to your original images and temporary data in the directory ```distorted```.
## FAQ
- *Where do I get data sets, e.g., those referenced in ```full_eval.py```?* The MipNeRF360 data set is provided by the authors of the original paper on the project site. Note that two of the data sets cannot be openly shared and require you to consult the authors directly. For Tanks&Temples and Deep Blending, please use the download links provided at the top of the page.

- *24 GB of VRAM for training is a lot! Can't we do it with less?* Yes, most likely. By our calculations it should be possible with **way** less memory (~8GB). If we can find the time we will try to achieve this. If some PyTorch veteran out there wants to tackle this, we look forward to your pull request!

