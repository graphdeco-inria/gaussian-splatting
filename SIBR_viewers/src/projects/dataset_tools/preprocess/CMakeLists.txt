# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr



project(SIBR_dataset_tools_preprocess)

add_subdirectory(alignMeshes)
add_subdirectory(cameraConverter)
add_subdirectory(clippingPlanes)
add_subdirectory(converters)
add_subdirectory(cropFromCenter)
add_subdirectory(distordCrop)
add_subdirectory(fullColmapProcess)
add_subdirectory(meshroomPythonScripts)
add_subdirectory(nvmToSIBR)
add_subdirectory(textureMesh)
add_subdirectory(tonemapper)
add_subdirectory(unwrapMesh)
add_subdirectory(utils)
add_subdirectory(prepareColmap4Sibr)
add_subdirectory(realityCaptureTools)
