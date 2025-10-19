# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr



project(sibr_ulr_all)

add_subdirectory(apps)
add_subdirectory(renderer)

include(install_runtime)
subdirectory_target(${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR} "projects/ulr")
