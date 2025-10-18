# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


project(clippingPlanes)

# libraries used:
# * sibr graphics (vector2i class)
# * sibr system (programArg class)
# * boost filesystem (path class)
# * openmp

# Define build output for project
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    sibr_graphics
    sibr_assets
    sibr_raycaster
    sibr_system
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/dataset_tools/preprocess")

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
    STANDALONE  ${INSTALL_STANDALONE}   ## mean call install_runtime with bundle dependencies resolution
    COMPONENT   ${PROJECT_NAME}_install ## will create custom target to install only this project
)
