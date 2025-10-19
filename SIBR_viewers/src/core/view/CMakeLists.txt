# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


project(sibr_view)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

file(GLOB SHADERS "shaders/*.frag" "shaders/*.vert" "shaders/*.geom" "shaders/*.fp" "shaders/*.vp" "shaders/*.gp")
source_group("Source Files\\shaders" FILES ${SHADERS})

file(GLOB INTERFACE_SOURCES "interface/*.cpp" "interface/*.h" )
source_group("Source Files\\interface" FILES ${INTERFACE_SOURCES})

file(GLOB SOURCES
	"*.cpp" "*.h" "*.hpp"
	"shaders/*.frag" "shaders/*.vert" "shaders/*.geom"  "shaders/*.fp" "shaders/*.vp" "shaders/*.gp"
	"interface/*.cpp" "interface/*.h"
	)

## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(
	${Boost_INCLUDE_DIRS}
	${imgui_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	OpenMP::OpenMP_CXX
	imgui
	sibr_graphics
	sibr_assets
	sibr_raycaster
	sibr_scene
	sibr_video
)

add_definitions( -DSIBR_VIEW_EXPORTS -DBOOST_ALL_DYN_LINK  )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER ${SIBR_FOLDER})

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
	SHADERS "${SHADERS}"
	RSC_FOLDER "core"
    #COMPONENT   ${PROJECT_NAME}_install ## will create custom target to install only this project
)
