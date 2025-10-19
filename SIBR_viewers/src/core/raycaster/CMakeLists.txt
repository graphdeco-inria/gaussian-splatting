# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


project(sibr_raycaster)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${LIBS_SOURCES})

include_directories(
	${Boost_INCLUDE_DIRS}
	${nanoflann_INCLUDE_DIRS}
)
if(WIN32)
target_link_libraries(${PROJECT_NAME}
	OpenMP::OpenMP_CXX
	embree3
	sibr_graphics
	sibr_assets
	nanoflann
)
else()
target_link_libraries(${PROJECT_NAME}
	OpenMP::OpenMP_CXX
	embree
# CLUSTER
#	/data/graphdeco/share/tbb/lib64/libtbb.so
	sibr_graphics
	sibr_assets
	nanoflann
)
endif()

add_definitions( -DSIBR_RAYCASTER_EXPORTS -DBOOST_ALL_DYN_LINK  )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER ${SIBR_FOLDER})

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
)
