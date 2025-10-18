# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


project(sibr_imgproc)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(
	${Boost_INCLUDE_DIRS}
	${mrf_INCLUDE_DIRS}
)
if(WIN32)
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	OpenMP::OpenMP_CXX
	glfw3
	mrf
	sibr_system
	sibr_graphics
	sibr_assets
	sibr_raycaster
)
else()
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	OpenMP::OpenMP_CXX
	${GLFW_LIBRARY}
	mrf
	sibr_system
	sibr_graphics
	sibr_assets
	sibr_raycaster
)
endif()

if (NOT WIN32)
	target_link_libraries(${PROJECT_NAME}
 		# GLEW
 		rt m dl X11 pthread Xrandr Xinerama Xxf86vm Xcursor
		# X11 Xi Xrandr Xxf86vm Xinerama Xcursor dl rt m pthread
	)
endif()

add_definitions( -DSIBR_IMGPROC_EXPORTS -DBOOST_ALL_DYN_LINK  )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER ${SIBR_FOLDER})


## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
)
