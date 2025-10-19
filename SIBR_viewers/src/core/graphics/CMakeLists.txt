# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


project(sibr_graphics)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

file(GLOB RESOURCES "resources/*.ini")
source_group("Resources Files" FILES ${RESOURCES})

## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(
	${Boost_INCLUDE_DIRS}
	${imgui_INCLUDE_DIRS}
)
if(WIN32)
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	OpenMP::OpenMP_CXX
	imgui
	glfw3
	sibr_system
)
else()
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	OpenMP::OpenMP_CXX
	imgui
	${GLFW_LIBRARY}
	sibr_system
)
endif()

if (NOT WIN32)
	target_link_libraries(${PROJECT_NAME}
 		#GLEW
 		rt m dl X11 pthread Xrandr Xinerama Xxf86vm Xcursor
		# X11 Xi Xrandr Xxf86vm Xinerama Xcursor dl rt m pthread
	)
endif()

add_definitions(-DSIBR_GRAPHICS_EXPORTS -DIMGUI_EXPORTS -DBOOST_ALL_DYN_LINK)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER ${SIBR_FOLDER})


## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
	INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
	RESOURCES  ${RESOURCES}
	RSC_FOLDER "core"
)
