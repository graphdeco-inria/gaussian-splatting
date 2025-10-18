# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


## Try to find the ASSIMP library
## Once done this will define
##
##  	ASSIMP_FOUND 		- system has ASSIMP
##  	ASSIMP_INCLUDE_DIR 	- The ASSIMP include directory
##  	ASSIMP_LIBRARIES 	- The libraries needed to use ASSIMP
##  	ASSIMP_CMD 			- the full path of ASSIMP executable
##	ASSIMP_DYNAMIC_LIB	- the Assimp dynamic lib (available only on windows as .dll file for the moment)
##
## Edited for using a bugfixed version of Assimp

if(NOT ASSIMP_DIR)
    set(ASSIMP_DIR "$ENV{ASSIMP_DIR}" CACHE PATH "ASSIMP root directory")
endif()
if(ASSIMP_DIR)
	file(TO_CMAKE_PATH ${ASSIMP_DIR} ASSIMP_DIR)
endif()


## set the LIB POSTFIX to find in a right directory according to what kind of compiler we use (32/64bits)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(ASSIMP_SEARCH_LIB "lib64")
	set(ASSIMP_SEARCH_BIN "bin64")
	set(ASSIMP_SEARCH_LIB_PATHSUFFIXE "x64")
else()
	set(ASSIMP_SEARCH_LIB "lib32")
	set(ASSIMP_SEARCH_BIN "bin32")
	set(ASSIMP_SEARCH_LIB_PATHSUFFIXE "x86")
endif()

set(PROGRAMFILESx86 "PROGRAMFILES(x86)")


FIND_PATH(ASSIMP_INCLUDE_DIR
	NAMES assimp/config.h
	PATHS
		${ASSIMP_DIR}
		## linux
		/usr
		/usr/local
		/opt/local
		## windows
		"$ENV{PROGRAMFILES}/Assimp"
		"$ENV{${PROGRAMFILESx86}}/Assimp"
		"$ENV{ProgramW6432}/Assimp"
	PATH_SUFFIXES include
)


FIND_LIBRARY(ASSIMP_LIBRARY
	NAMES assimp-vc140-mt
	PATHS
		${ASSIMP_DIR}/${ASSIMP_SEARCH_LIB}
		${ASSIMP_DIR}/lib
		${ASSIMP_DIR}/lib64
		## linux
		/usr/${ASSIMP_SEARCH_LIB}
		/usr/local/${ASSIMP_SEARCH_LIB}
		/opt/local/${ASSIMP_SEARCH_LIB}
		/usr/lib
		/usr/local/lib
		/opt/local/lib
		## windows
		"$ENV{PROGRAMFILES}/Assimp/${ASSIMP_SEARCH_LIB}"
		"$ENV{${PROGRAMFILESx86}}/Assimp/${ASSIMP_SEARCH_LIB}"
		"$ENV{ProgramW6432}/Assimp/${ASSIMP_SEARCH_LIB}"
		"$ENV{PROGRAMFILES}/Assimp/lib"
		"$ENV{${PROGRAMFILESx86}}/Assimp/lib"
		"$ENV{ProgramW6432}/Assimp/lib"
	PATH_SUFFIXES ${ASSIMP_SEARCH_LIB_PATHSUFFIXE}
)
set(ASSIMP_LIBRARIES ${ASSIMP_LIBRARY})


if(ASSIMP_LIBRARY)
	get_filename_component(ASSIMP_LIBRARY_DIR ${ASSIMP_LIBRARY} PATH)
	file(GLOB ASSIMP_DYNAMIC_LIB "${ASSIMP_LIBRARY_DIR}/assimp*.dll")
	if(NOT ASSIMP_DYNAMIC_LIB)
		message("ASSIMP_DYNAMIC_LIB is missing... at ${ASSIMP_LIBRARY_DIR}")
	endif()
	set(ASSIMP_DYNAMIC_LIB ${ASSIMP_DYNAMIC_LIB} CACHE PATH "Windows dll location")
endif()

MARK_AS_ADVANCED(ASSIMP_DYNAMIC_LIB ASSIMP_INCLUDE_DIR ASSIMP_LIBRARIES)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ASSIMP
	REQUIRED_VARS ASSIMP_INCLUDE_DIR ASSIMP_LIBRARIES
	FAIL_MESSAGE "ASSIMP wasn't found correctly. Set ASSIMP_DIR to the root SDK installation directory."
)

if(NOT ASSIMP_FOUND)
	set(ASSIMP_DIR "" CACHE STRING "Path to ASSIMP install directory")
endif()
