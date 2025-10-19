# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


## Important Note:
## This is not an official Find*cmake. It has been written for searching through
## a custom path (EMBREE_DIR) before checking elsewhere.
##
## FindEMBREE.cmake
## Find EMBREE's includes and library
##
## This module defines :
## 	[in] 	EMBREE_DIR, The base directory to search for EMBREE (as cmake var or env var)
## 	[out] 	EMBREE_INCLUDE_DIR where to find EMBREE.h
## 	[out] 	EMBREE_LIBRARIES, EMBREE_LIBRARY, libraries to link against to use EMBREE
## 	[out] 	EMBREE_FOUND, If false, do not try to use EMBREE.
##


if(NOT EMBREE_DIR)
    set(EMBREE_DIR "$ENV{EMBREE_DIR}" CACHE PATH "EMBREE root directory")
endif()
if(EMBREE_DIR)
	file(TO_CMAKE_PATH ${EMBREE_DIR} EMBREE_DIR)
endif()


## set the LIB POSTFIX to find in a right directory according to what kind of compiler we use (32/64bits)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(EMBREE_SEARCH_LIB "lib64")
	set(EMBREE_SEARCH_BIN "bin64")
	set(EMBREE_SEARCH_LIB_PATHSUFFIXE "x64")
else()
	set(EMBREE_SEARCH_LIB "lib32")
	set(EMBREE_SEARCH_BIN "bin32")
	set(EMBREE_SEARCH_LIB_PATHSUFFIXE "x86")
endif()

set(PROGRAMFILESx86 "PROGRAMFILES(x86)")

FIND_PATH(EMBREE_INCLUDE_DIR
	NAMES embree3/rtcore_geometry.h
	PATHS
		${EMBREE_DIR}
		## linux
		/usr
		/usr/local
		/opt/local
		## windows
		"$ENV{PROGRAMFILES}/EMBREE"
		"$ENV{${PROGRAMFILESx86}}/EMBREE"
		"$ENV{ProgramW6432}/EMBREE"
	PATH_SUFFIXES include
)

FIND_LIBRARY(EMBREE_LIBRARY
	NAMES embree3
	PATHS
		${EMBREE_DIR}/${EMBREE_SEARCH_LIB}
		${EMBREE_DIR}/lib
		## linux
		/usr/${EMBREE_SEARCH_LIB}
		/usr/local/${EMBREE_SEARCH_LIB}
		/opt/local/${EMBREE_SEARCH_LIB}
		/usr/lib
		/usr/local/lib
		/opt/local/lib
		## windows
		"$ENV{PROGRAMFILES}/EMBREE/${EMBREE_SEARCH_LIB}"
		"$ENV{${PROGRAMFILESx86}}/EMBREE/${EMBREE_SEARCH_LIB}"
		"$ENV{ProgramW6432}/EMBREE/${EMBREE_SEARCH_LIB}"
		"$ENV{PROGRAMFILES}/EMBREE/lib"
		"$ENV{${PROGRAMFILESx86}}/EMBREE/lib"
		"$ENV{ProgramW6432}/EMBREE/lib"
	PATH_SUFFIXES ${EMBREE_SEARCH_LIB_PATHSUFFIXE}
)
set(EMBREE_LIBRARIES ${EMBREE_LIBRARY})

MARK_AS_ADVANCED(EMBREE_INCLUDE_DIR EMBREE_LIBRARIES)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(EMBREE
	REQUIRED_VARS EMBREE_INCLUDE_DIR EMBREE_LIBRARIES
	FAIL_MESSAGE "EMBREE wasn't found correctly. Set EMBREE_DIR to the root SDK installation directory."
)

if(NOT EMBREE_FOUND)
	set(EMBREE_DIR "" CACHE STRING "Path to EMBREE install directory")
endif()
