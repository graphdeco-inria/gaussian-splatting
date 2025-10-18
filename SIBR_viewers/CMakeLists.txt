# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr

CMAKE_MINIMUM_REQUIRED(VERSION 3.22)

set (CMAKE_SYSTEM_VERSION 10.0.15063.0 CACHE INTERNAL "Cmake system version" FORCE)
PROJECT(sibr_projects)

set(REQUIRED_VERSION "3.22.0")
set(CHECKED_VERSION "3.27.0")

if (CMAKE_VERSION VERSION_LESS REQUIRED_VERSION)
    message(WARNING "Deprecated version of cmake. Please update to at least ${REQUIRED_VERSION} (${CHECKED_VERSION} recommended).")
elseif (CMAKE_VERSION VERSION_GREATER CHECKED_VERSION)
	message(WARNING "Untested version of cmake. If you checked everything is working properly, please update ${CHECKED_VERSION} in the main CmakeLists.txt with the version you tested.")
endif()

## Include cmake stuff (functions/macros) : Modules files
if(WIN32)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/windows)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/windows/Modules)
else()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/linux)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/linux/Modules)
endif()
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

## To maintain cmake versions compatibilities
include(cmake_policies)
setPolicies()

include(git_describe)
git_describe(GIT_BRANCH SIBR_CORE_BRANCH GIT_COMMIT_HASH SIBR_CORE_COMMIT_HASH GIT_TAG SIBR_CORE_TAG GIT_VERSION SIBR_CORE_VERSION)

message(STATUS "SIBR version :\n BRANCH ${SIBR_CORE_BRANCH}\n COMMIT_HASH ${SIBR_CORE_COMMIT_HASH}\n TAG ${SIBR_CORE_TAG}\n VERSION ${SIBR_CORE_VERSION}")

if(NOT WIN32)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()


if (WIN32)
	## Allow C++11 + other flags
	include(CheckCXXCompilerFlag)
	get_filename_component(currentBuildTool ${CMAKE_BUILD_TOOL} NAME_WE)	# tool that can launch the native build system. returned value may be the full path
	if(${currentBuildTool} MATCHES "(msdev|devenv|nmake|MSBuild)")

		add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W3;/DNOMINMAX;/MP;-D_USE_MATH_DEFINES>")
	    #add_definitions(/W3 /DNOMINMAX /MP -D_USE_MATH_DEFINES)# /D_ITERATOR_DEBUG_LEVEL=1 because you need all external DLl to compile with this flag too
	    set(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo;Release;Debug" CACHE STRING "" FORCE)
	    set(CMAKE_CXX_STANDARD 14)
		set(CMAKE_CXX_STANDARD_REQUIRED ON)
		set(CMAKE_CXX_EXTENSIONS OFF)
	elseif(${currentBuildTool} MATCHES "(make|gmake)")
	    add_definitions("-Wall -Wno-unknown-pragmas -Wno-sign-compare -g -std=c++14 -D__forceinline=\"inline\ __attribute__((always_inline))\"")
		# CHECK_CXX_COMPILER_FLAG("-std=gnu++11" COMPILER_SUPPORTS_CXX11)
		# CHECK_CXX_COMPILER_FLAG("-std=gnu++0x" COMPILER_SUPPORTS_CXX0X)
		# if(COMPILER_SUPPORTS_CXX11)
		# 	add_definitions(-std=gnu++11)
		# elseif(COMPILER_SUPPORTS_CXX0X)
		# 	add_definitions(-std=gnu++0x)
		# else()
		# 	message(SEND_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
		# endif()
	elseif(APPLE) ## \todo TODO: do a better test and send error on unsupported c++14 compiler
	    add_definitions(-std=c++14 -stdlib=libc++)
	endif()
else()
	## Allow C++11 + other flags
	include(CheckCXXCompilerFlag)
	get_filename_component(currentBuildTool ${CMAKE_BUILD_TOOL} NAME_WE)	# tool that can launch the native build system. returned value may be the full path
	if(${currentBuildTool} MATCHES "(msdev|devenv|nmake|MSBuild)")

		add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W3;/DNOMINMAX;/MP;-D_USE_MATH_DEFINES>")
	    #add_definitions(/W3 /DNOMINMAX /MP -D_USE_MATH_DEFINES)# /D_ITERATOR_DEBUG_LEVEL=1 because you need all external DLl to compile with this flag too
	    set(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo;Release;Debug" CACHE STRING "" FORCE)
	    set(CMAKE_CXX_STANDARD 14)
		set(CMAKE_CXX_STANDARD_REQUIRED ON)
		set(CMAKE_CXX_EXTENSIONS OFF)
	elseif(${currentBuildTool} MATCHES "(make|gmake|ninja)")
		add_definitions("-fpermissive -fPIC -Wall -Wno-unknown-pragmas -Wno-sign-compare -g -std=c++17 -D__forceinline=\"inline\ __attribute__((always_inline))\"")
	elseif(APPLE) ## \todo TODO: do a better test and send error on unsupported c++14 compiler
	    add_definitions(-std=c++17 -stdlib=libc++)
	endif()
endif()

set(INSTALL_STANDALONE ON)

## Set default build output binaries (used also in sub CMakeLists.txt) :
set(BIN_BUILT_DIR "bin")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(ARCHI_BUILT_DIR "x64")
	set(LIB_BUILT_DIR "lib64")
else()
	set(ARCHI_BUILT_DIR "x86")
	set(LIB_BUILT_DIR "lib")
endif()

option(SEPARATE_CONFIGURATIONS "Clearer separation between configurations" OFF)
SET(CMAKE_INSTALL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/install)
SET(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_ROOT})

if(DEFINED CMAKE_BUILD_TYPE) ## for mono config type (make/nmake/ninja based)
	if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
		set(CMAKE_DEBUG_POSTFIX "_d")
	elseif(${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")
		set(CMAKE_RELWITHDEBINFO_POSTFIX "_rwdi")
	elseif(${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
		set(CMAKE_MINSIZEREL_POSTFIX "_msr")
	elseif(${CMAKE_BUILD_TYPE} MATCHES "Release")
		set(CMAKE_RELEASE_POSTFIX "")
	endif()

	if(SEPARATE_CONFIGURATIONS)
		SET(CMAKE_INSTALL_PREFIX_${CMAKE_BUILD_TYPE} 	${CMAKE_INSTALL_ROOT}/${CMAKE_BUILD_TYPE})
	else()
		SET(CMAKE_INSTALL_PREFIX_${CMAKE_BUILD_TYPE} 	${CMAKE_INSTALL_ROOT})
	endif()

	MESSAGE(STATUS "Install path set to ${CMAKE_INSTALL_PREFIX}.")
	SET(CMAKE_OUTPUT_LIB_${CMAKE_BUILD_TYPE} 	${CMAKE_INSTALL_PREFIX_${CMAKE_BUILD_TYPE}}/lib)
	SET(CMAKE_OUTPUT_BIN_${CMAKE_BUILD_TYPE} 	${CMAKE_INSTALL_PREFIX_${CMAKE_BUILD_TYPE}}/bin)

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE} 	${CMAKE_OUTPUT_LIB_${CMAKE_BUILD_TYPE}})
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE} 	${CMAKE_OUTPUT_LIB_${CMAKE_BUILD_TYPE}})
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE} 	${CMAKE_OUTPUT_BIN_${CMAKE_BUILD_TYPE}})
	set(CMAKE_PDB_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE} 		${CMAKE_OUTPUT_BIN_${CMAKE_BUILD_TYPE}})
endif()
foreach(CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES}) ## for multi config types (MSVC based)
	string(TOUPPER ${CONFIG_TYPES} CONFIG_TYPES_UC)
	if(${CONFIG_TYPES} MATCHES "Debug")
		set(CMAKE_DEBUG_POSTFIX "_d")
	elseif(${CONFIG_TYPES} MATCHES "RelWithDebInfo")
		set(CMAKE_RELWITHDEBINFO_POSTFIX "_rwdi")
	elseif(${CONFIG_TYPES} MATCHES "MinSizeRel")
		set(CMAKE_MINSIZEREL_POSTFIX "_msr")
	elseif(${CMAKE_BUILD_TYPE} MATCHES "Release")
		set(CMAKE_RELEASE_POSTFIX "")
	endif()

	if(SEPARATE_CONFIGURATIONS)
		SET(CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC} ${CMAKE_INSTALL_ROOT}/${CONFIG_TYPES})
	else()
		SET(CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC} ${CMAKE_INSTALL_ROOT})
	endif()
	
	MESSAGE(STATUS "Install path for ${CONFIG_TYPES} set to ${CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC}}.")
	SET(CMAKE_OUTPUT_LIB_${CONFIG_TYPES_UC} 	${CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC}}/lib)
	SET(CMAKE_OUTPUT_BIN_${CONFIG_TYPES_UC} 	${CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC}}/bin)

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC} 	${CMAKE_OUTPUT_LIB_${CONFIG_TYPES_UC}})
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC} 	${CMAKE_OUTPUT_LIB_${CONFIG_TYPES_UC}})
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC} 	${CMAKE_OUTPUT_BIN_${CONFIG_TYPES_UC}})
	set(CMAKE_PDB_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC} 		${CMAKE_OUTPUT_BIN_${CONFIG_TYPES_UC}})
endforeach()


# Settings for RPATH
if (NOT WIN32)
	# Default config of Fedora at INRIA has no LD_LIBRARY_PATH (for security reasons I guess)
	# So at least I had "./" in RPATH and found link paths
	#set(CMAKE_SKIP_RPATH TRUE)
	#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
	SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

	SET(CMAKE_INSTALL_RPATH "$ORIGIN")
	#SET(CMAKE_INSTALL_RPATH "./")
	#SET(CMAKE_INSTALL_RPATH "./:/usr/lib64/:/usr/lib/:/usr/local/lib64/:/usr/local/lib/") # This one causes be a problem -> a "default" version of libGL (swrast) is located in /usr/lib64 and was selected instead of nvidia one (in /usr/lib64/nividia)

	SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()


set(SIBR_PROGRAMARGS "" CACHE STRING "Default program arguments used in Visual Studio target properties")
if ("${SIBR_PROGRAMARGS}" STREQUAL "")
  if (DEFINED ENV{SIBR_PROGRAMARGS})
    set(SIBR_PROGRAMARGS "$ENV{SIBR_PROGRAMARGS}" CACHE STRING "Default program arguments used in Visual Studio target properties" FORCE)
    message( STATUS "Using program options found in environment variable 'SIBR_PROGRAMARGS' => '${SIBR_PROGRAMARGS}'")
  else()
    message(
      "Note you can provide default program options for Visual Studio target properties by either setting"
      " a value for the cmake cached variable 'SIBR_PROGRAMARGS' or by setting a new environment "
      "variable 'SIBR_PROGRAMARGS'")
  endif()
endif()

add_custom_target(PREBUILD ALL)

## Include all projects
set(SIBR_PROJECTS_SAMPLES_SUBPAGE_REF "")
set(SIBR_PROJECTS_OURS_SUBPAGE_REF "")
set(SIBR_PROJECTS_TOOLBOX_SUBPAGE_REF "")
set(SIBR_PROJECTS_OTHERS_SUBPAGE_REF "")
set(SIBR_PROJECTS_SAMPLES_REF_REF "")
set(SIBR_PROJECTS_OURS_REF_REF "")
set(SIBR_PROJECTS_TOOLBOX_REF_REF "")
set(SIBR_PROJECTS_OTHERS_REF_REF "")
set(DOXY_APP_SPECIFIC_IMG_PATH "")
set(DOXY_DOC_EXCLUDE_PATTERNS_DIRS "")
ADD_SUBDIRECTORY(src)


## handle documentation
if (WIN32)
ADD_SUBDIRECTORY(docs)
endif()
