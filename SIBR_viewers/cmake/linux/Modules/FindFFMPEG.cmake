# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


## Try to find the FFMPEG library
## Once done this will define
##
##  	FFMPEG_FOUND 		- system has FFmpeg
##  	FFMPEG_INCLUDE_DIR 	- The FFmpeg include directory
##  	FFMPEG_LIBRARIES 	- The libraries needed to use FFmpeg
##		FFMPEG_DYNAMIC_LIBS	- DLLs for windows


if(NOT FFMPEG_DIR)
    set(FFMPEG_DIR "$ENV{FFMPEG_DIR}" CACHE PATH "FFMPEG_DIR root directory")
endif()

if(FFMPEG_DIR)
	file(TO_CMAKE_PATH ${FFMPEG_DIR} FFMPEG_DIR)
endif()

MACRO(FFMPEG_FIND varname shortname headername)
	
	# Path to include dirs
	FIND_PATH(FFMPEG_${varname}_INCLUDE_DIRS 
		NAMES "lib${shortname}/${headername}" 
		PATHS
			"${FFMPEG_DIR}/include" # modify this to adapt according to OS/compiler	
			"/usr/include"
			"/usr/include/ffmpeg"		
	)
		
	#Add libraries
	IF(${FFMPEG_${varname}_INCLUDE_DIRS} STREQUAL "FFMPEG_${varname}_INCLUDE_DIR-NOTFOUND")
		MESSAGE(STATUS "Can't find includes for ${shortname}...")
	ELSE()
		FIND_LIBRARY(FFMPEG_${varname}_LIBRARIES
			NAMES ${shortname}
			PATHS
				${FFMPEG_DIR}/lib
				"/usr/lib"
				"/usr/lib64"
				"/usr/local/lib"
				"/usr/local/lib64"
		)

		# set libraries and other variables
		SET(FFMPEG_${varname}_FOUND 1)
		SET(FFMPEG_${varname}_INCLUDE_DIRS ${FFMPEG_${varname}_INCLUDE_DIR})
		SET(FFMPEG_${varname}_LIBS ${FFMPEG_${varname}_LIBRARIES})
	ENDIF()
 ENDMACRO(FFMPEG_FIND)

#Calls to ffmpeg_find to get  librarires ------------------------------
FFMPEG_FIND(LIBAVFORMAT avformat avformat.h)
FFMPEG_FIND(LIBAVDEVICE avdevice avdevice.h)
FFMPEG_FIND(LIBAVCODEC  avcodec  avcodec.h)
FFMPEG_FIND(LIBAVUTIL   avutil   avutil.h)
FFMPEG_FIND(LIBSWSCALE  swscale  swscale.h)
 
# check if libs are found and set FFMPEG related variables
#SET(FFMPEG_FOUND "NO")
IF(FFMPEG_LIBAVFORMAT_FOUND 
	AND FFMPEG_LIBAVDEVICE_FOUND 
	AND FFMPEG_LIBAVCODEC_FOUND 
	AND FFMPEG_LIBAVUTIL_FOUND 
	AND FFMPEG_LIBSWSCALE_FOUND)
 
	# All ffmpeg libs are here
    SET(FFMPEG_FOUND "YES")
	SET(FFMPEG_INCLUDE_DIR ${FFMPEG_LIBAVFORMAT_INCLUDE_DIRS})
	SET(FFMPEG_LIBRARY_DIRS ${FFMPEG_LIBAVFORMAT_LIBRARY_DIRS})
	SET(FFMPEG_LIBRARIES
        ${FFMPEG_LIBAVFORMAT_LIBS}
        ${FFMPEG_LIBAVDEVICE_LIBS}
        ${FFMPEG_LIBAVCODEC_LIBS}
        ${FFMPEG_LIBAVUTIL_LIBS}
		${FFMPEG_LIBSWSCALE_LIBS}	)
		
	# add dynamic libraries
	if(WIN32)
		file(GLOB FFMPEG_DYNAMIC_LIBS "${FFMPEG_DIR}/bin/*.dll")
		if(NOT FFMPEG_DYNAMIC_LIBS)
			message("FFMPEG_DYNAMIC_LIBS is missing...")
	endif()
	set(FFMPEG_DYNAMIC_LIBS ${FFMPEG_DYNAMIC_LIBS} CACHE PATH "Windows dll location")
endif()
	
	mark_as_advanced(FFMPEG_INCLUDE_DIR FFMPEG_LIBRARY_DIRS FFMPEG_LIBRARIES FFMPEG_DYNAMIC_LIBS)
ELSE ()
    MESSAGE(STATUS "Could not find FFMPEG")
ENDIF()
 
 
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFMPEG
	REQUIRED_VARS FFMPEG_INCLUDE_DIR FFMPEG_LIBRARIES
	FAIL_MESSAGE "FFmpeg wasn't found correctly. Set FFMPEG_DIR to the root SDK installation directory."
)

if(NOT FFMPEG_FOUND)
	set(FFMPEG_DIR "" CACHE STRING "Path to FFmpeg install directory")
endif()
  
