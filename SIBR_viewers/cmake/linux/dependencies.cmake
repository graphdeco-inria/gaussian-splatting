# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


## Included once for all sub project.
## It contain the whole cmake instructions to find necessary common dependencies.
## 3rdParty (provided by sibr_addlibrary win3rdParty or from external packages) are then available in cmake sub projects.
##
## Do not include this file more than once but you can modify it to fit to your own project.
## So please, read it carefully because you can use on of these dependencies for your project or appen new one.
##
## As it is included after camke options, you can use conditional if(<CMAKE_PROJ_OPT>)/endif() to encapsulate your 3rdParty.
##

## win3rdParty function allowing to auto check/download/update binaries dependencies for current windows compiler
## Please open this file in order to get more documentation and usage examples.
include(Win3rdParty)

include(sibr_library)

Win3rdPartyGlobalCacheAction()

find_package(OpenGL REQUIRED)

set(OpenGL_GL_PREFERENCE "GLVND")

############
## Find GLEW
############
##for headless rendering
find_package(EGL QUIET)

if(EGL_FOUND)
    add_definitions(-DGLEW_EGL)
    message("Activating EGL support for headless GLFW/GLEW")
else()
    message("EGL not found : EGL support for headless GLFW/GLEW is disabled")
endif()

if (MSVC11 OR MSVC12)
    set(glew_multiset_arguments 
            CHECK_CACHED_VAR GLEW_INCLUDE_DIR	    PATH "glew-1.10.0/include" DOC "default empty doc"
            CHECK_CACHED_VAR GLEW_LIBRARIES         STRING LIST "debug;glew-1.10.0/${LIB_BUILT_DIR}/glew32d.lib;optimized;glew-1.10.0/${LIB_BUILT_DIR}/glew32.lib" DOC "default empty doc"
        )
elseif (MSVC14 OR MSVC17)
    set(glew_multiset_arguments 
            CHECK_CACHED_VAR GLEW_INCLUDE_DIR	    PATH "glew-2.0.0/include" DOC "default empty doc"
            CHECK_CACHED_VAR GLEW_SHARED_LIBRARY_RELEASE       PATH "glew-2.0.0/${LIB_BUILT_DIR}/glew32.lib"
            CHECK_CACHED_VAR GLEW_STATIC_LIBRARY_RELEASE       PATH "glew-2.0.0/${LIB_BUILT_DIR}/glew32s.lib"
            CHECK_CACHED_VAR GLEW_SHARED_LIBRARY_DEBUG         PATH "glew-2.0.0/${LIB_BUILT_DIR}/glew32d.lib"
            CHECK_CACHED_VAR GLEW_STATIC_LIBRARY_DEBUG         PATH "glew-2.0.0/${LIB_BUILT_DIR}/glew32sd.lib"
        )
else ()
    message("There is no provided GLEW library for your compiler, relying on find_package to find it")
endif()
sibr_addlibrary(NAME GLEW #VERBOSE ON
    MSVC11 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/glew-1.10.0.7z"
    MSVC12 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/glew-1.10.0.7z"
    MSVC14 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC15-splitted%20version/glew-2.0.0.7z"        # using recompiled version of glew
    MULTI_SET ${glew_multiset_arguments}
)
set(GLEW_VERBOSE ON)
FIND_PACKAGE(GLEW REQUIRED)
IF(GLEW_FOUND)
    INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})
ELSE(GLEW_FOUND)
    MESSAGE("GLEW not found. Set GLEW_DIR to base directory of GLEW.")
ENDIF(GLEW_FOUND)


##############
## Find ASSIMP
##############
if (MSVC11 OR MSVC12)
    set(assimp_set_arguments 
        CHECK_CACHED_VAR ASSIMP_DIR PATH "Assimp_3.1_fix"
    )
elseif (MSVC14 OR MSVC17)
    set(assimp_set_arguments 
        CHECK_CACHED_VAR ASSIMP_DIR PATH "Assimp-4.1.0"
    )
else ()
    message("There is no provided ASSIMP library for your compiler, relying on find_package to find it")
endif()

sibr_addlibrary(NAME ASSIMP #VERBOSE ON
        MSVC11 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/Assimp_3.1_fix.7z"
        MSVC12 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/Assimp_3.1_fix.7z"
        MSVC14 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC15-splitted%20version/Assimp-4.1.0.7z"
        MULTI_SET
            ${assimp_set_arguments}
)

find_package(ASSIMP REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIR})

################
## Find FFMPEG
################
sibr_addlibrary(NAME FFMPEG
    MSVC11 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/ffmpeg.zip"
    MSVC12 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/ffmpeg.zip"
    MSVC14 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC15-splitted%20version/ffmpeg-4.0.2-win64-win3rdParty.7z"
    SET CHECK_CACHED_VAR FFMPEG_DIR PATH ${FFMPEG_WIN3RDPARTY_DIR}
)
find_package(FFMPEG)
include_directories(${FFMPEG_INCLUDE_DIR})
## COMMENT OUT ALL FFMPEG FOR CLUSTER

###################
## Find embree3
###################
sibr_addlibrary(
    NAME embree3
    MSVC11 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/embree2.7.0.x64.windows.7z"
    MSVC14 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC15-splitted%20version/embree-3.6.1.x64.vc14.windows.7z"     # TODO SV: provide a valid version if required
)

# CLUSTER
#find_package(embree 3.0 REQUIRED PATHS "/data/graphdeco/share/embree/usr/local/lib64/cmake/" )
find_package(embree 3.0 )

###################
## Find eigen3
###################
sibr_addlibrary(
	NAME eigen3
	#MSVC11 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/eigen-eigen-dc6cfdf9bcec.7z"
	#MSVC14 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/eigen-eigen-dc6cfdf9bcec.7z"    # TODO SV: provide a valid version if required
    MSVC11 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC15-splitted%20version/eigen3.7z"
    MSVC14 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC15-splitted%20version/eigen3.7z"
    SET CHECK_CACHED_VAR eigen3_DIR PATH "eigen/share/eigen3/cmake"
)
include_directories(/usr/include/eigen3)
add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)

#############
## Find Boost
#############
set(Boost_REQUIRED_COMPONENTS "system;chrono;filesystem;date_time" CACHE INTERNAL "Boost Required Components")

if (WIN32)
    # boost multiset arguments
    if (MSVC11 OR MSVC12)
        set(boost_multiset_arguments 
                CHECK_CACHED_VAR BOOST_ROOT                 PATH "boost_1_55_0"
                CHECK_CACHED_VAR BOOST_INCLUDEDIR 		    PATH "boost_1_55_0"
                CHECK_CACHED_VAR BOOST_LIBRARYDIR 		    PATH "boost_1_55_0/${LIB_BUILT_DIR}"
                #CHECK_CACHED_VAR Boost_COMPILER             STRING "-${Boost_WIN3RDPARTY_VCID}" DOC "vcid (eg: -vc110 for MSVC11)"
                CHECK_CACHED_VAR Boost_COMPILER             STRING "-vc110" DOC "vcid (eg: -vc110 for MSVC11)" # NOTE: if it doesnt work, uncomment this option and set the right value for VisualC id
            )
    elseif (MSVC14 OR MSVC17)
        set(boost_multiset_arguments 
                CHECK_CACHED_VAR BOOST_ROOT                 PATH "boost-1.71"
                CHECK_CACHED_VAR BOOST_INCLUDEDIR 		    PATH "boost-1.71"
                CHECK_CACHED_VAR BOOST_LIBRARYDIR 		    PATH "boost-1.71/${LIB_BUILT_DIR}"
                CHECK_CACHED_VAR Boost_COMPILER             STRING "-vc141" DOC "vcid (eg: -vc110 for MSVC11)" # NOTE: if it doesnt work, uncomment this option and set the right value for VisualC id
            )
        
        option(BOOST_MINIMAL_VERSION "Only get minimal Boost dependencies" ON)

        if(${BOOST_MINIMAL_VERSION})
            set(BOOST_MSVC14_ZIP "boost-1.71-ibr-minimal.7z")
        else()
            set(BOOST_MSVC14_ZIP "boost-1.71.7z")
        endif()
    else ()
        message("There is no provided Boost library for your compiler, relying on find_package to find it")
    endif()

    sibr_addlibrary(NAME Boost VCID TIMEOUT 600 #VERBOSE ON
        MSVC11 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/boost_1_55_0.7z"
        MSVC12 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC11-splitted%20version/boost_1_55_0.7z"
        MSVC14 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC15-splitted%20version/${BOOST_MSVC14_ZIP}"    # boost compatible with msvc14
        MULTI_SET ${boost_multiset_arguments}
            CHECK_CACHED_VAR Boost_NO_SYSTEM_PATHS      BOOL ON DOC "Set to ON to disable searching in locations not specified by these boost cached hint variables"
            CHECK_CACHED_VAR Boost_NO_BOOST_CMAKE       BOOL ON DOC "Set to ON to disable the search for boost-cmake (package cmake config file if boost was built with cmake)"
    )
    if(NOT Boost_COMPILER AND Boost_WIN3RDPARTY_USE)
        message(WARNING "Boost_COMPILER is not set and it's needed.")
    endif()
endif()

find_package(Boost 1.65.0 REQUIRED COMPONENTS ${Boost_REQUIRED_COMPONENTS})
# for CLUSTER
##find_package(Boost 1.58.0 REQUIRED COMPONENTS ${Boost_REQUIRED_COMPONENTS})


if(WIN32)
	add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
    #add_definitions(/EHsc)
endif()

if(Boost_LIB_DIAGNOSTIC_DEFINITIONS)
    add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
endif()

#if(WIN32)
    add_definitions(-DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)
#endif()

include_directories(${BOOST_INCLUDEDIR} ${Boost_INCLUDE_DIRS})
link_directories(${BOOST_LIBRARYDIR} ${Boost_LIBRARY_DIRS})


##############
## Find OpenMP
##############
find_package(OpenMP)

##############
## Find OpenCV
##############
if (WIN32)
	if (${MSVC_TOOLSET_VERSION} EQUAL 143)
		MESSAGE("SPECIAL OPENCV HANDLING")
		set(opencv_set_arguments 
		CHECK_CACHED_VAR OpenCV_DIR PATH "install" ## see OpenCVConfig.cmake
	    )
	elseif (MSVC11 OR MSVC12)
	    set(opencv_set_arguments 
		CHECK_CACHED_VAR OpenCV_DIR PATH "opencv/build" ## see OpenCVConfig.cmake
	    )
	elseif (MSVC14)
	    set(opencv_set_arguments 
		CHECK_CACHED_VAR OpenCV_DIR PATH "opencv-4.5.0/build" ## see OpenCVConfig.cmake
	    )
	else ()
	    message("There is no provided OpenCV library for your compiler, relying on find_package to find it")
	endif()
else()
	    message("There is no provided OpenCV library for your compiler, relying on find_package to find it")
endif()

sibr_addlibrary(NAME OpenCV #VERBOSE ON
        MSVC11 "https://repo-sam.inria.fr/fungraph/dependencies/sibr/~0.9/opencv.7z"
        MSVC12 "https://repo-sam.inria.fr/fungraph/dependencies/sibr/~0.9/opencv.7z"
        MSVC14 "https://repo-sam.inria.fr/fungraph/dependencies/sibr/~0.9/opencv-4.5.0.7z"    # opencv compatible with msvc14 and with contribs
        MSVC17 "https://repo-sam.inria.fr/fungraph/dependencies/sibr/~0.9/opencv4-8.7z" 
		SET ${opencv_set_arguments}
    )
find_package(OpenCV 4.5 REQUIRED) ## Use directly the OpenCVConfig.cmake provided
## FOR CLUSTER
###find_package(OpenCV 4.5 REQUIRED PATHS "/data/graphdeco/share/opencv/usr/local/lib64/cmake/opencv4/" ) ## Use directly the OpenCVConfig.cmake provided

    ##https://stackoverflow.com/questions/24262081/cmake-relwithdebinfo-links-to-debug-libs
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

add_definitions(-DOPENCV_TRAITS_ENABLE_DEPRECATED) 

if(OpenCV_INCLUDE_DIRS)
    foreach(inc ${OpenCV_INCLUDE_DIRS})
        if(NOT EXISTS ${inc})
            set(OpenCV_INCLUDE_DIR "" CACHE PATH "additional custom include DIR (in case of trouble to find it (fedora 17 opencv package))")
        endif()
    endforeach()
    if(OpenCV_INCLUDE_DIR)
        list(APPEND OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIR})
        include_directories(${OpenCV_INCLUDE_DIRS})
    endif()
endif()

###################
## Find GLFW
###################
sibr_addlibrary(
    NAME glfw3
    MSVC11 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC15-splitted%20version/glfw-3.2.1.7z"
    MSVC14 "https://repo-sam.inria.fr/fungraph/dependencies/ibr-common/win3rdParty-MSVC15-splitted%20version/glfw-3.2.1.7z"     # TODO SV: provide a valid version if required
    SET CHECK_CACHED_VAR glfw3_DIR PATH "glfw-3.2.1"
)

### FOR CLUSTER COMMENT OUT lines above, uncomment lines below
##find_package(GLFW REQUIRED 3.3 )
##message("***********=============> GLFW IS " ${GLFW_LIBRARY})
##message("***********=============> GLFW IS " ${GLFW_LIBRARIES})

find_package(glfw3 REQUIRED)

sibr_gitlibrary(TARGET imgui
    GIT_REPOSITORY 	"https://gitlab.inria.fr/sibr/libs/imgui.git"
    GIT_TAG			"741fb3ab6c7e1f7cef23ad0501a06b7c2b354944"
)

## FOR CLUSTER COMMENT OUT nativefiledialog
sibr_gitlibrary(TARGET nativefiledialog
    GIT_REPOSITORY 	"https://gitlab.inria.fr/sibr/libs/nativefiledialog.git"
    GIT_TAG			"ae2fab73cf44bebdc08d997e307c8df30bb9acec"
)


sibr_gitlibrary(TARGET mrf
    GIT_REPOSITORY 	"https://gitlab.inria.fr/sibr/libs/mrf.git"
    GIT_TAG			"30c3c9494a00b6346d72a9e37761824c6f2b7207"
)

sibr_gitlibrary(TARGET nanoflann
    GIT_REPOSITORY 	"https://gitlab.inria.fr/sibr/libs/nanoflann.git"
    GIT_TAG			"7a20a9ac0a1d34850fc3a9e398fc4a7618e8a69a"
)

sibr_gitlibrary(TARGET picojson
    GIT_REPOSITORY 	"https://gitlab.inria.fr/sibr/libs/picojson.git"
    GIT_TAG			"7cf8feee93c8383dddbcb6b64cf40b04e007c49f"
)

sibr_gitlibrary(TARGET rapidxml
    GIT_REPOSITORY 	"https://gitlab.inria.fr/sibr/libs/rapidxml.git"
    GIT_TAG			"069e87f5ec5ce1745253bd64d89644d6b894e516"
)

sibr_gitlibrary(TARGET xatlas
    GIT_REPOSITORY 	"https://gitlab.inria.fr/sibr/libs/xatlas.git"
    GIT_TAG			"0fbe06a5368da13fcdc3ee48d4bdb2919ed2a249"
    INCLUDE_DIRS 	"source/xatlas"
)

Win3rdPartyGlobalCacheAction()
