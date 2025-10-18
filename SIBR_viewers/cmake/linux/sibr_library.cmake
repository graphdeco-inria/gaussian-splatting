# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


# NOTE
# This feature is used to easily download, store and link external dependencies. This
# requires to prepare pre-compiled libraries (to download). For now, packages have
# only be prepare for Windows 64-bit with Visual Studio 2012. (You should re-build
# everything if you want to use another version of Visual Studio/ another compiler).

# NOTE ABOUT UNIX SYSTEMS
# There is no need for "searching mechanism". This function is discard and your
# libraries should be installed is the standard folders that are:
#
# /usr/include/
# /usr/lib/
# /usr/lib64/
# for packages downloaded using apt-get/yum
# 
# /usr/local/include/
# /usr/local/lib/
# /usr/local/lib64/
# for packages manually installed ("make install")
#
# if you encounter problems when linking (e.g. lib not found even if it is installed),
# please check these folders are in your search PATH environment variables.

set(EXTLIBS_PACKAGE_FOLDER "${CMAKE_SOURCE_DIR}/extlibs")

function(sibr_addlibrary)
    if(NOT WIN32)
        return()
    endif()

    file(MAKE_DIRECTORY ${EXTLIBS_PACKAGE_FOLDER})
    cmake_parse_arguments(args "VCID" "VERBOSE;TIMEOUT;DEFAULT_USE;NAME;VERSION;MSVC11;MSVC12;MSVC14;MSVC17" "MULTI_SET;SET" ${ARGN})


    if (NOT "${args_VERSION}" MATCHES "")
        message(WARNING "VERSION is not implemented yet")
    endif()

    set(lcname "")
    set(ucname "")
    string(TOLOWER "${args_NAME}" lcname)
    string(TOUPPER "${args_NAME}" ucname)

    set(LIB_PACKAGE_FOLDER "${EXTLIBS_PACKAGE_FOLDER}/${lcname}")
    win3rdParty(${ucname}
                    $<args_VCID:VCID>
                    VERBOSE     ${args_VERBOSE}
                    TIMEOUT     ${args_TIMEOUT}
                    DEFAULT_USE ${args_DEFAULT_USE}
                    MSVC11 "${LIB_PACKAGE_FOLDER}" "${args_MSVC11}"
                    MSVC12 "${LIB_PACKAGE_FOLDER}" "${args_MSVC12}"
                    MSVC14 "${LIB_PACKAGE_FOLDER}" "${args_MSVC14}" # TODO SV: make sure to build this library if required
					MSVC17 "${LIB_PACKAGE_FOLDER}" "${args_MSVC17}"
                    SET         ${args_SET}
                    MULTI_SET   ${args_MULTI_SET}
                )
			
    # Add include/ directory
    # and lib/ directories

    # TODO SV: paths not matching with current hierarchy. example: libraw/libraw-0.17.1/include
    # SR:	The link directories will also be used to lookup for dependency DLLs to copy in the install directory.
    #		Some libraries put the DLLs in the bin/ directory, so we include those.
    file(GLOB subdirs RELATIVE ${LIB_PACKAGE_FOLDER} ${LIB_PACKAGE_FOLDER}/*)
    set(dirlist "")
    foreach(dir ${subdirs})
        if(IS_DIRECTORY ${LIB_PACKAGE_FOLDER}/${dir})
            # message("adding ${LIB_PACKAGE_FOLDER}/${dir}/include/ to the include directories")
            include_directories("${LIB_PACKAGE_FOLDER}/${dir}/include/")
            # message("adding ${LIB_PACKAGE_FOLDER}/${dir}/lib[64] to the link directories")
            link_directories("${LIB_PACKAGE_FOLDER}/${dir}/")
            link_directories("${LIB_PACKAGE_FOLDER}/${dir}/lib/")
            link_directories("${LIB_PACKAGE_FOLDER}/${dir}/lib64/")
            link_directories("${LIB_PACKAGE_FOLDER}/${dir}/bin/")
        endif()
    endforeach()

endfunction()

include(FetchContent)
include(git_describe)
include(install_runtime)

function(sibr_gitlibrary)
    cmake_parse_arguments(args "" "TARGET;GIT_REPOSITORY;GIT_TAG;ROOT_DIR;SOURCE_DIR" "INCLUDE_DIRS" ${ARGN})
    if(NOT args_TARGET)
        message(FATAL "Error on sibr_gitlibrary : please define your target name.")
        return()
    endif()

    if(NOT args_ROOT_DIR)
        set(args_ROOT_DIR ${args_TARGET})
    endif()

    if(NOT args_SOURCE_DIR)
        set(args_SOURCE_DIR ${args_TARGET})
    endif()

    if(args_GIT_REPOSITORY AND args_GIT_TAG)
        if(EXISTS ${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/${args_SOURCE_DIR}/.git)
            git_describe(
                PATH ${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/${args_SOURCE_DIR}
                GIT_URL SIBR_GITLIBRARY_URL
                GIT_BRANCH SIBR_GITLIBRARY_BRANCH
                GIT_COMMIT_HASH SIBR_GITLIBRARY_COMMIT_HASH
                GIT_TAG SIBR_GITLIBRARY_TAG
            )

            if((SIBR_GITLIBRARY_URL STREQUAL args_GIT_REPOSITORY) AND
                ((SIBR_GITLIBRARY_BRANCH STREQUAL args_GIT_TAG) OR
                 (SIBR_GITLIBRARY_TAG STREQUAL args_GIT_TAG) OR
                 (SIBR_GITLIBRARY_COMMIT_HASH STREQUAL args_GIT_TAG)))
                message(STATUS "Library ${args_TARGET} already available, skipping.")
                set(SIBR_GITLIBRARY_DECLARED ON)
            else()
                message(STATUS "Adding library ${args_TARGET} from git...")
            endif()
        endif()

        FetchContent_Declare(${args_TARGET}
            GIT_REPOSITORY 	${args_GIT_REPOSITORY}
            GIT_TAG			${args_GIT_TAG}
            GIT_SHALLOW		ON
            SOURCE_DIR 		${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/${args_SOURCE_DIR}
            SUBBUILD_DIR    ${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/subbuild
            BINARY_DIR      ${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/build
        )
        FetchContent_GetProperties(${args_TARGET})
        string(TOLOWER "<name>" lcTargetName)

        if((NOT SIBR_GITLIBRARY_DECLARED) AND (NOT ${lcTargetName}_POPULATED))
            message(STATUS "Populating library ${args_TARGET}...")
            FetchContent_Populate(${args_TARGET} QUIET
                GIT_REPOSITORY 	${args_GIT_REPOSITORY}
                GIT_TAG			${args_GIT_TAG}
                SOURCE_DIR 		${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/${args_SOURCE_DIR}
                SUBBUILD_DIR    ${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/subbuild
                BINARY_DIR      ${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/build
            )
        endif()

        add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/${args_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/extlibs/${args_ROOT_DIR}/build)

        get_target_property(type ${args_TARGET} TYPE)
        if(NOT (type STREQUAL "INTERFACE_LIBRARY"))
            set_target_properties(${args_TARGET} PROPERTIES FOLDER "extlibs")

            ibr_install_target(${args_TARGET}
                COMPONENT   ${args_TARGET}_install  ## will create custom target to install only this project
            )
        endif()

        list(APPEND ${args_TARGET}_INCLUDE_DIRS ${EXTLIBS_PACKAGE_FOLDER}/${args_ROOT_DIR})
        list(APPEND ${args_TARGET}_INCLUDE_DIRS ${EXTLIBS_PACKAGE_FOLDER}/${args_ROOT_DIR}/${args_SOURCE_DIR})

        foreach(args_INCLUDE_DIR ${args_INCLUDE_DIRS})
            list(APPEND ${args_TARGET}_INCLUDE_DIRS ${EXTLIBS_PACKAGE_FOLDER}/${args_ROOT_DIR}/${args_SOURCE_DIR}/${args_INCLUDE_DIR})
        endforeach()

        include_directories(${${args_TARGET}_INCLUDE_DIRS})
    else()
        message(FATAL "Error on sibr_gitlibrary for target ${args_TARGET}: missing git tag or git url.")
    endif()
endfunction()