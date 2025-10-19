# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


if(__git_describe_INCLUDED__)
	return()
else()
	set(__git_describe_INCLUDED__ ON)
endif()

find_package(Git)
if(Git_FOUND)
  message(STATUS "Git found: ${GIT_EXECUTABLE}")
else()
  message(FATAL_ERROR "Git not found. Aborting")
endif()

macro(git_describe)
    cmake_parse_arguments(GIT_DESCRIBE "" "GIT_URL;GIT_BRANCH;GIT_COMMIT_HASH;GIT_TAG;GIT_VERSION;PATH" "" ${ARGN})

    if(NOT GIT_DESCRIBE_PATH)
        set(GIT_DESCRIBE_PATH ${CMAKE_SOURCE_DIR})
    endif()

    if(GIT_DESCRIBE_GIT_URL)
        # Get the current remote
        execute_process(
            COMMAND git remote
            WORKING_DIRECTORY   ${GIT_DESCRIBE_PATH}
            OUTPUT_VARIABLE     GIT_DESCRIBE_GIT_REMOTE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        # Get the current remote
        execute_process(
            COMMAND git remote get-url ${GIT_DESCRIBE_GIT_REMOTE}
            WORKING_DIRECTORY   ${GIT_DESCRIBE_PATH}
            OUTPUT_VARIABLE     ${GIT_DESCRIBE_GIT_URL}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(GIT_DESCRIBE_GIT_BRANCH)
        # Get the current working branch
        execute_process(
            COMMAND git rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY   ${GIT_DESCRIBE_PATH}
            OUTPUT_VARIABLE     ${GIT_DESCRIBE_GIT_BRANCH}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(GIT_DESCRIBE_GIT_COMMIT_HASH)
        # Get the latest abbreviated commit hash of the working branch
        execute_process(
            COMMAND git rev-parse HEAD
            WORKING_DIRECTORY   ${GIT_DESCRIBE_PATH}
            OUTPUT_VARIABLE     ${GIT_DESCRIBE_GIT_COMMIT_HASH}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(GIT_DESCRIBE_GIT_TAG)
        # Get the tag
        execute_process(
            COMMAND git describe --tags --exact-match
            WORKING_DIRECTORY   ${GIT_DESCRIBE_PATH}
            OUTPUT_VARIABLE     ${GIT_DESCRIBE_GIT_TAG}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(GIT_DESCRIBE_GIT_VERSION)
        # Get the version from git describe
        execute_process(
            COMMAND git describe
            WORKING_DIRECTORY   ${GIT_DESCRIBE_PATH}
            OUTPUT_VARIABLE     ${GIT_DESCRIBE_GIT_VERSION}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if(${GIT_DESCRIBE_GIT_VERSION} STREQUAL "")
            execute_process(
                COMMAND git rev-parse --abbrev-ref HEAD
                WORKING_DIRECTORY   ${GIT_DESCRIBE_PATH}
                OUTPUT_VARIABLE     GIT_DESCRIBE_GIT_VERSION_BRANCH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            execute_process(
                COMMAND git log -1 --format=%h
                WORKING_DIRECTORY   ${GIT_DESCRIBE_PATH}
                OUTPUT_VARIABLE     GIT_DESCRIBE_GIT_VERSION_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )

            set(${GIT_DESCRIBE_GIT_VERSION} "${GIT_DESCRIBE_GIT_VERSION_BRANCH}-${GIT_DESCRIBE_GIT_VERSION_COMMIT}")
        endif()
    endif()

endmacro()