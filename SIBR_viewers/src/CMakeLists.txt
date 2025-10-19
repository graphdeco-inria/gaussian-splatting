# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


################################################################################
# This CMakeLists.txt manages which projects should be built and add their     #
# dependencies.                                                                #
################################################################################
set(SIBR_FOLDER "core")
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

option(BUILD_SIBR "Build core libs of SIBR (sibr_system, sibr_graphics, sibr_view, sibr_assets, ...)" ON)

#https://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  foreach(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  endforeach()
  SET(${result} ${dirlist})
ENDMACRO()

set(SIBR_PROJECTS_FOLDER           "${CMAKE_CURRENT_SOURCE_DIR}/projects")
SUBDIRLIST(SUBDIRS ${SIBR_PROJECTS_FOLDER})

list(APPEND PROJECT_SUBFOLDERS "apps" "preprocess" "renderer" "scripts" "library")

# Moving ulr to the top of the list
list(PREPEND SUBDIRS "dataset_tools" "ulr" "basic")
list(REMOVE_DUPLICATES SUBDIRS)

## DEPS ##
include(include_once)

message(STATUS "\n\n****************** Handling core dependencies ******************")

include_once(dependencies)		## Map/bind 3rdParty/external dependencies packages to cmake

message(STATUS "****************************************************************\n\n")

foreach(subdir ${SUBDIRS})
  set(${subdir}_ROOT_DIR "${SIBR_PROJECTS_FOLDER}/${subdir}")
  set(PROJECT_NAME "BUILD_IBR_${subdir}")
  string(TOUPPER ${PROJECT_NAME} PROJECT_NAME)
  if(${${PROJECT_NAME}})
    foreach(PROJECT_SUBFOLDER ${PROJECT_SUBFOLDERS})
      if(EXISTS "${${subdir}_ROOT_DIR}/${PROJECT_SUBFOLDER}/cmake/Modules")
        list(APPEND CMAKE_MODULE_PATH ${${subdir}_ROOT_DIR}/${PROJECT_SUBFOLDER}/cmake/Modules)
      endif()

      if(EXISTS "${${subdir}_ROOT_DIR}/${PROJECT_SUBFOLDER}/cmake/dependencies.cmake")
        message(STATUS "********* Handling ${subdir} ${PROJECT_SUBFOLDER} dependencies *********")
        include_once("${${subdir}_ROOT_DIR}/${PROJECT_SUBFOLDER}/cmake/dependencies.cmake")

        message(STATUS "****************************************************************\n\n")
      endif()
    endforeach()
  endif()
endforeach()

Win3rdPartyGlobalCacheAction()

include_directories(.)

if (BUILD_SIBR)
  add_subdirectory(core/system)
  add_subdirectory(core/graphics)
  add_subdirectory(core/renderer)
  add_subdirectory(core/raycaster)
  add_subdirectory(core/view)
  add_subdirectory(core/scene)
  add_subdirectory(core/assets)
  add_subdirectory(core/imgproc)
  add_subdirectory(core/video)
endif()

set(PROJECTS_ON_AT_FIRST_BUILD "basic" "gaussianviewer" "remote")

foreach(subdir ${SUBDIRS})
  message(STATUS "Adding ${subdir} project")
  set(PROJECT_NAME "BUILD_IBR_${subdir}")
  string(TOUPPER ${PROJECT_NAME} PROJECT_NAME)

  if(NOT (DEFINED ${PROJECT_NAME}))
    foreach(PROJECT_SUBFOLDER ${PROJECT_SUBFOLDERS})
      if(EXISTS "${${subdir}_ROOT_DIR}/${PROJECT_SUBFOLDER}/CMakeLists.txt")
        if(subdir IN_LIST PROJECTS_ON_AT_FIRST_BUILD)
          option(${PROJECT_NAME} "Build project \"${subdir}\"" ON)
        else()
          option(${PROJECT_NAME} "Build project \"${subdir}\"" OFF)
        endif()
        break()
      endif()
    endforeach()
  endif()

  message(STATUS "${PROJECT_NAME} is ${${PROJECT_NAME}}")

  if(${${PROJECT_NAME}})
    if(EXISTS "${${subdir}_ROOT_DIR}/CMakeLists.txt")
      add_subdirectory("${${subdir}_ROOT_DIR}")
    else()
      foreach(PROJECT_SUBFOLDER ${PROJECT_SUBFOLDERS})
        if(EXISTS "${${subdir}_ROOT_DIR}/${PROJECT_SUBFOLDER}/CMakeLists.txt")
          add_subdirectory("${${subdir}_ROOT_DIR}/${PROJECT_SUBFOLDER}")
        endif()
      endforeach()
    endif()

    if(EXISTS "${${subdir}_ROOT_DIR}/documentation/" AND BUILD_DOCUMENTATION)
      unset(PROJECT_PAGE)
      unset(PROJECT_LINK)
      unset(PROJECT_DESCRIPTION)
      unset(PROJECT_TYPE)
      include("${${subdir}_ROOT_DIR}/documentation/${subdir}_doc.cmake" OPTIONAL)

      if(NOT DEFINED PROJECT_PAGE)
        set(PROJECT_PAGE "${subdir}Page")
      endif()

      if(NOT DEFINED PROJECT_TYPE)
        set(PROJECT_TYPE "OTHERS")
      endif()

      set(PROJECT_SUBPAGE_REF "  - @subpage ${PROJECT_PAGE}")
      set(PROJECT_REF_REF "  - @ref ${PROJECT_PAGE}")

      if(DEFINED PROJECT_LINK)
        string(APPEND PROJECT_SUBPAGE_REF " (${PROJECT_LINK})")
        string(APPEND PROJECT_REF_REF " (${PROJECT_LINK})")
      endif()

      if(DEFINED PROJECT_DESCRIPTION)
        string(APPEND PROJECT_SUBPAGE_REF ": ${PROJECT_DESCRIPTION}")
        string(APPEND PROJECT_REF_REF " (${PROJECT_DESCRIPTION})")
      endif()

      string(APPEND SIBR_PROJECTS_${PROJECT_TYPE}_SUBPAGE_REF_LOCAL "${PROJECT_SUBPAGE_REF}\n")
      string(APPEND SIBR_PROJECTS_${PROJECT_TYPE}_REF_REF_LOCAL "${PROJECT_REF_REF}\n")

      if(EXISTS "${${subdir}_ROOT_DIR}/documentation/img")
        set(DOXY_APP_SPECIFIC_IMG_PATH_LOCAL "${DOXY_APP_SPECIFIC_IMG_PATH_LOCAL} ${${subdir}_ROOT_DIR}/documentation/img")
      endif()

      if(EXISTS "${${subdir}_ROOT_DIR}/LICENSE.md")
        set(DOXY_DOC_EXCLUDE_PATTERNS_DIRS_LOCAL "${DOXY_DOC_EXCLUDE_PATTERNS_DIRS_LOCAL} ${${subdir}_ROOT_DIR}/LICENSE.md")
      endif()
    endif()
  else()
    set(DOXY_DOC_EXCLUDE_PATTERNS_DIRS_LOCAL "${DOXY_DOC_EXCLUDE_PATTERNS_DIRS_LOCAL} ${${subdir}_ROOT_DIR}")
  endif()
endforeach()
  
set(SIBR_PROJECTS_SAMPLES_SUBPAGE_REF "${SIBR_PROJECTS_SAMPLES_SUBPAGE_REF_LOCAL}" PARENT_SCOPE)
set(SIBR_PROJECTS_OURS_SUBPAGE_REF "${SIBR_PROJECTS_OURS_SUBPAGE_REF_LOCAL}" PARENT_SCOPE)
set(SIBR_PROJECTS_TOOLBOX_SUBPAGE_REF "${SIBR_PROJECTS_TOOLBOX_SUBPAGE_REF_LOCAL}" PARENT_SCOPE)
set(SIBR_PROJECTS_OTHERS_SUBPAGE_REF "${SIBR_PROJECTS_OTHERS_SUBPAGE_REF_LOCAL}" PARENT_SCOPE)
set(SIBR_PROJECTS_SAMPLES_REF_REF "${SIBR_PROJECTS_SAMPLES_REF_REF_LOCAL}" PARENT_SCOPE)
set(SIBR_PROJECTS_OURS_REF_REF "${SIBR_PROJECTS_OURS_REF_REF_LOCAL}" PARENT_SCOPE)
set(SIBR_PROJECTS_TOOLBOX_REF_REF "${SIBR_PROJECTS_TOOLBOX_REF_REF_LOCAL}" PARENT_SCOPE)
set(SIBR_PROJECTS_OTHERS_REF_REF "${SIBR_PROJECTS_OTHERS_REF_REF_LOCAL}" PARENT_SCOPE)
set(DOXY_APP_SPECIFIC_IMG_PATH "${DOXY_APP_SPECIFIC_IMG_PATH_LOCAL}" PARENT_SCOPE)
set(DOXY_DOC_EXCLUDE_PATTERNS_DIRS "${DOXY_DOC_EXCLUDE_PATTERNS_DIRS_LOCAL}" PARENT_SCOPE)

if (BUILD_IBR_TFGL_INTEROP)
  add_subdirectory(projects/tfgl_interop/renderer/custom_ops)
endif()
