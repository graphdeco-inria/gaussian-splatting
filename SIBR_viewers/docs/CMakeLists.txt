# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


#########################################################
# Include doxygen documentation target
#########################################################
option(BUILD_DOCUMENTATION "build doxygen documentation ('Build' DOCUMENTATION target, and find the compiled docs in install/docs/index.html)" OFF)
if(BUILD_DOCUMENTATION)
	set(DOXYGEN_REQUIRED_VERSION "1.8.17")
	find_package(Doxygen)
	if(NOT DOXYGEN_FOUND)
		message(FATAL_ERROR "Doxygen not found, unable to generate documentation.")
	elseif(DOXYGEN_VERSION VERSION_LESS DOXYGEN_REQUIRED_VERSION)
		message(FATAL_ERROR "Doxygen version is less than ${DOXYGEN_REQUIRED_VERSION} (Current version is ${DOXYGEN_VERSION}).")
	else()
		set(DOXY_DOC_DEST_DIR	${CMAKE_INSTALL_ROOT}/docs)						## used in the doxyfile.in
		
		set(DOXY_DOC_INPUT_ROOT_DIRS        "${CMAKE_HOME_DIRECTORY}/src ${CMAKE_HOME_DIRECTORY}/docs ${CMAKE_CURRENT_BINARY_DIR}/generated")    				## used in the doxyfile.in
		set(DOXY_DOC_EXCLUDE_PATTERNS_DIRS  "${DOXY_DOC_EXCLUDE_PATTERNS_DIRS}") ## used in the doxyfile.in
		set(DOXY_DOC_COMMON_IMG_PATH        "${CMAKE_CURRENT_SOURCE_DIR}/img ${CMAKE_HOME_DIRECTORY}/src/projects")
		set(DOXY_DOC_PAGES_DIR				"${CMAKE_CURRENT_SOURCE_DIR}/pages")
		set(DOXY_DOC_GENERATED_DOC_DIR		"${CMAKE_CURRENT_BINARY_DIR}/generated")
		
		string(REPLACE "\\" "\\\\" SIBR_PROJECTS_SAMPLES_SUBPAGE_REF	"${SIBR_PROJECTS_SAMPLES_SUBPAGE_REF}")
		string(REPLACE "\\" "\\\\" SIBR_PROJECTS_OURS_SUBPAGE_REF		"${SIBR_PROJECTS_OURS_SUBPAGE_REF}")
		string(REPLACE "\\" "\\\\" SIBR_PROJECTS_TOOLBOX_SUBPAGE_REF	"${SIBR_PROJECTS_TOOLBOX_SUBPAGE_REF}")
		string(REPLACE "\\" "\\\\" SIBR_PROJECTS_OTHERS_SUBPAGE_REF		"${SIBR_PROJECTS_OTHERS_SUBPAGE_REF}")
		string(REPLACE "\\" "\\\\" SIBR_PROJECTS_SAMPLES_REF_REF		"${SIBR_PROJECTS_SAMPLES_REF_REF}")
		string(REPLACE "\\" "\\\\" SIBR_PROJECTS_OURS_REF_REF			"${SIBR_PROJECTS_OURS_REF_REF}")
		string(REPLACE "\\" "\\\\" SIBR_PROJECTS_TOOLBOX_REF_REF		"${SIBR_PROJECTS_TOOLBOX_REF_REF}")
		string(REPLACE "\\" "\\\\" SIBR_PROJECTS_OTHERS_REF_REF			"${SIBR_PROJECTS_OTHERS_REF_REF}")

		string(REPLACE "\n" "\\n" SIBR_PROJECTS_SAMPLES_SUBPAGE_REF		"${SIBR_PROJECTS_SAMPLES_SUBPAGE_REF}")
		string(REPLACE "\n" "\\n" SIBR_PROJECTS_OURS_SUBPAGE_REF		"${SIBR_PROJECTS_OURS_SUBPAGE_REF}")
		string(REPLACE "\n" "\\n" SIBR_PROJECTS_TOOLBOX_SUBPAGE_REF		"${SIBR_PROJECTS_TOOLBOX_SUBPAGE_REF}")
		string(REPLACE "\n" "\\n" SIBR_PROJECTS_OTHERS_SUBPAGE_REF		"${SIBR_PROJECTS_OTHERS_SUBPAGE_REF}")
		string(REPLACE "\n" "\\n" SIBR_PROJECTS_SAMPLES_REF_REF			"${SIBR_PROJECTS_SAMPLES_REF_REF}")
		string(REPLACE "\n" "\\n" SIBR_PROJECTS_OURS_REF_REF			"${SIBR_PROJECTS_OURS_REF_REF}")
		string(REPLACE "\n" "\\n" SIBR_PROJECTS_TOOLBOX_REF_REF			"${SIBR_PROJECTS_TOOLBOX_REF_REF}")
		string(REPLACE "\n" "\\n" SIBR_PROJECTS_OTHERS_REF_REF			"${SIBR_PROJECTS_OTHERS_REF_REF}")

		file(GLOB doxygen_config_files "*.in")
		foreach(filename ${doxygen_config_files})
			message(STATUS "Generating ${filename}...")
			get_filename_component(output_filename ${filename} NAME_WLE)
			message(STATUS "Output in ${CMAKE_CURRENT_BINARY_DIR}/${output_filename}...")
			configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${output_filename} @ONLY)
		endforeach()

		add_custom_target(DOCUMENTATION ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/doxygen_prebuild.cmake
			COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/doxyfile"
			WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}
			COMMENT "Building user's documentation into ${DOXY_DOC_DEST_DIR} dir..."
		)
	endif()
endif()
