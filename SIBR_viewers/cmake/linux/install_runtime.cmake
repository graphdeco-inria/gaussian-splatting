# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


## This file is mainly used to allow runtime installation
## There are some utilities cmake functions to ease the generic deployement (abstract common usage of cmake)...
##
## You cannot run your programm automaticaly from your CNAKE_BINARY_DIR when you build
## as it will miss all dependencies and ressources files...
## You have to run install target in order to test your programm.
##
## The only one function/macros you may use inside your sub-CMakeLists.txt (sub-project) is :
## ******************
## ibr_install_target macro => see documentation at the end of this file
## ******************
## It use these utilities cmake functions to abstract the installation in an uniform way for all sub-projects.
##
if(__install_runtime_cmake_INCLUDED__)
	return()
else()
	set(__install_runtime_cmake_INCLUDED__ ON)
endif()


##
## Allow to write a resource config file which contain additional ressource paths
## (used by IBR_Common Resource system to load shaders and potentialy images, plugins and so on)
##
## ADD option list all the paths to add in the file (relative paths are interpreted relative to working dir of the executable)
## INSTALL option to specify where we want to install this file
##
## Example usage:
## resourceFile(ADD "shaders" "${PROJECT_NAME}_rsc" INSTALL bin)
##
macro(resourceFile)
	cmake_parse_arguments(rsc "" "INSTALL;FILE_PATH;CONFIG_TYPE" "ADD" ${ARGN}) ## both args are directory path

	if(rsc_ADD)
		unset(IBR_RSC_FILE_CONTENT_LIST)
		if(EXISTS "${rsc_FILE_PATH}")
			file(READ "${rsc_FILE_PATH}" IBR_RSC_FILE_CONTENT)
			string(REGEX REPLACE "\n" ";" IBR_RSC_FILE_CONTENT_LIST "${IBR_RSC_FILE_CONTENT}")
		endif()
		list(APPEND IBR_RSC_FILE_CONTENT_LIST "${rsc_ADD}")
		list(REMOVE_DUPLICATES IBR_RSC_FILE_CONTENT_LIST)
		file(WRITE "${rsc_FILE_PATH}" "")
		foreach(rscDir ${IBR_RSC_FILE_CONTENT_LIST})
			file(APPEND "${rsc_FILE_PATH}" "${rscDir}\n")
		endforeach()
		unset(rsc_ADD)
	endif()

	if(rsc_INSTALL)
		install(FILES ${rsc_FILE_PATH} CONFIGURATIONS ${rsc_CONFIG_TYPE} DESTINATION ${rsc_INSTALL})
		unset(rsc_INSTALL)
	endif()
endmacro()


##
## Install *.pdb generated file for the current cmake project
## assuming the output target name is the cmake project name.
## This macro is useful for crossplateform multi config mode.
##
## Usage Example:
##
## 	if(DEFINED CMAKE_BUILD_TYPE)						## for make/nmake based
##		installPDB(${PROJECT_NAME} ${CMAKE_BUILD_TYPE} RUNTIME_DEST bin ARCHIVE_DEST lib LIBRARY_DEST lib)
## 	endif()
##	foreach(CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES}) 	## for multi config types (MSVC based)
##		installPDB(${PROJECT_NAME} ${CONFIG_TYPES} RUNTIME_DEST bin ARCHIVE_DEST lib LIBRARY_DEST lib)
##	endforeach()
##
macro(installPDB targetName configType)
	cmake_parse_arguments(instpdb "" "COMPONENT" "ARCHIVE_DEST;LIBRARY_DEST;RUNTIME_DEST" ${ARGN}) ## both args are directory path

	if(NOT MSVC)
		return()
	endif()

    ## Check if DESTINATION are provided according to the TYPE of the given target (see install command doc to see correspodances)
    get_target_property(type ${targetName} TYPE)
    if(${type} MATCHES "EXECUTABLE" AND instpdb_RUNTIME_DEST)
        set(pdb_DESTINATION ${instpdb_RUNTIME_DEST})
    elseif(${type} MATCHES "STATIC_LIBRARY" AND instpdb_ARCHIVE_DEST)
        set(pdb_DESTINATION ${instpdb_ARCHIVE_DEST})
    elseif(${type} MATCHES "MODULE_LIBRARY" AND instpdb_LIBRARY_DEST)
        set(pdb_DESTINATION ${instpdb_LIBRARY_DEST})
    elseif(${type} MATCHES "SHARED_LIBRARY")
        if(WIN32 AND instpdb_RUNTIME_DEST)
            set(pdb_DESTINATION ${instpdb_RUNTIME_DEST})
        else()
            set(pdb_DESTINATION ${instpdb_LIBRARY_DEST})
        endif()
    endif()

    if(NOT pdb_DESTINATION)
		set(pdb_DESTINATION bin) ## default destination of the pdb file
	endif()

	if(NOT instpdb_COMPONENT)
		set(instpdb_COMPONENT )
	else()
		set(instpdb_COMPONENT COMPONENT ${instpdb_COMPONENT})
	endif()

	string(TOUPPER ${configType} CONFIG_TYPES_UC)
	get_target_property(PDB_PATH ${targetName} PDB_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC})

	get_target_property(confModePostfix ${targetName} ${CONFIG_TYPES_UC}_POSTFIX)
	if(NOT confModePostfix)
		set(confModePostfix "")
	endif()
	set_target_properties(${targetName} PROPERTIES  PDB_NAME_${CONFIG_TYPES_UC} ${targetName}${confModePostfix})
	get_target_property(PDB_NAME ${targetName} PDB_NAME_${CONFIG_TYPES_UC})# if not set, this is empty

	if(EXISTS "${PDB_PATH}/${PDB_NAME}.pdb")
		install(FILES "${PDB_PATH}/${PDB_NAME}.pdb" CONFIGURATIONS ${configType} DESTINATION ${pdb_DESTINATION} ${instpdb_COMPONENT} OPTIONAL)
	endif()
endmacro()


##
## Add additional target to install a project independently and based on its component
## configMode is used to prevent default Release installation (we want also to install in other build/config type)
##
macro(installTargetProject targetOfProject targetOfInstallProject)
 	if(DEFINED CMAKE_BUILD_TYPE) ## for make/nmake based
		set(configMode ${CMAKE_BUILD_TYPE})
	elseif(MSVC)
		## $(Configuration) will be one of the following : Debug, Release, MinSizeRel, RelWithDebInfo
		set(configMode $(Configuration))
 	endif()
	if(configMode)
        get_target_property(srcFiles ${targetOfProject} SOURCES)
		add_custom_target(	${targetOfInstallProject} #ALL
							${CMAKE_COMMAND} -DBUILD_TYPE=${configMode} -DCOMPONENT=${targetOfInstallProject} -P ${CMAKE_BINARY_DIR}/cmake_install.cmake
							DEPENDS ${srcFiles}
							COMMENT "run the installation only for ${targetOfProject}" VERBATIM
							)
		add_dependencies(${targetOfInstallProject} ${targetOfProject})

		get_target_property(INSTALL_BUILD_FOLDER ${targetOfProject} FOLDER)
		set_target_properties(${targetOfInstallProject} PROPERTIES FOLDER ${INSTALL_BUILD_FOLDER})
	endif()
endmacro()

# Collect all currently added targets in all subdirectories
#
# Parameters:
# - _result the list containing all found targets
# - _dir root directory to start looking from
function(get_all_targets _result _dir)
    get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
    foreach(_subdir IN LISTS _subdirs)
        get_all_targets(${_result} "${_subdir}")
    endforeach()

    get_directory_property(_sub_targets DIRECTORY "${_dir}" BUILDSYSTEM_TARGETS)
    set(${_result} ${${_result}} ${_sub_targets} PARENT_SCOPE)
endfunction()

##
## Add targets for building and installing subdirectories
macro(subdirectory_target target directory build_folder)
	add_custom_target(${target}
		COMMENT "run build for all projects in this directory" VERBATIM
	)
	get_all_targets(ALL_TARGETS ${directory})
	add_dependencies(${target} ${ALL_TARGETS})
	add_custom_target(${target}_install
		${CMAKE_COMMAND} -DBUILD_TYPE=$<CONFIG> -DCOMPONENT=${target}_install -P ${CMAKE_BINARY_DIR}/cmake_install.cmake
		COMMENT "run install for all projects in this directory" VERBATIM
	)
	add_dependencies(${target}_install ${target})

	set_target_properties(${target}			PROPERTIES FOLDER ${build_folder})
	set_target_properties(${target}_install PROPERTIES FOLDER ${build_folder})
endmacro()


##  CMAKE install all required dependencies for an application (included system OS files like msvc*.dll for example)
##
## install_runtime(<installedFilePathTargetAppToResolve>
##      [TARGET                 name]
##      [PLUGINS 				name 		[nameN ...] [PLUGIN_PATH_NAME currentPathName [FROM_REL_PATH matchDirFromCurrentPathName] [PLUGIN_PATH_DEST installDir] ]
##      [PLUGINS 				...]
##      [DIRS 					path 		[pathN ...] ]
##		[TARGET_LIBRARIES  		filePath	[filePathN ...] ]
##		[TARGET_PACKAGES   		packageName [packageNameN ...] ]
##		[COMPONENT				installComponentName]
##		[PLAUSIBLES_POSTFIX		Debug_postfix [MinSizeRel_postfix relWithDebInfo_postfix ...] ]
##      [VERBOSE]
## )
##
## installedFilePathTargetAppToResolve : the final installed targetApp absolute full file path name you want to resolve
##
## TARGET           :   The target app we want to install. If given, it's used to look for link libraries paths (best choice to use, strongly advised to use it)
##
## PLUGINS 			: 	Some application built use/load some plugins which can't be detect inside its binary,
##						so, here you can specify which plugins the application use/load in order to install them
##						and resolve also there dependencies.
## 		With PLUGINS multi FLAGS 	:
## 	 		PLUGIN_PATH_NAME 	: The current plugin full file path we want to install
##			FROM_REL_PATH		: [optional: default only the file is kept] From which matching dir of the plugin path we want to install (keep the directories structure)
##			PLUGIN_PATH_DEST	: [optional: default relative to executable directory] Where (full path to the install directory) we will install the plugin file (or file path)
##
## DIRS 			:	A list of directories to looking for dependencies
## TARGET_LIBRARIES :	DEPRECATED (use TARGET flag instead) : The cmake content variables used for the target_link_libraries(<targetApp> ...)
## TARGET_PACKAGES 	: 	DEPRECATED (use TARGET flag instead) : The cmake package names used for the findPackage(...) for your targetApp
##						ADVICE: This flag add entries in cache (like: <packageName>_DIR), it could be useful to fill these variable!
## COMPONENT		:	(default to runtime) Is the component name associated to the installation
##						It is used when you want to install separatly some part of your projets (see install cmake doc)
## VERBOSE			: 	For debug or to get more informations in the output console
##
## Usage:
##	 install_runtime(${CMAKE_INSTALL_PREFIX}/${EXECUTABLE_NAME}${CMAKE_EXECUTABLE_SUFFIX}
##		VERBOSE
##      TARGET  ${PROJECT_NAME}
##      PLAUSIBLES_POSTFIX  _d
##      PLUGINS
##		    PLUGIN_PATH_NAME    ${PLUGIN_PATH_NAME}${CMAKE_SHARED_MODULE_SUFFIX} ## will be installed (default exec path if no PLUGINS_DEST) and then will be resolved
##			FROM_REL_PATH		plugins ## optional, used especially for keeping qt plugins tree structure
##          PLUGIN_PATH_DEST    ${CMAKE_INSTALL_PREFIX}/plugins ## (or relative path 'plugins' will be interpreted relative to installed executable)
##		DIRS				${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR}
##		TARGET_LIBRARIES	${OPENGL_LIBRARIES}         ## DEPRECATED (use TARGET flag instead)
##							${GLEW_LIBRARIES}
##							${GLUT_LIBRARIES}
##							${Boost_LIBRARIES}
##							${SuiteSparse_LIBRARIES}
##							${CGAL_LIBRARIES}
##		TARGET_PACKAGES		OPENGL                      ## DEPRECATED (use TARGET flag instead)
##							GLEW
##							GLUT
##							CGAL
##							Boost
##							SuiteSparse
##	)
##
## For plugins part, it use our internal parse_arguments_multi.cmake
##
function(install_runtime installedFilePathTargetAppToResolve)
    set(optionsArgs "VERBOSE")
    set(oneValueArgs "COMPONENT;INSTALL_FOLDER;CONFIG_TYPE")
    set(multiValueArgs "DIRS;PLUGINS;TARGET_LIBRARIES;TARGET_PACKAGES;TARGET;PLAUSIBLES_POSTFIX")
    cmake_parse_arguments(inst_run "${optionsArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(IS_ABSOLUTE ${installedFilePathTargetAppToResolve})
    else()
        set(installedFilePathTargetAppToResolve ${inst_run_INSTALL_FOLDER}/${installedFilePathTargetAppToResolve})
    endif()

	get_filename_component(EXEC_NAME ${installedFilePathTargetAppToResolve} NAME_WE)
	get_filename_component(EXEC_PATH ${installedFilePathTargetAppToResolve} PATH)

	if(NOT inst_run_COMPONENT)
		set(inst_run_COMPONENT runtime)
	endif()


    ## Try to append as more possible as possible paths to find dependencies (deprecated since we can use target_properties to get back paths)
    set(libPaths )
	foreach(libraryFileName ${inst_run_TARGET_LIBRARIES})
		if(IS_DIRECTORY "${libraryFileName}")
			list(APPEND libPaths "${libraryFileName}")
		else()
			get_filename_component(libpath "${libraryFileName}" PATH)
			if(EXISTS "${libpath}")
				list(APPEND libPaths "${libpath}")
			endif()
		endif()
	endforeach()

    ## This macro is used internaly here to recursilvely get path of LINK_LIBRARIES of each non imported target
    ## Typically if you have 2 internal dependencies between cmake targets, we want cmake to be able to get back path where are these dependencies
    macro(recurseDepList target)
        get_target_property(linkLibs ${target} LINK_LIBRARIES)
        foreach(lib ${linkLibs})
            string(FIND ${lib} ">" strId) ## cmake is using generator-expression?
			if(TARGET ${lib})
				## Skipping interface libraries as they're system ones
                get_target_property(type ${lib} TYPE)
				get_target_property(imported ${lib} IMPORTED)
				if(type STREQUAL "INTERFACE_LIBRARY")
					get_target_property(imp_loc ${lib} INTERFACE_IMPORTED_LOCATION)
					if(imp_loc)
						get_filename_component(imp_loc ${imp_loc} PATH)
						list(APPEND targetLibPath ${imp_loc})
					endif()
					get_target_property(loc ${lib} INTERFACE_LOCATION)
					if(loc)
						get_filename_component(loc ${loc} PATH)
						list(APPEND targetLibPath ${loc})
					endif()
                ## it's not a path but a single target name
                ## for build-target which are part of the current cmake configuration : nothing to do as cmake already know the output path
                ## for imported target, we need to look for theire imported location
                elseif(imported)
                    get_target_property(imp_loc ${lib} IMPORTED_LOCATION)
                    if(imp_loc)
                        get_filename_component(imp_loc ${imp_loc} PATH)
                        list(APPEND targetLibPath ${imp_loc})
                    endif()
                    get_target_property(loc ${lib} LOCATION)
                    if(loc)
                        get_filename_component(loc ${loc} PATH)
                        list(APPEND targetLibPath ${loc})
                    endif()
                else()
                    recurseDepList(${lib})
                endif()
            elseif(NOT ${strId} MATCHES -1) ## mean cmake use generator-expression (CMAKE VERSION > 3.0)
                string(REGEX MATCH      ">:[@A-Za-z_:/.0-9-]+"           targetLibPath ${lib})
                string(REGEX REPLACE    ">:([@A-Za-z_:/.0-9-]+)" "\\1"   targetLibPath ${targetLibPath})
                get_filename_component(targetLibPath ${targetLibPath} PATH)
            elseif(EXISTS ${lib})
                set(targetLibPath ${lib})
                get_filename_component(targetLibPath ${targetLibPath} PATH)
            else()
                #message(STATUS "[install_runtime] skip link library : ${lib} , of target ${target}")
            endif()
            if(targetLibPath)
                list(APPEND targetLinkLibsPathList ${targetLibPath})
            endif()
        endforeach()
        if(targetLinkLibsPathList)
            list(REMOVE_DUPLICATES targetLinkLibsPathList)
        endif()
    endmacro()
    if(inst_run_TARGET)
        recurseDepList(${inst_run_TARGET})
        if(targetLinkLibsPathList)
            list(APPEND libPaths ${targetLinkLibsPathList})
        endif()
    endif()

	if(libPaths)
		list(REMOVE_DUPLICATES libPaths)
        foreach(libPath ${libPaths})
            get_filename_component(path ${libPath} PATH)
            list(APPEND libPaths ${path})
        endforeach()
	endif()


	## possible speciale dir(s) according to the build system and OS
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(BUILD_TYPES_FOR_DLL "x64")
		if(WIN32)
			list(APPEND BUILD_TYPES_FOR_DLL "Win64")
		endif()
	else()
		set(BUILD_TYPES_FOR_DLL "x86")
		if(WIN32)
			list(APPEND BUILD_TYPES_FOR_DLL "Win32")
		endif()
	endif()


	## Try to append as more as possible paths to find dependencies (here, mainly for *.dll)
	foreach(dir ${inst_run_DIRS} ${libPaths})
		if(EXISTS "${dir}/bin")
			list(APPEND inst_run_DIRS "${dir}/bin")
        elseif(EXISTS "${dir}")
            list(APPEND inst_run_DIRS "${dir}")
		endif()
	endforeach()
    list(REMOVE_DUPLICATES inst_run_DIRS)
	foreach(dir ${inst_run_DIRS})
		if(EXISTS "${dir}")
			list(APPEND argDirs ${dir})
			foreach(BUILD_TYPE_FOR_DLL ${BUILD_TYPES_FOR_DLL})
				if(EXISTS "${dir}/${BUILD_TYPE_FOR_DLL}")
					list(APPEND argDirs "${dir}/${BUILD_TYPE_FOR_DLL}")
				endif()
				foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES}) ## for windows multi-generator (MSVC)
					if(EXISTS "${dir}/${BUILD_TYPE_FOR_DLL}/${OUTPUTCONFIG}")
						list(APPEND argDirs "${dir}/${BUILD_TYPE_FOR_DLL}/${OUTPUTCONFIG}")
					endif()
				endforeach()
				if(CMAKE_BUILD_TYPE) ## for single generator (makefiles)
					if(EXISTS "${dir}/${BUILD_TYPE_FOR_DLL}/${CMAKE_BUILD_TYPE}")
						list(APPEND argDirs "${dir}/${BUILD_TYPE_FOR_DLL}/${CMAKE_BUILD_TYPE}")
					endif()
				endif()
			endforeach()
			foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES}) ## for windows multi-generator (MSVC)
				if(EXISTS "${dir}/${OUTPUTCONFIG}")
					list(APPEND argDirs "${dir}/${OUTPUTCONFIG}")
				endif()
				foreach(BUILD_TYPE_FOR_DLL ${BUILD_TYPES_FOR_DLL})
					if(EXISTS "${dir}/${OUTPUTCONFIG}/${BUILD_TYPE_FOR_DLL}")
						list(APPEND argDirs "${dir}/${OUTPUTCONFIG}/${BUILD_TYPE_FOR_DLL}")
					endif()
				endforeach()
			endforeach()
			if(CMAKE_BUILD_TYPE) ## for single generator (makefiles)
				if(EXISTS "${dir}/${CMAKE_BUILD_TYPE}")
					list(APPEND argDirs "${dir}/${CMAKE_BUILD_TYPE}")
				endif()
				foreach(BUILD_TYPE_FOR_DLL ${BUILD_TYPES_FOR_DLL})
					if(EXISTS "${dir}/${CMAKE_BUILD_TYPE}/${BUILD_TYPE_FOR_DLL}")
						list(APPEND argDirs "${dir}/${CMAKE_BUILD_TYPE}/${BUILD_TYPE_FOR_DLL}")
					endif()
				endforeach()
			endif()
		endif()
	endforeach()
	if(argDirs)
		list(REMOVE_DUPLICATES argDirs)
	endif()


	## Try to append as more possible paths to find dependencies (here, mainly for *.dll)
	foreach(packageName ${inst_run_TARGET_PACKAGES})
		if(EXISTS "${${packageName}_DIR}")
			list(APPEND packageDirs ${${packageName}_DIR})
			list(APPEND packageDirs ${${packageName}_DIR}/bin)
			foreach(BUILD_TYPE_FOR_DLL ${BUILD_TYPES_FOR_DLL})
				if(EXISTS "${${packageName}_DIR}/bin/${BUILD_TYPE_FOR_DLL}")
					list(APPEND packageDirs "${${packageName}_DIR}/bin/${BUILD_TYPE_FOR_DLL}")
				endif()
				foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES}) ## for windows multi-generator (MSVC)
					if(EXISTS "${${packageName}_DIR}/bin/${BUILD_TYPE_FOR_DLL}/${OUTPUTCONFIG}")
						list(APPEND packageDirs "${${packageName}_DIR}/bin/${BUILD_TYPE_FOR_DLL}/${OUTPUTCONFIG}")
					endif()
				endforeach()
				if(CMAKE_BUILD_TYPE) ## for single generator (makefiles)
					if(EXISTS "${${packageName}_DIR}/bin/${BUILD_TYPE_FOR_DLL}/${CMAKE_BUILD_TYPE}")
						list(APPEND packageDirs "${${packageName}_DIR}/bin/${BUILD_TYPE_FOR_DLL}/${CMAKE_BUILD_TYPE}")
					endif()
				endif()
			endforeach()
			foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES}) ## for windows multi-generator (MSVC)
				if(EXISTS "${${packageName}_DIR}/bin/${OUTPUTCONFIG}")
					list(APPEND packageDirs "${${packageName}_DIR}/bin/${OUTPUTCONFIG}")
				endif()
				foreach(BUILD_TYPE_FOR_DLL ${BUILD_TYPES_FOR_DLL})
					if(EXISTS "${${packageName}_DIR}/bin/${OUTPUTCONFIG}/${BUILD_TYPE_FOR_DLL}")
						list(APPEND packageDirs "${${packageName}_DIR}/bin/${OUTPUTCONFIG}/${BUILD_TYPE_FOR_DLL}")
					endif()
				endforeach()
			endforeach()
			if(CMAKE_BUILD_TYPE) ## for single generator (makefiles)
				if(EXISTS "${${packageName}_DIR}/bin/${CMAKE_BUILD_TYPE}")
					list(APPEND packageDirs "${${packageName}_DIR}/bin/${CMAKE_BUILD_TYPE}")
				endif()
				foreach(BUILD_TYPE_FOR_DLL ${BUILD_TYPES_FOR_DLL})
					if(EXISTS "${${packageName}_DIR}/bin/${CMAKE_BUILD_TYPE}/${BUILD_TYPE_FOR_DLL}")
						list(APPEND packageDirs "${${packageName}_DIR}/bin/${CMAKE_BUILD_TYPE}/${BUILD_TYPE_FOR_DLL}")
					endif()
				endforeach()
			endif()
		else()
			set(${packageName}_DIR "$ENV{${packageName}_DIR}" CACHE PATH "${packageName}_DIR root directory for looking for dirs containning *.dll")
		endif()
	endforeach()
	if(packageDirs)
		list(REMOVE_DUPLICATES packageDirs)
	endif()


	set(dirsToLookFor "${EXEC_PATH}")
	if(packageDirs)
		list(APPEND dirsToLookFor ${packageDirs})
	endif()
	if(argDirs)
		list(APPEND dirsToLookFor ${argDirs})
	endif()
	get_property(used_LINK_DIRECTORIES DIRECTORY PROPERTY LINK_DIRECTORIES)
	if (used_LINK_DIRECTORIES)
		list(APPEND dirsToLookFor ${used_LINK_DIRECTORIES})
		list(REMOVE_DUPLICATES dirsToLookFor)
	endif()


    ## handle plugins
	set(pluginsList "")
    include(parse_arguments_multi) ## this function will process recursively items of the sub-list [default print messages]
    function(parse_arguments_multi_function results)
        cmake_parse_arguments(pamf "VERBOSE" "PLUGIN_PATH_DEST;FROM_REL_PATH;EXEC_PATH;COMPONENT" "" ${ARGN}) ## EXEC_PATH and COMPONENT are for exclusive internal use
		list(REMOVE_DUPLICATES pamf_UNPARSED_ARGUMENTS)
        foreach(PLUGIN_PATH_NAME ${pamf_UNPARSED_ARGUMENTS})
            if(EXISTS ${PLUGIN_PATH_NAME})
                if(IS_DIRECTORY ${PLUGIN_PATH_NAME})
                    if(pamf_VERBOSE)
                        message(WARNING "${PLUGIN_PATH_NAME} IS_DIRECTORY, cannot installed a directory, please give a path filename")
                    endif()
                else()
                    if(NOT pamf_PLUGIN_PATH_DEST)
                        set(PLUGIN_PATH_DEST ${pamf_EXEC_PATH}) ## the default dest value
					else()
						set(PLUGIN_PATH_DEST ${pamf_PLUGIN_PATH_DEST})
                    endif()

					if(pamf_FROM_REL_PATH)
						file(TO_CMAKE_PATH ${PLUGIN_PATH_NAME} PLUGIN_PATH_NAME)
						get_filename_component(PLUGIN_PATH ${PLUGIN_PATH_NAME} PATH)
						unset(PLUGIN_PATH_LIST)
						unset(PLUGIN_PATH_LIST_COUNT)
						unset(PLUGIN_REL_PATH_LIST)
						unset(PLUGIN_REL_PATH)
						string(REPLACE "/" ";" PLUGIN_PATH_LIST ${PLUGIN_PATH}) ## create a list of dir
						list(FIND 	PLUGIN_PATH_LIST ${pamf_FROM_REL_PATH} id)
						list(LENGTH PLUGIN_PATH_LIST PLUGIN_PATH_LIST_COUNT)
						if(${id} GREATER 0)
							math(EXPR id "${id}+1") ## matches relative path not include
							math(EXPR PLUGIN_PATH_LIST_COUNT "${PLUGIN_PATH_LIST_COUNT}-1") ## the end of the list
							foreach(i RANGE ${id} ${PLUGIN_PATH_LIST_COUNT})
								list(GET 	PLUGIN_PATH_LIST 	${i} out)
								list(APPEND PLUGIN_REL_PATH_LIST 	${out})
							endforeach()
							foreach(dir ${PLUGIN_REL_PATH_LIST})
								set(PLUGIN_REL_PATH "${PLUGIN_REL_PATH}/${dir}")
							endforeach()
						endif()
						set(PLUGIN_PATH_DEST ${PLUGIN_PATH_DEST}${PLUGIN_REL_PATH})
					endif()

                    install(FILES ${PLUGIN_PATH_NAME} CONFIGURATIONS ${inst_run_CONFIG_TYPE} DESTINATION ${PLUGIN_PATH_DEST} COMPONENT ${pamf_COMPONENT})
                    get_filename_component(pluginName ${PLUGIN_PATH_NAME} NAME)
                    if(IS_ABSOLUTE ${PLUGIN_PATH_DEST})
                    else()
                        set(PLUGIN_PATH_DEST ${inst_run_INSTALL_FOLDER}/${PLUGIN_PATH_DEST})
                    endif()
                    list(APPEND pluginsList ${PLUGIN_PATH_DEST}/${pluginName})
                endif()
            else()
                message(WARNING "You need to provide a valid PLUGIN_PATH_NAME")
                set(pluginsList )
            endif()
        endforeach()
        set(${results} ${pluginsList} PARENT_SCOPE)
    endfunction()

    if(inst_run_VERBOSE)
        list(APPEND extra_flags_to_add VERBOSE)
    endif()
    list(APPEND extra_flags_to_add EXEC_PATH ${EXEC_PATH} COMPONENT ${inst_run_COMPONENT}) ## for internal use inside overloaded function
    list(LENGTH inst_run_PLUGINS inst_run_PLUGINS_count)
    if(${inst_run_PLUGINS_count} GREATER 0)
        parse_arguments_multi(PLUGIN_PATH_NAME inst_run_PLUGINS ${inst_run_PLUGINS} ## see internal overload parse_arguments_multi_function for processing each sub-list
                                NEED_RESULTS ${inst_run_PLUGINS_count}  ## this is used to check when we are in the first loop (in order to reset parse_arguments_multi_results)
                                EXTRAS_FLAGS ${extra_flags_to_add}      ## this is used to allow catching additional internal flags of our overloaded function
        )
    endif()

    #message(parse_arguments_multi_results = ${parse_arguments_multi_results})
    list(APPEND pluginsList ${parse_arguments_multi_results})



	## Install rules for required system runtimes such as MSVCRxx.dll
	set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP ON)
	include(InstallRequiredSystemLibraries)
	if(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
		install(FILES 			${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
				CONFIGURATIONS 	${inst_run_CONFIG_TYPE}
				DESTINATION 	${EXEC_PATH}
				COMPONENT   	${inst_run_COMPONENT}
		)
	endif()

	## print what we are doing to do
	if(inst_run_VERBOSE)
		message(STATUS "[install_runtime] On install target call, cmake will try to resolve dependencies for given app:\n ${installedFilePathTargetAppToResolve} (with plausible postfix: ${inst_run_PLAUSIBLES_POSTFIX})")
		if(pluginsList)
			message(STATUS "   and also for plugins :")
			foreach(plugin ${pluginsList})
				message(STATUS "      ${plugin}")
			endforeach()
		endif()
		message(STATUS "   Looking for dependencies into:")
		foreach(dir ${dirsToLookFor})
			message(STATUS "      ${dir}")
		endforeach()
	endif()

	## Install rules for required dependencies libs/plugins for the target app
	## will resolve all installed target files with config modes postfixes
	string(TOUPPER ${inst_run_CONFIG_TYPE} inst_run_CONFIG_TYPE_UC)
	get_target_property(postfix ${inst_run_TARGET} "${inst_run_CONFIG_TYPE_UC}_POSTFIX")
	install(CODE "set(target 						\"${inst_run_TARGET}\")" 												COMPONENT ${inst_run_COMPONENT}  CONFIGURATIONS ${CONFIG_TYPE})
	install(CODE "set(inst_run_CONFIG_TYPE 			\"${inst_run_CONFIG_TYPE}\")" 											COMPONENT ${inst_run_COMPONENT}  CONFIGURATIONS ${CONFIG_TYPE})
	install(CODE "set(inst_run_INSTALL_FOLDER 		\"${inst_run_INSTALL_FOLDER}\")" 										COMPONENT ${inst_run_COMPONENT}  CONFIGURATIONS ${CONFIG_TYPE})
	install(CODE "set(app	 						\"${EXEC_PATH}/${EXEC_NAME}${postfix}${CMAKE_EXECUTABLE_SUFFIX}\")" 	COMPONENT ${inst_run_COMPONENT}  CONFIGURATIONS ${CONFIG_TYPE})
	install(CODE "set(dirsToLookFor 				\"${dirsToLookFor}\")" 													COMPONENT ${inst_run_COMPONENT}  CONFIGURATIONS ${CONFIG_TYPE})
	install(CODE
		[[
			if("${CMAKE_INSTALL_CONFIG_NAME}" STREQUAL "${inst_run_CONFIG_TYPE}")
				message(STATUS "Installing ${target} dependencies...")

				file(GET_RUNTIME_DEPENDENCIES
					EXECUTABLES ${app}
					RESOLVED_DEPENDENCIES_VAR _r_deps
					UNRESOLVED_DEPENDENCIES_VAR _u_deps
					CONFLICTING_DEPENDENCIES_PREFIX _c_deps
					DIRECTORIES ${dirsToLookFor}
					PRE_EXCLUDE_REGEXES "api-ms-*"
					POST_EXCLUDE_REGEXES ".*system32/.*\\.dll" ".*SysWOW64/.*\\.dll"
				)
			
				if(_u_deps)
					message(WARNING "There were unresolved dependencies for executable ${EXEC_FILE}: \"${_u_deps}\"!")
				endif()
				if(_c_deps_FILENAMES)
					message(WARNING "There were conflicting dependencies for executable ${EXEC_FILE}: \"${_c_deps_FILENAMES}\"!")
				endif()
			
				foreach(_file ${_r_deps})
					file(INSTALL
					DESTINATION "${inst_run_INSTALL_FOLDER}/bin"
					TYPE SHARED_LIBRARY
					FOLLOW_SYMLINK_CHAIN
					FILES "${_file}"
				)
				endforeach()
			endif()
		]]
	   COMPONENT ${inst_run_COMPONENT} CONFIGURATIONS ${CONFIG_TYPE}
	)

endfunction()

## High level macro to install resources in the correct folder
##
## EXECUTABLE: [opt] option to copy files as programs
## RELATIVE  : [opt] copy files relatively to current folder
## TYPE      : [opt] type and folder where to store the files
## FOLDER    : [opt] subfolder to use
## FILES     : [opt] contains a list of resources files to copy to install folder
macro(ibr_install_rsc target)
	cmake_parse_arguments(install_rsc_${target} "EXECUTABLE;RELATIVE" "TYPE;FOLDER" "FILES" ${ARGN})
	set(rsc_target "${target}_${install_rsc_${target}_TYPE}")

	if(install_rsc_${target}_FOLDER)
		set(rsc_folder "${install_rsc_${target}_TYPE}/${install_rsc_${target}_FOLDER}")
	else()
		set(rsc_folder "${install_rsc_${target}_TYPE}")
	endif()

	add_custom_target(${rsc_target}
					COMMENT "run the ${install_rsc_${target}_TYPE} installation only for ${target} (component ${rsc_target})"
					VERBATIM)
	foreach(scriptFile ${install_rsc_${target}_FILES})
		if(install_rsc_${target}_RELATIVE)
			file(RELATIVE_PATH relativeFilename ${CMAKE_CURRENT_SOURCE_DIR} ${scriptFile})
		else()
			get_filename_component(relativeFilename ${scriptFile} NAME)
		endif()

		if(DEFINED CMAKE_BUILD_TYPE)						## for make/nmake based
			add_custom_command(TARGET ${rsc_target} POST_BUILD
							COMMAND ${CMAKE_COMMAND} -E
							copy_if_different ${scriptFile} ${CMAKE_INSTALL_PREFIX_${CMAKE_BUILD_TYPE}}/${rsc_folder}/${relativeFilename})
		endif()
		foreach(CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES}) 	## for multi config types (MSVC based)
			string(TOUPPER ${CONFIG_TYPES} CONFIG_TYPES_UC)
			add_custom_command(TARGET ${rsc_target} POST_BUILD
							COMMAND ${CMAKE_COMMAND} -E
							copy_if_different ${scriptFile} ${CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC}}/${rsc_folder}/${relativeFilename})
		endforeach()
	endforeach()

	get_target_property(INSTALL_RSC_BUILD_FOLDER ${target} FOLDER)
	set_target_properties(${rsc_target} PROPERTIES FOLDER ${INSTALL_RSC_BUILD_FOLDER})

	add_dependencies(${target} ${rsc_target})
	add_dependencies(PREBUILD ${rsc_target})

	if(DEFINED CMAKE_BUILD_TYPE)						## for make/nmake based
		resourceFile(ADD ${rsc_folder} CONFIG_TYPE ${CMAKE_BUILD_TYPE} FILE_PATH "${CMAKE_INSTALL_PREFIX_${CMAKE_BUILD_TYPE}}/ibr_resources.ini")
		
		if(install_rsc_${target}_EXECUTABLE)
			install(
				PROGRAMS ${install_rsc_${target}_FILES}
				CONFIGURATIONS ${CMAKE_BUILD_TYPE}
				DESTINATION "${CMAKE_INSTALL_PREFIX_${CMAKE_BUILD_TYPE}}/${rsc_folder}"
			)
		else()
			install(
				FILES ${install_rsc_${target}_FILES}
				CONFIGURATIONS ${CMAKE_BUILD_TYPE}
				DESTINATION "${CMAKE_INSTALL_PREFIX_${CMAKE_BUILD_TYPE}}/${rsc_folder}"
			)
		endif()
	endif()
	foreach(CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES}) 	## for multi config types (MSVC based)
		string(TOUPPER ${CONFIG_TYPES} CONFIG_TYPES_UC)
		resourceFile(ADD ${rsc_folder} CONFIG_TYPE ${CONFIG_TYPES} FILE_PATH "${CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC}}/ibr_resources.ini")
		
		if(install_rsc_${target}_EXECUTABLE)
			install(
				PROGRAMS ${install_rsc_${target}_FILES}
				CONFIGURATIONS ${CONFIG_TYPES}
				DESTINATION "${CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC}}/${rsc_folder}"
			)
		else()
			install(
				FILES ${install_rsc_${target}_FILES}
				CONFIGURATIONS ${CONFIG_TYPES}
				DESTINATION "${CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC}}/${rsc_folder}"
			)
		endif()
	endforeach()
endmacro()


## High level macro to install in an homogen way all our ibr targets (it use some functions inside this file)
##
## RSC_FILE_ADD : [opt] is used to auto write/append relative paths of target resources into a common file
## INSTALL_PDB  : [opt] is used to auto install PDB file (when using MSVC according to the target type)
## STANDALONE   : [opt] bool ON/OFF var to call install_runtime or not (for bundle resolution)
##       DIRS   : [opt] used if STANDALONE set to ON, see install_runtime doc
##       PLUGINS: [opt] used if STANDALONE set to ON, see install_runtime doc
## MSVC_CMD     : [opt] used to specify an absolute filePathName application to launch with the MSVC IDE Debugger associated to this target (project file)
## MSVC_ARGS    : [opt] load the MSVC debugger with correct settings (app path, args, working dir)
##
macro(ibr_install_target target)
	cmake_parse_arguments(ibrInst${target} "VERBOSE;INSTALL_PDB" "COMPONENT;MSVC_ARGS;STANDALONE;RSC_FOLDER" "SHADERS;RESOURCES;SCRIPTS;DIRS;PLUGINS" ${ARGN})
	
	if(ibrInst${target}_RSC_FOLDER)
		set(rsc_folder "${ibrInst${target}_RSC_FOLDER}")
	else()
		set(rsc_folder "${target}")
	endif()

	if(ibrInst${target}_SHADERS)
		ibr_install_rsc(${target} EXECUTABLE TYPE "shaders" FOLDER ${rsc_folder} FILES "${ibrInst${target}_SHADERS}")
    endif()
	
	if(ibrInst${target}_RESOURCES)
		ibr_install_rsc(${target} TYPE "resources" FOLDER ${rsc_folder} FILES "${ibrInst${target}_RESOURCES}")
    endif()
	
	if(ibrInst${target}_SCRIPTS)
		ibr_install_rsc(${target} EXECUTABLE TYPE "scripts" FOLDER ${rsc_folder} FILES "${ibrInst${target}_SCRIPTS}")
    endif()

    if(ibrInst${target}_COMPONENT)
        set(installCompArg COMPONENT ${ibrInst${target}_COMPONENT})
        ## Create a custom install target based on COMPONENT
        installTargetProject(${target} ${ibrInst${target}_COMPONENT})
	endif()
	
	if(DEFINED CMAKE_BUILD_TYPE)						## for make/nmake based
		set_target_properties(${target} PROPERTIES ${CMAKE_BUILD_TYPE}_POSTFIX 	"${CMAKE_${CMAKE_BUILD_TYPE}_POSTFIX}")
		get_target_property(CURRENT_TARGET_BUILD_TYPE_POSTFIX ${target} ${CMAKE_BUILD_TYPE}_POSTFIX)
	endif()
	foreach(CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES}) 	## for multi config types (MSVC based)
		string(TOUPPER ${CONFIG_TYPES} CONFIG_TYPES_UC)
		set_target_properties(${target} PROPERTIES ${CONFIG_TYPES_UC}_POSTFIX 	"${CMAKE_${CONFIG_TYPES_UC}_POSTFIX}")
		get_target_property(CURRENT_TARGET_BUILD_TYPE_POSTFIX ${target} ${CONFIG_TYPES_UC}_POSTFIX)
	endforeach()

	## Specify default installation rules
	if(DEFINED CMAKE_BUILD_TYPE)						## for make/nmake based
		install(TARGETS	${target}
			CONFIGURATIONS ${CMAKE_BUILD_TYPE}
			LIBRARY		DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE}} ${installCompArg}
			ARCHIVE		DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE}} ${installCompArg}
			RUNTIME 	DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE}} ${installCompArg}
		)
		install(TARGETS	${target}
			CONFIGURATIONS ${CMAKE_BUILD_TYPE}
			LIBRARY		DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE}} ${installCompArg}
			ARCHIVE		DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE}} ${installCompArg}
		)
	endif()
	foreach(CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES}) 	## for multi config types (MSVC based)
		string(TOUPPER ${CONFIG_TYPES} CONFIG_TYPES_UC)
		install(TARGETS	${target}
			CONFIGURATIONS ${CONFIG_TYPES}
			LIBRARY		DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC}} ${installCompArg}
			ARCHIVE		DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC}} ${installCompArg}
			RUNTIME 	DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC}} ${installCompArg}
		)
		install(TARGETS	${target}
			CONFIGURATIONS ${CONFIG_TYPES}
			LIBRARY		DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC}} ${installCompArg}
			ARCHIVE		DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC}} ${installCompArg}
		)
	endforeach()

    if(ibrInst${target}_INSTALL_PDB)
        if(DEFINED CMAKE_BUILD_TYPE)						## for make/nmake based
			installPDB(${target} ${CMAKE_BUILD_TYPE}
				LIBRARY_DEST ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE}}
				ARCHIVE_DEST ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE}}
				RUNTIME_DEST ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE}}
			)
        endif()
        foreach(CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES}) 	## for multi config types (MSVC based)
			string(TOUPPER ${CONFIG_TYPES} CONFIG_TYPES_UC)
			installPDB(${target} ${CONFIG_TYPES}
				LIBRARY_DEST ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC}}
				ARCHIVE_DEST ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC}}
				RUNTIME_DEST ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC}}
			)
        endforeach()
    endif()

    ## install dynamic necessary dependencies
    if(ibrInst${target}_STANDALONE)
        get_target_property(type ${target} TYPE)
        if(${type} MATCHES "EXECUTABLE")

            if(ibrInst${target}_VERBOSE)
                set(VERBOSE VERBOSE)
            else()
                set(VERBOSE )
			endif()
			
			if(DEFINED CMAKE_BUILD_TYPE)						## for make/nmake based
				install_runtime(bin/${target}${CMAKE_EXECUTABLE_SUFFIX} ## default relative to CMAKE_INSTALL_PREFIX
					INSTALL_FOLDER		"${CMAKE_INSTALL_PREFIX_${CMAKE_BUILD_TYPE}}"
					CONFIG_TYPE			${CMAKE_BUILD_TYPE}
					${VERBOSE}
					TARGET              ${target}
					${installCompArg}
					PLUGINS	## will be installed
										${ibrInst${target}_PLUGINS}
					DIRS				${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CMAKE_BUILD_TYPE}}
										${ibrInst${target}_DIRS}
				)
			endif()
			foreach(CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES}) 	## for multi config types (MSVC based)
				string(TOUPPER ${CONFIG_TYPES} CONFIG_TYPES_UC)
				install_runtime(bin/${target}${CMAKE_EXECUTABLE_SUFFIX} ## default relative to CMAKE_INSTALL_PREFIX
					INSTALL_FOLDER		"${CMAKE_INSTALL_PREFIX_${CONFIG_TYPES_UC}}"
					CONFIG_TYPE			${CONFIG_TYPES}
					${VERBOSE}
					TARGET              ${target}
					${installCompArg}
					PLUGINS	## will be installed
										${ibrInst${target}_PLUGINS}
					DIRS				${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPES_UC}}
										${ibrInst${target}_DIRS}
				)
			endforeach()
        else()
            message(WARNING "STANDALONE option is only compatible with EXECUTABLES target type. Skip the STANDALONE installation process.")
        endif()
    endif()

    ## Provide a way to directly load the MSVC debugger with correct settings
    if(MSVC)
        if(ibrInst${target}_MSVC_CMD)  ## command absolute filePathName is optional as the default is to use the installed target file application
            set(msvcCmdArg  COMMAND ${ibrInst${target}_MSVC_CMD}) ## flag following by the value (both to pass to the MSVCsetUserCommand function)
        endif()
        if(ibrInst${target}_MSVC_ARGS) ## args (between quotes) are optional
            set(msvcArgsArg ARGS ${ibrInst${target}_MSVC_ARGS})   ## flag following by the value (both to pass to the MSVCsetUserCommand function)
        endif()
        get_target_property(type ${target} TYPE)
        if( (ibrInst${target}_MSVC_CMD OR ibrInst${target}_MSVC_ARGS) OR (${type} MATCHES "EXECUTABLE") )
			include(MSVCsetUserCommand)
			if(DEFINED CMAKE_BUILD_TYPE)						## for make/nmake based
				MSVCsetUserCommand(	${target}
					PATH 			${CMAKE_OUTPUT_BIN_${CMAKE_BUILD_TYPE}} ##FILE option not necessary since it deduced from targetName
									ARGS				"${SIBR_PROGRAMARGS}"
					${msvcCmdArg}
					#${msvcArgsArg}
					WORKING_DIR		${CMAKE_OUTPUT_BIN_${CMAKE_BUILD_TYPE}}
				)
			endif()
			foreach(CONFIG_TYPES ${CMAKE_CONFIGURATION_TYPES}) 	## for multi config types (MSVC based)
				string(TOUPPER ${CONFIG_TYPES} CONFIG_TYPES_UC)
				MSVCsetUserCommand(	${target}
					PATH 			${CMAKE_OUTPUT_BIN_${CONFIG_TYPES_UC}} ##FILE option not necessary since it deduced from targetName
									ARGS				"${SIBR_PROGRAMARGS}"
					${msvcCmdArg}
					#${msvcArgsArg}
					WORKING_DIR		${CMAKE_OUTPUT_BIN_${CONFIG_TYPES_UC}}
				)
			endforeach()
        elseif(NOT ${type} MATCHES "EXECUTABLE")
            #message("Cannot set MSVCsetUserCommand with target ${target} without COMMAND parameter as it is not an executable (skip it)")
        endif()
    endif()

endmacro()
