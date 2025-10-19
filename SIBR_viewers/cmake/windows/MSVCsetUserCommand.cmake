# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


if(__MSVCsetUserCommand_cmake_INCLUDED__)
	return()
else()
	set(__MSVCsetUserCommand_cmake_INCLUDED__ ON)
endif()

## Allow to configure the Debugger settings of visual studio
## Note: Using this command under linux doesn't affect anything
## On run Debug Windows local : visual will try to load a specific COMMAND with ARGS in the provided WORKING_DIR
##
## usage:
## MSVCsetUserCommand(	<targetName>
##    [COMMAND 			<myCustomAppToLaunch> | [ PATH <myCustomDirWhereIsDefaultTargetFileNameToLaunch> [FILE <myCustomExecFileToLaunch>] ] ]
##    ARGS 				<associatedArguments>
##    WORKING_DIR		<whereStartTheProgram>
## )
##
## Warning 1 : All arugments () must be passed under quotes
## Warning 2 : WORKING_DIR path arg have to finish with remain slah '/'
## Warning 3 : use COMMAND for external app OR PATH (optionaly with FILE) option(s) to set your built/installed/moved target
##
## Example 1:
## include(MSVCsetUserCommand)
## MSVCsetUserCommand(	UnityRenderingPlugin
## 	  COMMAND 			"C:/Program Files (x86)/Unity/Editor/Unity.exe"
## 	  ARGS				"-force-opengl -projectPath \"${CMAKE_HOME_DIRECTORY}/UnityPlugins/RenderingPluginExample/UnityProject\""
## 	  WORKING_DIR		"${CMAKE_HOME_DIRECTORY}/UnityPlugins/RenderingPluginExample/UnityProject"
## 	  VERBOSE
## )
##
## Example 2:
## include(MSVCsetUserCommand)
## MSVCsetUserCommand(	ibrApp
## 	  PATH 				"C:/Program Files (x86)/workspace/IBR/install"
##	  FILE				"ibrApp${CMAKE_EXECUTABLE_SUFFIX}" ## this option line is optional since the target name didn't change between build and install step
## 	  ARGS				"-path \"${CMAKE_HOME_DIRECTORY}/dataset\""
## 	  WORKING_DIR		"${CMAKE_HOME_DIRECTORY}"
## 	  VERBOSE
## )
##
function(MSVCsetUserCommand targetName)
    cmake_parse_arguments(MSVCsuc "VERBOSE" "PATH;FILE;COMMAND;ARGS;WORKING_DIR" "" ${ARGN} )

	## If no arguments are given, do not create an unecessary .vcxproj.user file
	set(MSVCsuc_DEFAULT OFF)

	if(MSVCsuc_PATH AND MSVCsuc_DEFAULT)
		set(MSVCsuc_DEFAULT OFF)
	endif()

	if(MSVCsuc_FILE AND MSVCsuc_DEFAULT)
		set(MSVCsuc_DEFAULT OFF)
	endif()

	if(NOT MSVCsuc_COMMAND)
		if(MSVCsuc_PATH AND MSVCsuc_FILE)
			set(MSVCsuc_COMMAND "${MSVCsuc_PATH}\\${MSVCsuc_FILE}")
		elseif(MSVCsuc_PATH)
			set(MSVCsuc_COMMAND "${MSVCsuc_PATH}\\$(TargetFileName)")
		else()
			set(MSVCsuc_COMMAND "$(TargetPath)") ## => $(TargetDir)\$(TargetName)$(TargetExt)
		endif()
	elseif(MSVCsuc_DEFAULT)
		set(MSVCsuc_DEFAULT OFF)
	endif()

        # NOTE: there was a typo here. there is an else if written after else statement
        # changing the order of the else if statement
	if(MSVCsuc_WORKING_DIR)
		file(TO_NATIVE_PATH ${MSVCsuc_WORKING_DIR} MSVCsuc_WORKING_DIR)
	elseif(MSVCsuc_DEFAULT)
		set(MSVCsuc_DEFAULT OFF)
	else()
		set(MSVCsuc_WORKING_DIR "$(ProjectDir)")
	endif()

	if(NOT MSVCsuc_ARGS)
		set(MSVCsuc_ARGS "")
	elseif(MSVCsuc_DEFAULT)
		set(MSVCsuc_DEFAULT OFF)
	endif()

	if(MSVC10 OR (MSVC AND MSVC_VERSION GREATER 1600)) # 2010 or newer

		if(CMAKE_SIZEOF_VOID_P EQUAL 8)
			set(PLATEFORM_BITS x64)
		else()
			set(PLATEFORM_BITS Win32)
		endif()

		if(NOT MSVCsuc_DEFAULT AND PLATEFORM_BITS)

			file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${targetName}.vcxproj.user"
		"<?xml version=\"1.0\" encoding=\"utf-8\"?>
<Project ToolsVersion=\"4.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">
  <PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='Release|${PLATEFORM_BITS}'\">
    <LocalDebuggerCommand>${MSVCsuc_COMMAND}</LocalDebuggerCommand>
    <LocalDebuggerCommandArguments>${MSVCsuc_ARGS}</LocalDebuggerCommandArguments>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
	<LocalDebuggerWorkingDirectory>${MSVCsuc_WORKING_DIR}</LocalDebuggerWorkingDirectory>
  </PropertyGroup>
  <PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='Debug|${PLATEFORM_BITS}'\">
    <LocalDebuggerCommand>${MSVCsuc_COMMAND}</LocalDebuggerCommand>
    <LocalDebuggerCommandArguments>${MSVCsuc_ARGS}</LocalDebuggerCommandArguments>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
    <LocalDebuggerWorkingDirectory>${MSVCsuc_WORKING_DIR}</LocalDebuggerWorkingDirectory>
  </PropertyGroup>
    <PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='MinSizeRel|${PLATEFORM_BITS}'\">
    <LocalDebuggerCommand>${MSVCsuc_COMMAND}</LocalDebuggerCommand>
    <LocalDebuggerCommandArguments>${MSVCsuc_ARGS}</LocalDebuggerCommandArguments>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
    <LocalDebuggerWorkingDirectory>${MSVCsuc_WORKING_DIR}</LocalDebuggerWorkingDirectory>
  </PropertyGroup>
    <PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='RelWithDebInfo|${PLATEFORM_BITS}'\">
    <LocalDebuggerCommand>${MSVCsuc_COMMAND}</LocalDebuggerCommand>
    <LocalDebuggerCommandArguments>${MSVCsuc_ARGS}</LocalDebuggerCommandArguments>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
    <LocalDebuggerWorkingDirectory>${MSVCsuc_WORKING_DIR}</LocalDebuggerWorkingDirectory>
  </PropertyGroup>
</Project>"
			)
			if(MSVCsuc_VERBOSE)
				message(STATUS "[MSVCsetUserCommand] Write ${CMAKE_CURRENT_BINARY_DIR}/${targetName}.vcxproj.user file")
				message(STATUS "   to execute ${MSVCsuc_COMMAND} ${MSVCsuc_ARGS}")
				message(STATUS "   from derectory ${MSVCsuc_WORKING_DIR}")
				message(STATUS "   on visual studio run debugger button")
			endif()

		else()
			message(WARNING "PLATEFORM_BITS is undefined...")
		endif()

	else()
		if(MSVCsuc_VERBOSE)
			message(WARNING "MSVCsetUserCommand is disable because too old MSVC is used (need MSVC10 2010 or newer)")
		endif()
	endif()

endfunction()
