# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


## This file should be include and use only on WIN32 OS and once
## It allow to auto check/download and use a preconfigured 3rdParty binaries for cmake usage
## It use the downloadAndExtractZipFile cmake module to work.
##
if(__Win3rdParty_cmake_INCLUDED__)
	return()
else()
	set(__Win3rdParty_cmake_INCLUDED__ ON)
endif()


##
## To be sure to reset an empty cached variable but keep any other kind of variables
##
## Usage:
## check_cached_var(<var> <resetedCachedValue> <cacheType> <cacheDoc> [FORCE])
##
## <var> is the cached cmake variable you need to reset
## <resetedCachedValue> is the new default value of the reseted cached cmake variable
## <cacheType> is the kind of GUI cache input can be : FILEPATH; PATH; STRING or BOOL
## <cacheDoc> is the associated GUI cache input documentation display in the GUI
## FORCE option could be use to reset a cached variable even if it is not empty.
##
macro(check_cached_var var resetedCachedValue cacheType cacheDoc)
    # message(STATUS "inside check_cached_var macro. argn=${ARGN}")
    cmake_parse_arguments(ccv "FORCE" "" "" ${ARGN})

    if(ccv_FORCE)
        set(FORCE FORCE)
    else()
        set(FORCE )
    endif()

    if(NOT ${var} OR ccv_FORCE)
        unset(${var} CACHE)
        # message(STATUS "setting new cache value. var ${var} = ${resetedCachedValue}")
        set(${var}	"${resetedCachedValue}" CACHE ${cacheType} "${cacheDoc}" ${FORCE})
    endif()
endmacro()


##
## Win3rdParty function allow to specify a directory which contain all necessary windows dependenties.
## By uploading 3rdParty directory (which contain dependencies, *.lib, *.dll... for a specific version of compiler) onto Gforge file tab,
## you get back an URL of download you can give to this function with a directory name. So you can provide multiple 3rdParty version of same dependencies (MSVC11, MSVC12...).
## By providing a prefix to this function, you allow to use different kind of 3rdParty which can be handled by CMAKE OPTIONS depending on what your framework need for example.
##
## Usage 1:
##    Win3rdParty(<prefix> MSVC<XX> <DirName> <URL>
##					[MSVC<XX> <DirName> <URL>] [...]
##					[VCID] [DEFAULT_USE] [VERBOSE] )
##
## * <prefix> allow to identify which 3rdParty you process (prefix name)
## * MSVC<XX> flag could be MSVC11 or MSVC12 (any element of the MSVC_VERSIONS_LIST) and refer to a 3rdParty compiler with :
##   * <DirName> which will be the local pathName of the downloaded 3rdParty : relative to CMAKE_BINARY_DIR
##   * <URL> which is the link location of the 3rdParty zip
## * VCID flag will make available a cache variable ${prefix}_WIN3RDPARTY_VCID
## * DEFAULT_USE flag [ON|OFF] may be used to set default value of cmake cached variable : <prefix>_WIN3RDPARTY_USE [default to ON]
##
## WARNING:
## This function define CACHE variables you can use after :
## * ${prefix}_WIN3RDPARTY_USE : allow to check/downloaded win3rdParty dir (it will force the cached variables for this dependency folder generally <prefix>_DIR>)
## * ${prefix}_WIN3RDPARTY_DIR : where is your local win3rdParty dir (the PATH)
## * ${prefix}_WIN3RDPARTY_VCID : [if VCID flag is used] the MSVC id (commonly used to prefix/suffix library name, see boost or CGAL)
##
## If you want to add a win3rdParty version, please:
## 1- build dependencies on your local side with the compiler you want
## 2- build your own zip with your built dependencies
## 3- upload it (onto the forge where the project is stored) and copy the link location in order to use it for this function
## 4- if you just introduced a new MSVC version, add it to the MSVC_VERSIONS_LIST bellow
##
## In a second pass, you can also use this function to set necessary cmake cached variables in order to let cmake find packages of these 3rdParty.
##
## Usage 2:
##    win3rdParty(<prefix> [VERBOSE] MULTI_SET|SET
##          CHECK_CACHED_VAR <cmakeVar> <cmakeCacheType> [LIST] <cmakeValue> [DOC <stringToolTips>]
##        [ CHECK_CACHED_VAR <cmakeVar> <cmakeCacheType> [LIST] <cmakeValue> [DOC <stringToolTips>] ] [...]
##
## * MULTI_SET or SET flags are used to tell cmake that all next arguments will use repeated flags with differents entries (SET mean we will provide only one set of arguments, without repetition)
## * CHECK_CACHED_VAR are the repeated flag which contain differents entries
##      * <cmakeVar> is the cmake variable you want to be cached for the project
##      * <cmakeCacheType> is the kind of cmake variable (couble be: FILEPATH; PATH; STRING or BOOL) => see check_cached_var.
##      * LIST optional flag could be used with CHECK_CACHED_VAR when <cmakeCacheType> = STRING. It allow to handle multiple STRINGS value list.
##      * <cmakeValue> is the value of the variable (if FILEPATH, PATH or STRING: use quotes, if BOOL : use ON/OFF)
##      * DOC optional flag is used to have a tooltips info about this new cmake variable entry into the GUI (use quotes).
##
## Full example 1 :
##    win3rdParty(COMMON MSVC11 "win3rdParty-MSVC11" "https://path.to/an.archive.7z"
##                SET CHECK_CACHED_VAR SuiteSparse_DIR PATH "SuiteSparse-4.2.1" DOC "default empty doc"
##    )
##
## WARNING:
## For the 2nd usage (with MULTI_SET), if you planned to set some CACHED_VAR using/composed by ${prefix}_WIN3RDPARTY_* just set in this macro (usage 1),
## then (due to the not yet existing var) you will need to call this function 2 times :
## One for the 1st usage (downloading of the current compiler 3rdParty).
## One for the MLUTI_SET flag which will use existsing ${prefix}_WIN3RDPARTY_* cached var.
##
## Full example 2 :
##    win3rdParty(COMMON MSVC11    "win3rdParty-MSVC11" "https://path.to/an.archive.7z")
##    win3rdParty(COMMON MULTI_SET
##       CHECK_CACHED_VAR CGAL_INCLUDE_DIR  PATH "CGAL-4.3/include" DOC "default empty doc"
##       CHECK_CACHED_VAR CGAL_LIBRARIES	STRING LIST "debug;CGAL-4.3/lib${LIB_POSTFIX}/CGAL-${WIN3RDPARTY_COMMON_VCID}-mt-gd-4.3.lib;optimized;CGAL-4.3/lib${LIB_POSTFIX}/CGAL-${WIN3RDPARTY_COMMON_VCID}-mt-4.3.lib"
##
##
## WARNING: This function use internaly :
## * downloadAndExtractZipFile.cmake
## * parse_arguments_multi.cmake
## * check_cached_var macro
##
function(win3rdParty prefix )

    # ARGV: list of all arguments given to the macro/function
    # ARGN: list of remaining arguments

    if(NOT WIN32)
        return()
    endif()

    ## set the handled version of MSVC
    ## if you plan to add a win3rdParty dir to download with a new MSVC version: build the win3rdParty dir and add the MSCV entry here.
    set(MSVC_VERSIONS_LIST "MSVC17;MSVC11;MSVC12;MSVC14")

    #include(CMakeParseArguments)   # CMakeParseArguments is obsolete since cmake 3.5
    # cmake_parse_arguments (<prefix> <options> <one_value_keywords> <multi_value_keywords> args)
    # <options> : options (flags) pass to the macro
    # <one_value_keywords> : options that neeed a value
    # <multi_value_keywords> : options that neeed more than one value
    cmake_parse_arguments(w3p "VCID" "VERBOSE;TIMEOUT;DEFAULT_USE" "${MSVC_VERSIONS_LIST};MULTI_SET;SET" ${ARGN})

    # message(STATUS "value of w3p_VCID = ${w3p_VCID}")
    # message(STATUS "value of w3p_VERBOSE = ${w3p_VERBOSE}")
    # message(STATUS "value of w3p_TIMEOUT = ${w3p_TIMEOUT}")
    # message(STATUS "value of w3p_DEFAULT_USE = ${w3p_DEFAULT_USE}")

    # foreach (loop_var ${MSVC_VERSIONS_LIST})
            # message(STATUS "value of w3p_${loop_var} = ${w3p_${loop_var}}")
    # endforeach(loop_var)

    # message(STATUS "value of w3p_MULTI_SET = ${w3p_MULTI_SET}")
    # message(STATUS "value of w3p_SET = ${w3p_SET}")

    # message("values for MSVC = ${w3p_MSVC14}")

    if(NOT w3p_TIMEOUT)
        set(w3p_TIMEOUT 300)
    endif()

    if(NOT DEFINED w3p_DEFAULT_USE)
        set(w3p_DEFAULT_USE ON)
    endif()
	

    ## 1st use (check/update|download) :
    set(${prefix}_WIN3RDPARTY_USE ${w3p_DEFAULT_USE} CACHE BOOL "Use required 3rdParty binaries from ${prefix}_WIN3RDPARTY_DIR or download it if not exist")


    ## We want to test if each version of MSVC was filled by the function (see associated parameters)
    ## As CMake is running only for one version of MSVC, if that MSVC version was filled, we get back associated parameters,
    ## otherwise we can't use the downloadAndExtractZipFile with win3rdParty.
    set(enableWin3rdParty OFF)
	
	foreach(MSVC_VER ${MSVC_VERSIONS_LIST})
		if(${MSVC_VER} AND w3p_${MSVC_VER} OR ${MSVC_TOOLSET_VERSION} EQUAL 143 AND ${MSVC_VER} STREQUAL "MSVC17") 
			list(LENGTH w3p_${MSVC_VER} count)
			if("${count}" LESS "2")
				#message(WARNING "You are using ${MSVC_VER} with ${prefix}_WIN3RDPARTY_USE=${${prefix}_WIN3RDPARTY_USE}, but win3rdParty function isn't filled for ${MSVC_VER}!")
			else()
				list(GET w3p_${MSVC_VER} 0 Win3rdPartyName)
				list(GET w3p_${MSVC_VER} 1 Win3rdPartyUrl)
				if(w3p_VCID)
					## try to get the VcId of MSVC. See also MSVC_VERSION cmake var in the doc.
					string(REGEX REPLACE "MS([A-Za-z_0-9-]+)" "\\1" vcId ${MSVC_VER})
					string(TOLOWER ${vcId} vcId)
					set(${prefix}_WIN3RDPARTY_VCID "${vcId}0" CACHE STRING "the MSVC id (commonly used to prefix/suffix library name, see boost or CGAL)")
					mark_as_advanced(${prefix}_WIN3RDPARTY_VCID)
				endif()
				set(enableWin3rdParty ON)
				set(suffixCompilerID ${MSVC_VER})
				break()
			endif()
		endif()
	endforeach()
    ## If previous step succeed to get MSVC dirname and URL of the current MSVC version, use it to auto download/update the win3rdParty dir
    if(enableWin3rdParty AND ${prefix}_WIN3RDPARTY_USE)

        if(IS_ABSOLUTE "${Win3rdPartyName}")
        else()
            set(Win3rdPartyName "${CMAKE_BINARY_DIR}/${Win3rdPartyName}")
        endif()

        if(NOT EXISTS "${Win3rdPartyName}")
            file(MAKE_DIRECTORY ${Win3rdPartyName})
        endif()

        include(downloadAndExtractZipFile)
        downloadAndExtractZipFile(  "${Win3rdPartyUrl}"                             ## URL link location
                                    "Win3rdParty-${prefix}-${suffixCompilerID}.7z"  ## where download it: relative path, so default to CMAKE_BINARY_DIR
                                    "${Win3rdPartyName}"                            ## where extract it : fullPath (default relative to CMAKE_BINARY_DIR)
            CHECK_DIRTY_URL "${Win3rdPartyName}/Win3rdPartyUrl"                     ## last downloaded url file : fullPath (default relative to CMAKE_BINARY_DIR)
            TIMEOUT ${w3p_TIMEOUT}
            VERBOSE ${w3p_VERBOSE}
        )
        file(GLOB checkDl "${Win3rdPartyName}/*")
        list(LENGTH checkDl checkDlCount)
        if("${checkDlCount}" GREATER "1")
        else()
            message("The downloadAndExtractZipFile didn't work...?")
            set(enableWin3rdParty OFF)
        endif()
    endif()

    ## Try to auto set ${prefix}_WIN3RDPARTY_DIR or let user set it manually
    set(${prefix}_WIN3RDPARTY_DIR "" CACHE PATH "windows ${Win3rdPartyName} dir to ${prefix} dependencies of the project")

    if(NOT ${prefix}_WIN3RDPARTY_DIR AND ${prefix}_WIN3RDPARTY_USE)
        if(EXISTS "${Win3rdPartyName}")
            unset(${prefix}_WIN3RDPARTY_DIR CACHE)
            set(${prefix}_WIN3RDPARTY_DIR "${Win3rdPartyName}" CACHE PATH "dir to ${prefix} dependencies of the project")
        endif()
    endif()

    if(EXISTS ${${prefix}_WIN3RDPARTY_DIR})
        message(STATUS "Found a 3rdParty ${prefix} dir : ${${prefix}_WIN3RDPARTY_DIR}.")
        set(enableWin3rdParty ON)
    elseif(${prefix}_WIN3RDPARTY_USE)
        message(WARNING "${prefix}_WIN3RDPARTY_USE=${${prefix}_WIN3RDPARTY_USE} but ${prefix}_WIN3RDPARTY_DIR=${${prefix}_WIN3RDPARTY_DIR}.")
        set(enableWin3rdParty OFF)
    endif()

    ## Final check
    if(NOT enableWin3rdParty)
        message("Disable ${prefix}_WIN3RDPARTY_USE (cmake cached var will be not set), due to a win3rdParty problem.")
        message("You still can set ${prefix}_WIN3RDPARTY_DIR to an already downloaded Win3rdParty directory location.")
        set(${prefix}_WIN3RDPARTY_USE OFF CACHE BOOL "Use required 3rdParty binaries from ${prefix}_WIN3RDPARTY_DIR or download it if not exist" FORCE)
    endif()

    ## 2nd use : handle multi values args to set cached cmake variables in order to ease the next find_package call
    if(${prefix}_WIN3RDPARTY_USE AND ${prefix}_WIN3RDPARTY_DIR)
        if(w3p_VERBOSE)
            message(STATUS "Try to set cmake cached variables for ${prefix} required libraries directly from : ${${prefix}_WIN3RDPARTY_DIR}.")
        endif()

        include(parse_arguments_multi)
        # message (STATUS "before defining an override of parse_arguments_multi_function")
        function(parse_arguments_multi_function ) ## overloaded function to handle all CHECK_CACHED_VAR values list (see: parse_arguments_multi)
            # message(STATUS "inside overloaded parse_arguments_multi_function defined in Win3rdParty.cmake")
            # message(STATUS ${ARGN})
            ## we know the function take 3 args : var cacheType resetedCachedValue (see check_cached_var)
            cmake_parse_arguments(pamf "" "DOC" "LIST" ${ARGN})

            ## var and cacheType are mandatory (with the resetedCachedValue)
            set(var         ${ARGV0})
            set(cacheType   ${ARGV1})
            # message(STATUS "var=${var} and cacheType=${cacheType} list=${pamf_LIST}")
            if(pamf_DOC)
                set(cacheDoc    ${pamf_DOC})
            else()
                set(cacheDoc    "")
            endif()
            if(pamf_LIST)
                set(value ${pamf_LIST})
            else()
                # message("USING ARGV2 with value ${ARGV2}")
                set(value ${ARGV2})
            endif()
            # message("inside override function in Win3rdparty.cmake value+ ${value}")
            if("${cacheType}" MATCHES "PATH" AND EXISTS "${${prefix}_WIN3RDPARTY_DIR}/${value}")
                # message("math with path")
                set(resetedCachedValue "${${prefix}_WIN3RDPARTY_DIR}/${value}") ## path relative to ${prefix}_WIN3RDPARTY_DIR
            elseif ("${cacheType}" MATCHES "PATH" AND EXISTS "${${prefix}_WIN3RDPARTY_DIR}")
                set(resetedCachedValue "${${prefix}_WIN3RDPARTY_DIR}") ## path relative to ${prefix}_WIN3RDPARTY_DIR
            elseif("${cacheType}" MATCHES "STRING")
                foreach(var IN LISTS value)
                    if(EXISTS "${${prefix}_WIN3RDPARTY_DIR}/${var}")
                        list(APPEND resetedCachedValue "${${prefix}_WIN3RDPARTY_DIR}/${var}") ## string item of the string list is a path => make relative to ${prefix}_WIN3RDPARTY_DIR
                    else()
                        list(APPEND resetedCachedValue ${var}) ## string item of the string list is not an existing path => simply use the item
                    endif()
                endforeach()
            else()
                set(resetedCachedValue "${value}") ## could be a BOOL or a STRING
            endif()

            ## call our macro to reset cmake cache variable if empty
            check_cached_var(${var} "${resetedCachedValue}" ${cacheType} "${cacheDoc}" FORCE)

        endfunction()
        # message (STATUS "after defining an override of parse_arguments_multi_function")

        if(w3p_MULTI_SET)
            parse_arguments_multi(CHECK_CACHED_VAR w3p_MULTI_SET ${w3p_MULTI_SET}) ## internaly will call our overloaded parse_arguments_multi_function
        elseif(w3p_SET)
            # message("calling set version of parse_arguments_multi with w3p_set = ${w3p_SET}")
            parse_arguments_multi(CHECK_CACHED_VAR w3p_SET ${w3p_SET})
        endif()

    endif()

endfunction()

## cmake variables introspection to globally activate/deactivate ${prefix}_WIN3RDPARTY_USE
## This "one shot" call (only one for the next cmake configure) will automatically then reset the global variable WIN3RDPARTY_USE to UserDefined (do nothing).
## use (call it) before and after the call of all your win3rdParty functions
function(Win3rdPartyGlobalCacheAction )
	set(WIN3RDPARTY_USE "UserDefined" CACHE STRING "Choose how to handle all cmake cached *_WIN3RDPARTY_USE for the next configure.\nCould be:\nUserDefined [default]\nActivateAll\nDesactivateAll" )
	set_property(CACHE WIN3RDPARTY_USE PROPERTY STRINGS "UserDefined;ActivateAll;DesactivateAll" )
	if(${WIN3RDPARTY_USE} MATCHES "UserDefined")
	else()
		if(${WIN3RDPARTY_USE} MATCHES "ActivateAll")
			set(win3rdPvalue ON)
		elseif(${WIN3RDPARTY_USE} MATCHES "DesactivateAll")
			set(win3rdPvalue OFF)
		endif()
		get_cmake_property(_variableNames CACHE_VARIABLES)
		foreach (_variableName ${_variableNames})
			string(REGEX MATCH 	"[A-Za-z_0-9-]+_WIN3RDPARTY_USE" win3rdpartyUseCacheVar ${_variableName})
			if(win3rdpartyUseCacheVar)
				string(REGEX REPLACE "([A-Za-z_0-9-]+_WIN3RDPARTY_USE)" "\\1" win3rdpartyUseCacheVar ${_variableName})
				set(${win3rdpartyUseCacheVar} ${win3rdPvalue} CACHE BOOL "Use required 3rdParty binaries from ${prefix}_WIN3RDPARTY_DIR or download it if not exist" FORCE)
				message(STATUS "${win3rdpartyUseCacheVar} cached variable set to ${win3rdPvalue}.")
			endif()
		endforeach()
		set(WIN3RDPARTY_USE "UserDefined" CACHE STRING "Choose how to handle all cmake cached *_WIN3RDPARTY_USE for the next configure.\nCould be:\nUserDefined [default]\nActivateAll\nDesactivateAll" FORCE)
		message(STATUS "reset WIN3RDPARTY_USE to UserDefined.")
	endif()
	mark_as_advanced(WIN3RDPARTY_USE)
endfunction()
