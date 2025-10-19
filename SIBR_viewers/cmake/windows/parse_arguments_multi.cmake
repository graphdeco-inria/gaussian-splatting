# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


if(NOT WIN32 OR __parse_arguments_multi_cmake_INCLUDED__)
	return()
else()
	set(__parse_arguments_multi_cmake_INCLUDED__ ON)
endif()

## This macro allow to process repeating multi value args from a given function which use cmake_parse_arguments module.
##
## cmake_parse_arguments multi args standard behavior:
##    function(foo)
##        cmake_parse_arguments(arg "" "" "MULTI" ${ARGN})
##        foreach(item IN LISTS arg_MULTI)
##            message(STATUS "${item}")
##        endforeach()
##    endfunction()
##    foo(MULTI x y MULTI z w)
##  The above code outputs 'z' and 'w'. It originally expected it to output all of 'x' 'y' 'z' 'w'.
##
## Using this macro inside a function which want to handle repeating multi args values
## will recursively iterate onto the multi tags list to process each sub list.
## It take as 1st argument the subTag flag to separate sub list from the main multi list.
## It take as 2nd argument the nameList of the main multi list (the multiValuesArgs from cmake_parse_arguments: here it is MULTI in the example)
## and that's why it is important that it should be a macro and not a function (to get access to external variable).
## Then you give the content of this list allowing to be processed by the macro.
##
## parse_arguments_multi macro call a parse_arguments_multi_function which do actually the process from the given sub-list.
## By default this function only print infos about what variables you are trying to pass/process (only verbose messages),
## but, by overloading this cmake function, you will be able to externalize the process of your multi argument list.
##
## Usage (into a function) : 
## parse_arguments_multi(<multiArgsSubTag> <multiArgsList> <multiArgsListContent> 
##      [NEED_RESULTS <multiArgsListSize>] [EXTRAS_FLAGS <...> <...> ...]
## )
##
## Simple usage example [user point of view]:
## foo(MULTI
##    SUB_MULTI x y
##    SUB_MULTI z w
## )
##
## Simple usage example [inside a function]:
##    function(foo)
##        cmake_parse_arguments(arg "" "" "MULTI" ${ARGN})
##        include(parse_arguments_multi)
##        function(parse_arguments_multi_function )
##          #message("I'm an overloaded cmake function used by parse_arguments_multi")
##          #message("I'm processing first part of my sub list: ${ARGN}")
##          message("ARGV0=${ARGV0}")
##          message("ARGV1=${ARGV1}")
##        endfunction()
##        parse_arguments_multi(SUB_MULTI arg_MULTI ${arg_MULTI}) ## this function will process recusively items of the sub-list [default print messages]
##    endfunction()
##
##  Will print:
##      ARGV0=z
##      ARGV1=w
##      ARGV0=x
##      ARGV1=y
##
## WARNING : DO NEVER ADD EXTRA THINGS TO parse_arguments_multi MACRO :
##          parse_arguments_multi(SUB_MULTI arg_MULTI ${arg_MULTI} EXTRAS foo bar SOMTHING) => will failed !!
## use EXTRAS_FLAGS instead !!
##
## Advanced usage example [user point of view]:
## bar(C:/prout/test.exe VERBOSE 
##      PLUGINS
##          PLUGIN_PATH_NAME x      PLUGIN_PATH_DEST w
##          PLUGIN_PATH_NAME a b    PLUGIN_PATH_DEST y
##          PLUGIN_PATH_NAME c
## )
##
## Advanced usage example [inside a function]:
##    function(bar execFilePathName)
##        cmake_parse_arguments(arg "VERBOSE" "" "PLUGINS" ${ARGN})
##
##        include(parse_arguments_multi)
##        function(parse_arguments_multi_function results)
##            cmake_parse_arguments(pamf "VERBOSE" "PLUGIN_PATH_DEST;EXEC_PATH" "" ${ARGN}) ## EXEC_PATH is for internal use
##            message("")
##            message("I'm an overloaded cmake function used by parse_arguments_multi from install_runtime function")
##            message("I'm processing first part of my sub list: ${ARGN}")
##            message("PLUGIN_PATH_NAME = ${pamf_UNPARSED_ARGUMENTS}")
##            message(pamf_VERBOSE = ${pamf_VERBOSE})
##            message("pamf_PLUGIN_PATH_DEST = ${pamf_PLUGIN_PATH_DEST}")
##            message(pamf_EXEC_PATH = ${pamf_EXEC_PATH})
##            if(NOT ${pamf_PLUGIN_PATH_DEST})
##              set(pamf_PLUGIN_PATH_DEST ${pamf_EXEC_PATH})
##            endif()
##            foreach(plugin ${pamf_UNPARSED_ARGUMENTS})
##              get_filename_component(pluginName ${plugin} NAME)
##              list(APPEND pluginsList ${pamf_PLUGIN_PATH_DEST}/${pluginName})
##            endforeach()
##            set(${results} ${pluginsList} PARENT_SCOPE)
##        endfunction()
##
##        if(arg_VERBOSE)
##            list(APPEND extra_flags_to_add VERBOSE) ## here we transmit the VERNOSE flag
##        endif()
##        get_filename_component(EXEC_PATH ${execFilePathName} PATH) ## will be the default value if PLUGIN_PATH_DEST option is not provided
##        list(APPEND extra_flags_to_add EXEC_PATH ${EXEC_PATH})  
##        list(LENGTH arg_PLUGINS arg_PLUGINS_count)
##        parse_arguments_multi(PLUGIN_PATH_NAME arg_PLUGINS ${arg_PLUGINS}
##                            NEED_RESULTS ${arg_PLUGINS_count}  ## this is used to check when we are in the first loop (in order to reset parse_arguments_multi_results)
##                            EXTRAS_FLAGS ${extra_flags_to_add} ## this is used to allow catching VERBOSE and PLUGIN_PATH_DEST flags of our overloaded function
##        )
##    endfunction()
##    message(parse_arguments_multi_results = ${parse_arguments_multi_results}) ## list of the whole pluginsList
##    #Will print w/x;a/y;b/y;C:/prout/c
##
##  NOTE that here, since our overloaded function need to provide a result list, we use the other parse_arguments_multi_function signature (the which one with a results arg)
##

function(parse_arguments_multi_function_default) ## used in case of you want to reset the default behavior of this function process
    message("[default function] parse_arguments_multi_function(ARGC=${ARGC} ARGV=${ARGV} ARGN=${ARGN})")
    message("This function is used by parse_arguments_multi and have to be overloaded to process sub list of multi values args")
endfunction()

function(parse_arguments_multi_function )   ## => the function to overload
    parse_arguments_multi_function_default(${ARGN})
endfunction()

## first default signature above
##------------------------------
## second results signature behind

function(parse_arguments_multi_function_default result) ## used in case of you want to reset the default behavior of this function process
    message("[default function] parse_arguments_multi_function(ARGC=${ARGC} ARGV=${ARGV} ARGN=${ARGN})")
    message("This function is used by parse_arguments_multi and have to be overloaded to process sub list of muluti values args")
endfunction()

function(parse_arguments_multi_function result)   ## => the function to overload
    parse_arguments_multi_function_default(result ${ARGN})
endfunction()

## => the macro to use inside your function which use cmake_parse_arguments
# NOTE: entry point of parse_arguments_multi, which is called from win3rdPart)
macro(parse_arguments_multi multiArgsSubTag multiArgsList #<${multiArgsList}> the content of the list
)
    # message (STATUS "")
    # message(STATUS "calling parse_arguemnts_multi defined in parse_arguments_multi.cmake:141")
    # message(STATUS "multiArgsSubTag = ${multiArgsSubTag}")	# CHECK_CACHED_VAR
    # message(STATUS "multiArgsList = ${multiArgsList}")	# it contains the name of the variable which is holding the list i.e w3p_MULTI_SET
    # message(STATUS "value of ${multiArgsList} = ${${multiArgsList}}") # a semicolon separated list of values passed to SET or MULTISET keyword in win3rdParty
    # message(STATUS "actual values ARGN = ${ARGN}")  # the same as ${${multiArgsList}}

    ## INFO
    ## starting from CMake 3.5 cmake_parse_arguments is not a module anymore and now is a native CMake command.
    ## the behaviour is different though
    ## In CMake 3.4, if you pass multiple times a multi_value_keyword, CMake returns the values of the LAST match
    ## In CMake 3.5 and above, CMake returns the whole list of values that were following that multi_value_keyword
    ## example:
    ## cmake_parse_arguments(
    ##			<prefix>
    ##			""		# options
    ##			""		# one value keywords
    ##			"MY_MULTI_VALUE_TAG"
    ##				MY_MULTI_VALUE_TAG value1 value2
    ##				MY_MULTI_VALUE_TAG value3 value4
    ##				MY_MULTI_VALUE_TAG value5 value6
    ##			)
    ## result in CMake 3.4
    ## <prefix>_MY_MULTI_VALUE_TAG = "value5;value6"
    ##
    ## result in CMake 3.8
    ## <prefix>_MY_MULTI_VALUE_TAG = "value5;value6"

    #include(CMakeParseArguments) #module CMakeParseArguments is obsolete since cmake 3.5
    # cmake_parse_arguments (<prefix> <options> <one_value_keywords> <multi_value_keywords> args)
    # <options> : options (flags) pass to the macro
    # <one_value_keywords> : options that neeed a value
    # <multi_value_keywords> : options that neeed more than one value
    cmake_parse_arguments(_pam "" "NEED_RESULTS" "${multiArgsSubTag};EXTRAS_FLAGS" ${ARGN})
    
    ## multiArgsList is the name of the list used by the multiValuesOption flag from the cmake_parse_arguments of the user function
    ## that's why we absolutly need to use MACRO here (and also for passing parse_arguments_multi_results when NEED_RESULTS flag is set)
    
    ## for debugging
    #message("")
    #message("[parse_arguments_multi] => ARGN = ${ARGN}")
    #message("_pam_NEED_RESULTS=${_pam_NEED_RESULTS}")
    #message("_pam_EXTRAS_FLAGS=${_pam_EXTRAS_FLAGS}")
    # foreach(var ${_pam_${multiArgsSubTag}})
    #     message("arg=${var}")
    # endforeach()

    if (${CMAKE_VERSION} VERSION_GREATER "3.5")
        # lets make ${_pam_${multiArgsSubTag}} behave as it is in version 3.4
        # that means, cmake_parse_arguments should have only the last values of a multi set for a given keyword

        # message("")
        # message("values in multiArgsList")
        # foreach(val ${${multiArgsList}})
        #     message(STATUS ${val})
        # endforeach()
        # message("end values in multiArgsList")


        set(lastIndexFound OFF)
        list(LENGTH ${multiArgsList} argnLength)
        # message(${argnLength})
        math(EXPR argnLength "${argnLength}-1")             # make last index a valid one
        set(recordIndex 0)
        set(records "")                                     # clear records list
        set(record0 "")                                    # clear first record list
        foreach(iter RANGE ${argnLength})
            list(GET ${multiArgsList} ${iter} value)
            # message(STATUS "index=${iter} value=${value}")
            if (${value} STREQUAL ${multiArgsSubTag})
                if (lastIndexFound)
                    list(APPEND records ${recordIndex})    # records store the list NAMES
                    math(EXPR recordIndex "${recordIndex}+1")
                    set(record${recordIndex} "")            # clear record list
                else ()
                    set(lastIndexFound ON)
                endif()

                set(lastIndex ${iter})
            else ()
                if (lastIndexFound)
                    # message(${value})
                    list(APPEND record${recordIndex} ${value})
                endif()
            endif()
        endforeach()

        # save the last list of values
        if (lastIndexFound)
            list(APPEND records ${recordIndex})    # records store the list NAMES
        endif()

        # set multiArgsList to make it behave like CMake 3.4
        # message("")
        # message("using my records")
        foreach(recordName ${records})
            # message(${recordName})
            # foreach(value ${record${recordName}})
            #     message(${value})
            # endforeach()
            # message("")
            set(_pam_${multiArgsSubTag} ${record${recordName}})
        endforeach()
        # message(${_pam_${multiArgsSubTag}})

        # message("")
        # message("using argn")
        # foreach(value ${ARGN})
        #     message(${value})
        # endforeach()
    endif() # end if cmake > 3.5

    # message("values with pam ${_pam_${multiArgsSubTag}}")

    ## check and init
    list(LENGTH ${multiArgsList} globalListCount)	# GLUT_TRACE: globalListCound=16 in CMake3.4 and CMake3.8
    # message(STATUS "nr items in multiArgsList: ${globalListCount}")
    math(EXPR globalListCount "${globalListCount}-1") ## because it will contain [multiArgsSubTag + ${multiArgsList}]
    if(_pam_NEED_RESULTS)
        if(${globalListCount} EQUAL ${_pam_NEED_RESULTS})
            ## first time we enter into this macro (because we call it recursively)
            unset(parse_arguments_multi_results)
        endif()
    endif()
    
    ## process the part of the multi agrs list
    ## ${ARGN} shouldn't be passed to the function in order to avoid missmatch size list ${multiArgsList} and _pam_${multiArgsSubTag}
    ## if you want to pass extra internal flags from your function to this callback, use EXTRAS_FLAGS
    if(_pam_NEED_RESULTS)
        parse_arguments_multi_function(parse_arguments_multi_function_result ${_pam_${multiArgsSubTag}} ${_pam_EXTRAS_FLAGS})
        list(APPEND parse_arguments_multi_results ${parse_arguments_multi_function_result})
    else()
        # message(STATUS "about to call parse_arguments_multi_function in parse_arguments_multi.cmake:177 ${_pam_${multiArgsSubTag}} and extra flags ${_pam_EXTRAS_FLAGS}")
        parse_arguments_multi_function(${_pam_${multiArgsSubTag}} ${_pam_EXTRAS_FLAGS})
    endif()

    ## remove just processed items from the main list to process (multiArgsList)
    list(REVERSE ${multiArgsList})
    list(LENGTH _pam_${multiArgsSubTag} subTagListCount)
    unset(ids)
    foreach(id  RANGE ${subTagListCount})
         list(APPEND ids ${id})
    endforeach()
    list(REMOVE_AT  ${multiArgsList} ${ids})
    list(REVERSE    ${multiArgsList})
    
    ## test if remain sub multi list to process (recursive call) or finish the process
    list(LENGTH ${multiArgsList} mainTagListCount)
    if(${mainTagListCount} GREATER 1)
        ## do not pass ${ARGN} just because it will re pass the initial 2 inputs args and we wont as they was consumed (in order to avoir conflicts)
        # message(STATUS "about to call a parse_arguments_multi but without knowing where the definition is going to be taken from")
        parse_arguments_multi(${multiArgsSubTag} ${multiArgsList} ${${multiArgsList}} 
                                NEED_RESULTS ${_pam_NEED_RESULTS} EXTRAS_FLAGS ${_pam_EXTRAS_FLAGS}
            )
    endif()
endmacro()
