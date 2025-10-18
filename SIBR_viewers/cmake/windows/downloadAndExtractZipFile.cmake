# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


## downloadAndExtractZipFile cmake function
## Provide a way to download zip file from public internet ZIP_URL host
## and to extract it in a specific EXCTRATED_ZIP_PATH destination.
## This function use 7-Zip external tool to maximize the compatibles formats.
## This will be not download again if the EXCTRATED_ZIP_PATH already exist and DL_FORCE is set to OFF.
## This will try to unzip file if already exist in the ZIP_DL_PATH.
##
## If EXCTRATED_ZIP_PATH and/or ZIP_DL_PATH are not full path,
## it will be interpreted relative to CMAKE_BINARY_DIR
##
## Usage example :
## include(downloadAndExtractZipFile)
## downloadAndExtractZipFile(
## 	http://www.cs.cornell.edu/~snavely/bundler/distr/bundler-v0.4-source.zip
## 	${CMAKE_BINARY_DIR}/Bundler/bundler-v0.4-source.zip
## 	${CMAKE_BINARY_DIR}/Bundler
##  [DL_FORCE ON|OFF]
##  [TIMEOUT]
##  [CHECK_DIRTY_URL]
## )
##
## option DL_FORCE will redownload the zip file [deafult to OFF]
## option TIMEOUT will end the unzip process after this period of time [default to 600s]
## option CHECK_DIRTY_URL will write into the given file the downloaded URL and then,
## next time, if the URL was updated, it detect it with this file
## and will download the last version. This prevent to alway set manually DL_FORCE to ON...
##
if(__downloadAndExtractZipFile_cmake_INCLUDED__)
	return()
else()
	set(__downloadAndExtractZipFile_cmake_INCLUDED__ ON)
endif()

function(downloadAndExtractZipFile ZIP_URL ZIP_DL_PATH EXCTRATED_ZIP_PATH)

    # message(STATUS "zipUrl=${ZIP_URL} zipDlPath=${ZIP_DL_PATH} extractedZipPath=${EXCTRATED_ZIP_PATH}")
    cmake_parse_arguments(dwnlezf "" "VERBOSE;DL_FORCE;TIMEOUT;CHECK_DIRTY_URL" "" ${ARGN})

	set(PROGRAMFILESx86 "PROGRAMFILES(x86)")

    ## Check entries mandatory args
    if(IS_ABSOLUTE "${ZIP_DL_PATH}")
    else()
        set(ZIP_DL_PATH "${CMAKE_BINARY_DIR}/${ZIP_DL_PATH}")
    endif()
    if(IS_ABSOLUTE "${EXCTRATED_ZIP_PATH}")
    else()
        set(EXCTRATED_ZIP_PATH "${CMAKE_BINARY_DIR}/${EXCTRATED_ZIP_PATH}")
    endif()
    if(NOT EXISTS "${EXCTRATED_ZIP_PATH}")
        file(MAKE_DIRECTORY ${EXCTRATED_ZIP_PATH})
    endif()

	# SB: Once, one of downloaded zip was corrupted by an error message coming from the server.
	if(EXISTS "${ZIP_DL_PATH}")
		# So I check for removing such corrupted files
		message("Removing previous ${ZIP_DL_PATH} (might be corrupted)")
		file(REMOVE "${ZIP_DL_PATH}")
		if(EXISTS "${dwnlezf_CHECK_DIRTY_URL}")
			# and remove the previous (corrupted) made 'Win3rdPartyUrl' file
			file(REMOVE "${dwnlezf_CHECK_DIRTY_URL}")
		endif()
	endif()

    ## Check entries optional args
	macro(readDirtyUrl )
		if(dwnlezf_CHECK_DIRTY_URL)
			if(IS_ABSOLUTE "${dwnlezf_CHECK_DIRTY_URL}")
			else()
				set(dwnlezf_CHECK_DIRTY_URL "${CMAKE_BINARY_DIR}/${dwnlezf_CHECK_DIRTY_URL}")
			endif()
			get_filename_component(unzipDir 	${EXCTRATED_ZIP_PATH} NAME)
			get_filename_component(unzipPath 	${EXCTRATED_ZIP_PATH} PATH)
			message(STATUS "Checking ${unzipDir} [from ${unzipPath}]...")
			if(EXISTS "${dwnlezf_CHECK_DIRTY_URL}")
				get_filename_component(CHECK_DIRTY_URL_FILENAME ${dwnlezf_CHECK_DIRTY_URL} NAME)
				file(STRINGS "${dwnlezf_CHECK_DIRTY_URL}" contents)
				list(GET contents 0 downloadURL)
				list(REMOVE_AT contents 0)
				if("${downloadURL}" MATCHES "${ZIP_URL}")
					if(dwnlezf_VERBOSE)
						message(STATUS "Your downloaded version (URL) seems to be up to date. Let me check if nothing is missing... (see ${dwnlezf_CHECK_DIRTY_URL}).")
					endif()
					file(GLOB PATHNAME_PATTERN_LIST "${EXCTRATED_ZIP_PATH}/*") ## is there something inside the downloaded destination ?
					unset(NAME_PATTERN_LIST)
					foreach(realPathPattern ${PATHNAME_PATTERN_LIST})
						get_filename_component(itemName ${realPathPattern} NAME)
						list(APPEND NAME_PATTERN_LIST ${itemName})
					endforeach()
					if(NAME_PATTERN_LIST)
						foreach(item ${contents})
							list(FIND NAME_PATTERN_LIST ${item} id)
							if(${id} MATCHES "-1")
								message(STATUS "${item} is missing, your downloaded version content changed, need to redownload it.")
								set(ZIP_DL_FORCE ON)
								break()
							else()
								list(REMOVE_AT NAME_PATTERN_LIST ${id})
								set(ZIP_DL_FORCE OFF)
							endif()
						endforeach()
						if(NOT ZIP_DL_FORCE AND NAME_PATTERN_LIST)
							message("Yours seems to be up to date (regarding to ${CHECK_DIRTY_URL_FILENAME})!\nBut there are additional files/folders into your downloaded destination (feel free to clean it if you want).")
							foreach(item ${NAME_PATTERN_LIST})
								if(item)
									message("${item}")
								endif()
							endforeach()
						endif()
					endif()
				else()
					set(ZIP_DL_FORCE ON)
					message(STATUS "Your downloaded version is dirty (too old).")
				endif()
			else()
				file(GLOB PATHNAME_PATTERN_LIST "${EXCTRATED_ZIP_PATH}/*") ## is there something inside the downloaded destination ?
				if(NOT PATHNAME_PATTERN_LIST)
					message("We found nothing into ${EXCTRATED_ZIP_PATH}, we will try to download it for you now.")
				endif()
				set(ZIP_DL_FORCE ON)
			endif()
		endif()
	endmacro()
	readDirtyUrl()
	if(NOT ZIP_DL_FORCE)
		return() ## do not need to further (as we are up to date, just exit the function
	endif()

	macro(writeDirtyUrl )
		if(dwnlezf_CHECK_DIRTY_URL)
			file(WRITE "${dwnlezf_CHECK_DIRTY_URL}" "${ZIP_URL}\n")
			file(GLOB PATHNAME_PATTERN_LIST "${EXCTRATED_ZIP_PATH}/*") ## is there something inside the downloaded destination ?
			unset(NAME_PATTERN_LIST)
			foreach(realPathPattern ${PATHNAME_PATTERN_LIST})
				get_filename_component(itemName ${realPathPattern} NAME)
				list(APPEND NAME_PATTERN_LIST ${itemName})
			endforeach()
			if(NAME_PATTERN_LIST)
				foreach(item ${NAME_PATTERN_LIST})
					file(APPEND "${dwnlezf_CHECK_DIRTY_URL}" "${item}\n")
				endforeach()
			endif()
		endif()
	endmacro()

	if(dwnlezf_DL_FORCE)
        set(ZIP_DL_FORCE ON)
    endif()

	if(NOT dwnlezf_TIMEOUT)
		set(dwnlezf_TIMEOUT 600)
	endif()
	math(EXPR dwnlezf_TIMEOUT_MIN "${dwnlezf_TIMEOUT}/60")

	macro(unzip whichZipFile)
		if(NOT SEVEN_ZIP_CMD)
			find_program(SEVEN_ZIP_CMD NAMES 7z 7za p7zip DOC "7-zip executable" PATHS "$ENV{PROGRAMFILES}/7-Zip" "$ENV{${PROGRAMFILESx86}}/7-Zip" "$ENV{ProgramW6432}/7-Zip")
		endif()
		if(SEVEN_ZIP_CMD)
            if(dwnlezf_VERBOSE)
                message(STATUS "UNZIP: please, WAIT UNTIL ${SEVEN_ZIP_CMD} finished...\n(no more than ${dwnlezf_TIMEOUT_MIN} min)")
            else()
                message(STATUS "UNZIP...wait...")
            endif()
			execute_process( 	COMMAND ${SEVEN_ZIP_CMD} x ${whichZipFile} -y
								WORKING_DIRECTORY ${EXCTRATED_ZIP_PATH}	TIMEOUT ${dwnlezf_TIMEOUT}
								RESULT_VARIABLE resVar OUTPUT_VARIABLE outVar ERROR_VARIABLE errVar
							)
			if(${resVar} MATCHES "0")
                if(dwnlezf_VERBOSE)
                    message(STATUS "SUCESS to unzip in ${EXCTRATED_ZIP_PATH}. Now we can remove the downloaded zip file.")
                endif()
				execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${whichZipFile})
				mark_as_advanced(SEVEN_ZIP_CMD)
			else()
				message(WARNING "something wrong in ${EXCTRATED_ZIP_PATH}\n with \"${SEVEN_ZIP_CMD} x ${whichZipFile} -y\", redo or try to unzip by yourself...")
				message("unzip: resVar=${resVar}")
				message("unzip: outVar=${outVar}")
				message("unzip: errVar=${errVar}")
				message("unzip: failed or canceled or timeout")
			endif()
		else()
			message(WARNING "You need 7zip (http://www.7-zip.org/download.html) to unzip the downloaded dir.")
			set(SEVEN_ZIP_CMD "" CACHE FILEPATH "7-zip executable")
			mark_as_advanced(CLEAR SEVEN_ZIP_CMD)
		endif()
	endmacro()

    if(dwnlezf_VERBOSE)
        message(STATUS "Trying to look ${ZIP_DL_PATH} if a zip file exist...")
    endif()
	if(EXISTS "${ZIP_DL_PATH}")

		## already downloaded, so just unzip it
		unzip(${ZIP_DL_PATH})
		writeDirtyUrl()

	elseif(ZIP_DL_FORCE)

		## the download part (+ unzip)
		message(STATUS "Let me try to download package for you : ${ZIP_URL}")
        if(dwnlezf_VERBOSE)
            message(STATUS "Downloading...\n   SRC=${ZIP_URL}\n   DEST=${ZIP_DL_PATH}.tmp\n   INACTIVITY_TIMEOUT=180s")
        endif()
		file(DOWNLOAD ${ZIP_URL} ${ZIP_DL_PATH}.tmp INACTIVITY_TIMEOUT 360 STATUS status SHOW_PROGRESS)

		list(GET status 0 numResult)
		if(${numResult} MATCHES "0")

            if(dwnlezf_VERBOSE)
                message(STATUS "Download succeed, so let me rename the tmp file to unzip it")
            endif()
			execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${ZIP_DL_PATH}.tmp ${ZIP_DL_PATH})
			unzip(${ZIP_DL_PATH})
			writeDirtyUrl()

		else()

			list(GET status 1 errMsg)
			message(WARNING "DOWNLOAD ${ZIP_URL} to ${ZIP_DL_PATH} failed\n:${errMsg}")
			message(WARNING "OK, you need to download the ${ZIP_URL} manually and put it into ${ZIP_DL_PATH}")
			message("Take a look at the project website page to check available URL.")

		endif()

	endif()

	## clean up the tmp downloaded file
	if(EXISTS "${ZIP_DL_PATH}.tmp")
		execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${ZIP_DL_PATH}.tmp)
	endif()

endfunction()
