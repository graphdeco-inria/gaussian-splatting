/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */


#pragma once

# include <vector>
# include "core/system/Config.hpp"
# include "core/system/String.hpp"

namespace sibr
{
	/**
	* \addtogroup sibr_system
	* @{
	*/

#ifdef SIBR_OS_WINDOWS 
	/** setup console to allow color printing in console
	 */
	SIBR_SYSTEM_EXPORT void			setupConsole(void) ;
	/** restore console to no colors
	 */
	SIBR_SYSTEM_EXPORT void			restoreConsole(void) ;

#endif

	/** Load the whole file into a std::string
	 * \param filename the file path
	 * \return the loaded content */
	SIBR_SYSTEM_EXPORT std::string	loadFile( const std::string& filename );

	/** Create directory (if it doesn't exist already) 
	 * \param path the directory path
	 */
	SIBR_SYSTEM_EXPORT void			makeDirectory( const std::string& path );

	/** List content of directory, sorted alphabetically.
	 * \param path directory path
	 * \param listHidden should hidden files be listed
	 * \param includeSubdirectories should subdirectories be explored
	 * \param allowedExtensions a list of allowed extensions to filter the list with (for instance {"png", "bmp"})
	 * \return a list of file names/subpaths
	 * \note To get each element full path, use path + "/" + itemPath
	 */
	SIBR_SYSTEM_EXPORT std::vector<std::string>	listFiles(const std::string & path, const bool listHidden = false, const bool includeSubdirectories = false, const std::vector<std::string> & allowedExtensions = {});

	/** List content of directory, sorted alphabetically, including subdirectories. 
	 * \param path directory path
	 * \param listHidden should hidden directories be listed
	 * \return a list of directory names/subpaths
	 * \note To get each element full path, use path + "/" + itemPath
	 */
	SIBR_SYSTEM_EXPORT std::vector<std::string>	listSubdirectories(const std::string& path, const bool listHidden = false);

	/** Copy directory. 
	 * \param src source path
	 * \param dst destination path
	 * \return success boolean
	 */
	SIBR_SYSTEM_EXPORT bool copyDirectory(const std::string& src, const std::string& dst);

	/** Copy file.
	 * \param src source path
	 * \param dst destination path
	 * \param overwrite if the file already exists, should it be overwritten
	 * \return success boolean
	 */
	SIBR_SYSTEM_EXPORT bool copyFile(const std::string& src, const std::string& dst, const bool overwrite = false);

	/** Empty a directory (if it exist already) 
	 * \param path the directory path
	 */
	SIBR_SYSTEM_EXPORT void			emptyDirectory(const std::string& path);

	/** Test if a file exists.
	 *\param path the file path
	 *\return true if file exists
	 */
	SIBR_SYSTEM_EXPORT bool			fileExists( const std::string& path );

	/** Test if a directory exists.
	 *\param path the directory path
	 *\return true if directory exists
	 */
	SIBR_SYSTEM_EXPORT bool			directoryExists( const std::string& path );

	/** \return the available memory on windows system in Ko*/
	SIBR_SYSTEM_EXPORT size_t		getAvailableMem();

	/** \return the binary directory on windows system*/
	SIBR_SYSTEM_EXPORT std::string	getInstallDirectory();

	/** \return the binary directory on windows system*/
	SIBR_SYSTEM_EXPORT std::string	getBinDirectory();

	/**
	 * \param subfolder optional subfolder for subproject
	 * \return the binary directory on windows system
	 */
	SIBR_SYSTEM_EXPORT std::string	getShadersDirectory(const std::string & subfolder = "");

	/** \return the scripts directory on windows system*/
	SIBR_SYSTEM_EXPORT std::string	getScriptsDirectory();

	/** \return the resources directory on windows system*/
	SIBR_SYSTEM_EXPORT std::string	getResourcesDirectory();

	/** \return the user specific application directory */
	SIBR_SYSTEM_EXPORT std::string	getAppDataDirectory();

	/** 
	 * \param subfolder the subfolder to get
	 * \return the provided subfolder path on windows system
	 */
	SIBR_SYSTEM_EXPORT std::string	getInstallSubDirectory(const std::string & subfolder);

	/** Selection mode for the file picker. */
	enum FilePickerMode {
		Default, Save, Directory
	};

	/**
	 * Present a native OS file picker.
	 * \param selectedElement will contain the path to the element selected by the user if any.
	 * \param mode the mode to use, pick from Save, Directory, Default.
	 * \param directoryPath the initial directory to present to the user.
	 * \param extensionsAllowed a list of file extensions to allow: "obj,ply" for instance.
	 * \return true if an element was selected, else false.
	 * \warning '.' relative path is unsupported for directoryPath.
	 */
	SIBR_SYSTEM_EXPORT bool showFilePicker(std::string & selectedElement,
		const FilePickerMode mode, const std::string & directoryPath = "", const std::string & extensionsAllowed = "");

	/** Measure and print the timing of a task.
	 *\param s description
	 *\param f function to run
	 *\param args arguments for the function
	 */
	template<typename FunType, typename ...ArgsType>
	void taskTiming(const std::string & s, FunType && f, ArgsType && ... args) {
		const auto start = std::chrono::high_resolution_clock::now();
		f(args...);
		const auto end = std::chrono::high_resolution_clock::now();
		std::cout << s << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	};

	SIBR_SYSTEM_EXPORT std::istream& safeGetline(std::istream& is, std::string& t);

	/*** @} */
} // namespace sibr
