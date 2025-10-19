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

#include "core/assets/Config.hpp"

#include <vector>
#include <string>

namespace sibr
{

	/** Singleton used to store a list of plausible path to look for (based on the ibr_resources.ini)
	\ingroup sibr_assets
	*/
	class SIBR_ASSETS_EXPORT Resources
	{
	public:
		/// Our singleton
		static Resources* Instance();

	protected:
		/// Constructor.
		Resources();

		/// Destructor
		virtual ~Resources();

	public:
		/** Look for the filename into plausible resource paths.
		 * \param filename file name
		 * \param success was the file found in the registered locations
		 * \return the full file path
		 */
		std::string getResourceFilePathName(std::string const & filename, bool & success);

		/** Look for the filename into plausible resource paths.
		 * \param filename file name
		 * \return the full file path
		 */
		std::string getResourceFilePathName(std::string const & filename);

	protected:
		std::vector<std::string>    _rscPaths; ///< List of directories to check into.
		static Resources *          _instance; ///< Singleton.
	};

} // namespace sibr
