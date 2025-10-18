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

# include "core/assets/Config.hpp"

namespace sibr
{
	/** General file loading interface.
	\ingroup sibr_assets
	*/
	class SIBR_ASSETS_EXPORT IFileLoader
	{
	public:

		/** Destructor. */
		virtual ~IFileLoader( void ) { }

		/** Load the file content from disk.
		\param filename path to the file
		\param verbose display information
		\return a boolean denoting success
		*/
		virtual bool load( const std::string& filename, bool verbose = true ) = 0;
	};

} // namespace sibr
