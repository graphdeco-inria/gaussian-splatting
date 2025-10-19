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


# include "core/graphics/Image.hpp"
# include "core/assets/Config.hpp"
# include "core/assets/IFileLoader.hpp"

namespace sibr
{
	/** Represent a active_images.txt file use to select a subset of a scene images.
	\ingroup sibr_assets
	*/
	class SIBR_ASSETS_EXPORT ActiveImageFile : public IFileLoader
	{

	public:

		/**
		Set the number of images contained in the associated scene.
		\param n the number of images.
		*/
		void setNumImages(int n) {	_numImages = n; }

		/**
		Load an active cameras listing from a file on disk.
		\param filename the path to the file.
		\param verbose output additional informations to the standard output.
		\return a boolean indicating if the loading was successful or not.
		*/
		bool load( const std::string& filename, bool verbose = true ) override;
		
		/**
		Load an active cameras listing from a file on disk, expecting numImage images in total.
		The active images file does not indicate the total number of cameras in the scene, thus the additional parameter.
		\param filename the path to the file.
		\param numImages the total number of images in the associated scene.
		\param verbose output additional informations to the standard output.
		\return a boolean indicating if the load was successful or not.
		*/
		bool load( const std::string& filename, int numImages, bool verbose = true );
		
		/**
		Return a reference to a boolean vector indicating, for each picture of 
		the associated scene, if it is active or not.
		\return the boolean vector described above.
		*/
		const std::vector<bool>&	active( void ) const { return _active; }


	private:
		std::vector<bool> _active; ///< Flags denoting which images are active.
		int _numImages = 0; ///< Number of images.

	};


} // namespace sibr
