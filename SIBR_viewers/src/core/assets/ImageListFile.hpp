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
# include "core/assets/ActiveImageFile.hpp"

namespace sibr
{
	/** Represents a list of input images.
	\ingroup sibr_assets
	*/
	class SIBR_ASSETS_EXPORT ImageListFile : public IFileLoader
	{
	public:

		/** Image infos. */
		struct Infos
		{
			std::string		filename; ///< image filename.
			uint			camId; ///< Associated camera ID.
			uint			width; ///< Image width.
			uint			height; ///< Image height.
		};

	public:

		/** Load the list file from disk.
		\param filename path to the list file
		\param verbose display information
		\return a boolean denoting success
		*/
		bool load( const std::string& filename , bool verbose = true);

		/** Images information.
		 * \return each image infos.
		 */
		const std::vector<Infos>&	infos( void ) const { return _infos; }

		/** Image absename.
		 *\return the basename
		 **/
		const std::string&			basename( void ) const { return _basename; }

		/** Load images.
			\return the loaded images
		*/
		template <class TImage>
		std::vector<TImage>			loadImages( void ) const;

		/** Load images, applying an active images file filter. 
			\param ac the active list file
			\return the loaded images
			\note Non-active images are present but empty.
		*/
		template <class TImage>
		std::vector<TImage>			loadImages( const ActiveImageFile& ac) const;
		

	private:
		std::vector<Infos>		_infos; ///< Image infos.
		std::string				_basename; ///< Root name.

	};

	///// DEFINITIONS /////


	template <class TImage>
	std::vector<TImage>			ImageListFile::loadImages( const ActiveImageFile& ac ) const {
		std::vector<TImage> out;

		SIBR_LOG << "[ImageListFile] loading images";
		out.resize(_infos.size());
		if (_infos.empty() == false)
		{
			#pragma omp parallel for
			for (int i = 0; i < _infos.size(); ++i)
				if( ac.active()[i] )
					out[i].load(_basename + "/" + _infos.at(i).filename, false);
		}
		else
			SIBR_WRG << "cannot load images (ImageListFile is empty. Did you use ImageListFile::load(...) before ?";

		std::cout << std::endl;
		return out;
	}

	template <class TImage>
	std::vector<TImage>			ImageListFile::loadImages( void ) const {
		std::vector<TImage> out;

		std::cerr << "[ImageListFile] loading images";
		out.resize(_infos.size());
		if (_infos.empty() == false)
		{
			#pragma omp parallel for
			for (int i = 0; i < _infos.size(); ++i)
				out[i].load(_basename + "/" + _infos.at(i).filename, false);
		}
		else
			SIBR_WRG << "cannot load images (ImageListFile is empty. Did you use ImageListFile::load(...) before ?";

		return out;
	}

} // namespace sibr
