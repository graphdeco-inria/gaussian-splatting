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


# include <boost/filesystem.hpp>
# include <fstream>
# include "core/assets/ActiveImageFile.hpp"

namespace sibr
{
	bool ActiveImageFile::load( const std::string& filename, bool verbose )
	{

		std::fstream	file(filename, std::ios::in);
		if(_numImages == 0 )
			SIBR_WRG << "No Images Loaded while loading '"<<filename<<"'"<<std::endl;
		

		// always create the array size of cameras so it can be queried.
		_active.resize(_numImages);	
		for(int i=0; i < _numImages; i++)
			_active[i]=false;

		if (file && !boost::filesystem::is_empty(filename))
		{
			while (file.eof() == false)
			{
				int imageId;
				file >>  imageId;
				_active[imageId] = true;
			}

			if( verbose )
				SIBR_FLOG << "'"<< filename <<"' successfully loaded." << std::endl;
			else
				std::cerr<< "." ;
			return true;
		}
	 	else {
			for(int i=0; i < _numImages; i++)
				_active[i]=true;
			if( verbose )
				SIBR_WRG << "file not found: '"<<filename<<"'"<<std::endl;
		}
		return false;
	}

	bool ActiveImageFile::load( const std::string& filename, int numImage, bool verbose )
	{
		_numImages = numImage;
		return load(filename, verbose);
	}

} // namespace sibr
