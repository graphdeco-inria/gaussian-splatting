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

#include "Config.hpp"
#include <core/graphics/Image.hpp>

#include <boost/filesystem.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>


namespace sibr {

	/** \brief Utility class to crop and rescale images, especially for uniformizing IBR datasets.
	* \ingroup sibr_imgproc
	*/
	class SIBR_IMGPROC_EXPORT CropScaleImageUtility
	{
	public:

		/** Image infos. */
		struct Image {
			std::string	filename; ///< Image file name.
			unsigned	width; ///< Image width.
			unsigned	height; ///< Image height.
		};

		/** Load a list_images.txt file and extract the image paths.
		 * \param inputFileName path to the listing
		 * \return a list of image paths
		 */
		std::vector<std::string> getPathToImgs(const std::string & inputFileName);
		
		/**
		 * Log processing informations to a file.
		 * \param resolution the estimated resolution
		 * \param nrImages the number of images
		 * \param elapsedTime the time taken by the processing
		 * \param wasTransformed was transforamtion applied
		 * \param log_file_name the destination file path
		 */
		void logExecution(const sibr::Vector2i & resolution, unsigned nrImages, long long elapsedTime, bool wasTransformed, const char* log_file_name);
		
		/**
		 * Save a list of images to a list_images.txt file, where each image has a line "name w h".
		 * \param path_to_file the destination file path
		 * \param listOfImages the images to save to the list
		 */
		void writeListImages(const std::string path_to_file, const std::vector<Image> & listOfImages);

		/**
		 * Write a scale float factor to a text file.
		 * \param path_to_file the destination file path
		 * \param scaleFactor the value to write
		 */
		void writeScaleFactor(const std::string path_to_file, float scaleFactor);

		/**
		 * Write a resolution to a text file, as "w h".
		 * \param path_to_file the destination file path
		 * \param targetResolution the resolution to write
		 */
		void writeTargetResolution(const std::string path_to_file, const sibr::Vector2i & targetResolution);

		/**
		 * Extract an image resolution from a "wxh" string.
		 * \param param the string to parse
		 * \return the resolution
		 */
		sibr::Vector2i parseResolution(const std::string & param);

	};
}