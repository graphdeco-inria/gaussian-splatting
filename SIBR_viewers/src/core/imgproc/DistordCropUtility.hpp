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
#include <core/system/Vector.hpp>
#include <core/system/Array2d.hpp>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp> 

#include<fstream>
#include <queue>


namespace sibr {

	/** \brief Helpers for cropping undistorted dataset images so that margins are removed, while keeping the dataset consistent.
	* \ingroup sibr_imgproc
	*/
	class SIBR_IMGPROC_EXPORT DistordCropUtility
	{
	public:

		/** Image crop boundaries . */
		struct Bounds
		{
			/// Default constructor.
			Bounds() {}

			/** Initialize with an image boundaries.
			 *\param img the image to use
			 **/
			Bounds(const sibr::ImageRGB & img) {
				xMax = (int)img.w() - 1;
				xMin = 0;
				yMax = (int)img.h() - 1;
				yMin = 0;
				xRatio = 1.0f;
				yRatio = 1.0f;
			}

			/** Initialize with a given resolution.
			 *\param res the resolution
			 **/
			Bounds(const sibr::Vector2i & res) {
				xMax = res.x() - 1;
				xMin = 0;
				yMax = res.y() - 1;
				yMin = 0;
				xRatio = 1.0f;
				yRatio = 1.0f;
			}

			/** \return a string representing the bounds, for logging. */
			std::string display() const {
				std::stringstream s;
				s << "[" << xMin << ", " << xMax << "]x[" << yMin << ", " << yMax << "]";
				return s.str();
			}

			int xMax = 0; ///< Max x value.
			int xMin = 0; ///< Min x value.
			int yMax = 0; ///< Max y value.
			int yMin = 0; ///< Min y value.
			int width = 0; ///< Region width.
			int height = 0; ////< Region height.

			float xRatio = 1.0f; ///< Scaling ratio along X axis.
			float yRatio = 1.0f; ///< Scaling ratio along Y axis.
		};

		/** Check if a pixel color is close to a reference color.
		 *\param pixelColor the color to test
		 *\param backgroundColor the reference color
		 *\param threshold_black_color the tolerance threshold
		 *\return true if ||pixel - background||^2 < threshold
		 */
		bool isBlack(const sibr::Vector3ub & pixelColor, Vector3i backgroundColor, int threshold_black_color);

		/*
		* Check if a file name is made out only of digits and not letters (like texture file names).
		* \param s the filename to test
		* \return true if the string only contains digits
		*/
		bool is_number(const std::string& s);

		/*
		* Add pixel(x,y) to the processing queue if it is close to backgroundColor.
		* Note that only the visited status of black pixels is updated (to avoid adding them multiple times) because we don't care about other pixels.
		* \param pixel the coordinates of the pixel to test
		* \param img the image the pixel is coming from
		* \param queue the queue, pixel will be added to it if close to backgroundColor
		* \param arrayVisited visited status of each pixel (to avoid adding a pixel to the queue multiple times)
		* \param backgroundColor the reference color
		* \param threshold_black_color the tolerance threshold
		* \sa isBlack
		*/
		void addPixelToQueue(const sibr::Vector2i & pixel, const sibr::ImageRGB & img, std::priority_queue<sibr::Vector2i> & queue, sibr::Array2d<bool> & arrayVisited, Vector3i backgroundColor, int threshold_black_color);

		/**
		 * Estimate a region that won't contain any black pixels.
		 * \param isBlack 2D array listing which pixels should be excluded
		 * \param bounds will contain the region boundaries
		 * \param thinest_bounding_box_size minimum size of the bounds along any dimension
		 */
		void findBounds(sibr::Array2d<bool> & isBlack, Bounds & bounds, int thinest_bounding_box_size);

		/** Estimate a region of an image so that no pixels of a reference color are contained in it.
		 *\param img the image to crop
		 *\param backgroundColor the reference color
		 *\param threshold_black_color the color tolerance threshold
		 *\param thinest_bounding_box_size minimum size of the bounds along any dimension
		 *\param toleranceFactor Additional tolerance factor: if set to 0 the bounds will be tight, if set to 1 it will cover the full image.
		 *\return the estimated region boundaries
		 */
		Bounds getBounds(const sibr::ImageRGB & img, Vector3i backgroundColor, int threshold_black_color, int thinest_bounding_box_size, float toleranceFactor);

		/**
		 * Estimate the average resolution of a set of images quickly using multithread to speed up the required loading.
		 * \param imagePaths list of paths to the images
		 * \param resolutions will contain each image resolution
		 * \param batch_size number of images loaded per thread internally
		 * \return the average resolution
		 */
		sibr::Vector2i calculateAvgResolution(const std::vector< Path > & imagePaths, std::vector<sibr::Vector2i> & resolutions, const int batch_size = 150);
		
		/**
		 * Find a common crop region for a set of images so that all pixels of a reference color are excluded from all images, while minimizing information loss.
		 * \param root the dataset root path (for writing list files)
		 * \param imagePaths list of image paths
		 * \param resolutions will contain the image resolutions
		 * \param avgWidth average image width, if 0 will be recomputed (slow for large datasets)
		 * \param avgHeight average image height, if 0 will be recomputed (slow for large datasets)
		 * \param batch_size batch size for multithreaded image loading
		 * \param resolutionThreshold ratio of the minimum allowed dimensions over the average image dimensions
		 * \param threshold_ratio_bounding_box_size maximum change in aspect ratio
		 * \param backgroundColor the reference background color
		 * \param threshold_black_color the color tolerance threshold
		 * \param thinest_bounding_box_size minimum size of the bounds along any dimension
		 * \param toleranceFactor Additional tolerance factor: if set to 0 the bounds will be tight, if set to 1 it will cover the full image.
		 * \return 
		 */
		sibr::Vector2i findBiggestImageCenteredBox(const Path & root, const std::vector< Path > & imagePaths, std::vector<sibr::Vector2i> & resolutions, int avgWidth = 0, int avgHeight = 0,
			const int batch_size = 150,
			float resolutionThreshold = 0.15f,
			float threshold_ratio_bounding_box_size = 0.2f,
			Vector3i backgroundColor = Vector3i(0, 0, 0),
			int threshold_black_color = 10,
			int thinest_bounding_box_size = 5,
			float toleranceFactor = 0.0f);

		/**
		 * Find the resolution of the smallest image in a set.
		 * \note In the past, this function was also supposed to exclude images based on a certain criterion and write them to a file.
		 * \param root dataset root path (for writing an exclude list file, see note)
		 * \param imagePaths list of paths to the images
		 * \return the minimum resolution
		 */
		sibr::Vector2i findMinImageSize(const Path & root, const std::vector< Path > & imagePaths);


	};
}

