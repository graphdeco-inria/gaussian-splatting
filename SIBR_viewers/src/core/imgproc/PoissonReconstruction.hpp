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


namespace sibr {

	/** \brief Performs gradient integration for tasks such as Poisson-based inpainting, smooth filling, ...
	 * See the constructor for additional details.
	 * \ingroup sibr_imgproc
	 */
	class SIBR_IMGPROC_EXPORT PoissonReconstruction
	{
	public:

		/** Initialize reconstructor for a given problem. Gradients and target are expected to be RGB32F, mask is L32F.
		  In the mask, pixels with value = 0 are to be inpainted, value > 0.5 are pixels to be used as source/constraint,  value < -0.5 are pixels to be left unchanged and unused.
		  To compute the gradients from an image, prefer using PoissonReconstruction::computeGradients (weird results have been observed when using cv::Sobel and similar).
		\param gradientsX the RGB32F horizontal color gradients to integrate along
		\param gradientsY the RGB32F vertical color gradients to integrate along
		\param mask the L32F mask denoting how each pixel should be treated. 
		\param img_target the RGB32 image to use as a source constraint (will be copied internally)
		**/
		PoissonReconstruction(
			const cv::Mat3f & gradientsX,
			const cv::Mat3f & gradientsY,
			const cv::Mat1f & mask,
			const cv::Mat3f & img_target
		);

		/** Solve the reconstruction problem. */
		void solve(void);

		/** \return the result of the reconstruction */
		cv::Mat result() const { return _img_target; }

		/** helper to get the pixel coordinates of valid pixels for agiven pixel and image size.
		 *\param pos the central pixel position
		 *\param width number of columns/width
		 *\param height number of rows/height
		 *\return a vector containing neighboring pixels coordinates.
		 */
		static std::vector< sibr::Vector2i > getNeighbors(sibr::Vector2i pos, int width, int height);

		/** Compute the gradients of an RGB32F matrix using forward finite differences.
		 *\param src the matrix to compute the gradients of
		 *\param gradX will contain the horizontal gradients
		 *\param gradY will contain the vertical gradients
		 */
		static void computeGradients(const cv::Mat3f & src, cv::Mat3f & gradX, cv::Mat3f & gradY);
		
	private:
		cv::Mat _img_target; ///< Main image.
		cv::Mat _gradientsX; ///< Gradients.
		cv::Mat _gradientsY; ///< Gradients.
		cv::Mat _mask; ///< Mask guide.

		std::vector<sibr::Vector2i> _pixels; ///< list of valid pixels.
		std::vector<sibr::Vector2i> _boundaryPixels; ///< List of boundary pixels.
		std::vector<int > _pixelsId; ///< Pixel IDs list.
		std::vector<std::vector<int> > _neighborMap; ///< Each pixel valid neighbors.

		/** Parse the mask and the additional label condition into a list of pixels to modified and boundaries conditions. */
		void parseMask(void);

		/** Make sure that every modified pixel is connected to some boundary condition, all non connected pixels are discarded. */
		void checkConnectivity(void);

		/** Heuristic to fill isolated black pixels. */
		void postProcessing(void);

		/** Are we in the mask (ie mask==0). 
		\param pos the pixel to test for
		\return true if mask(pix) == 0
		*/
		bool isInMask(sibr::Vector2i & pos);

		/* Are we ignored (ie mask==-1).
		\param pos the pixel to test for
		\return true if mask(pix) == -1
		*/
		bool isIgnored(sibr::Vector2i & pos);

	};

}
