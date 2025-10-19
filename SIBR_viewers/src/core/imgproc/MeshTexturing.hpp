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
#include <core/graphics/Mesh.hpp>
#include <core/assets/InputCamera.hpp>
#include "core/raycaster/Raycaster.hpp"


namespace sibr {

	/** \brief Reproject images onto a mesh using the associated camera poses, 
	 * and accumulate colors in UV-space to generate a texture map.
	 * \ingroup sibr_imgproc
	 */
	class SIBR_IMGPROC_EXPORT MeshTexturing
	{
	public:

		/** \brief Export options
		 */
		enum Options : uint {
			NONE = 0,
			FLIP_VERTICAL = 1, ///< Flip the final result.
			FLOOD_FILL = 2, ///< Perform flood filling.
			POISSON_FILL = 4 ///< Perform poisson filling (slow).
		};

		/** Constructor.
		* \param sideSize dimension of the texture
		*/
		MeshTexturing(unsigned int sideSize);

		/** Set the current mesh to texture.
		 * \param mesh the mesh to use.
		 * \warning The mesh MUST have texcoords.
		 * \note If the mesh has no normals, they will be computed.
		 */
		void setMesh(const sibr::Mesh::Ptr mesh);

		/** Reproject a set of images into the texture map, using the associated cameras.
		* \param cameras the cameras poses
		* \param images the images to reproject
		*/
		void reproject(const std::vector<InputCamera::Ptr> & cameras, const std::vector<sibr::ImageRGB::Ptr> & images, const float sampleRatio = 1.0);

		/** Get the final result. 
		* \param options the options to apply to the generated texture map.
		*/
		sibr::ImageRGB::Ptr getTexture(uint options = NONE) const;

		/** Performs flood fill of an image, following a mask.
		* \param image the image to fill
		* \param mask mask where the zeros regions will be filled
		* \return the filled image.
		*/
		template<typename T_Type, unsigned int T_NumComp>
		static typename Image< T_Type, T_NumComp>::Ptr floodFill(const Image<T_Type, T_NumComp> & image, const sibr::ImageL8 & mask) {

			typename Image< T_Type, T_NumComp>::Ptr filled(new Image< T_Type, T_NumComp>(image.w(), image.h()));

			SIBR_LOG << "[Texturing] Flood filling..." << std::endl;
			// Perform filling.
			// We need the empty pixels marked as non zeros, and the filled marked as zeros.
			cv::Mat1b flipMask = mask.toOpenCV().clone();
			flipMask = 255 - flipMask;
			cv::Mat1f dummyDist(flipMask.rows, flipMask.cols, 0.0f);
			cv::Mat1i labels(flipMask.rows, flipMask.cols, 0);

			// Run distance transform to obtain the IDs.
			cv::distanceTransform(flipMask, dummyDist, labels, cv::DIST_L2, cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);

			// Build a pixel ID to source pixel table, using the pixels in the mask.
			const sibr::Vector2i basePos(-1, -1);
			std::vector<sibr::Vector2i> colorTable(flipMask.rows*flipMask.cols, basePos);
#pragma omp parallel
			for (int py = 0; py < flipMask.rows; ++py) {
				for (int px = 0; px < flipMask.cols; ++px) {
					if (flipMask(py, px) != 0) {
						continue;
					}
					const int label = labels(py, px);
					colorTable[label] = { px,py };
				}
			}

			// Now we can turn the label image into a color image again.
#pragma omp parallel
			for (int py = 0; py < flipMask.rows; ++py) {
				for (int px = 0; px < flipMask.cols; ++px) {
					// Don't touch existing pixels.
					if (flipMask(py, px) == 0) {
						filled(px, py) = image(px, py);
						continue;
					}
					const int label = labels(py, px);
					filled(px, py) = image(colorTable[label]);
				}
			}
			return filled;
		}

		/** Performs poisson fill of an image, following a mask.
		* \param image the image to fill
		* \param mask mask where the zeros regions will be filled
		* \return the filled image.
		* \warning This is slow for large images (>8k).
		*/
		static sibr::ImageRGB32F::Ptr poissonFill(const sibr::ImageRGB32F & image, const sibr::ImageL8 & mask);

	private:

		/** Test if the UV-space mesh covers a pixel of the texture map.
		* \param px pixel x coordinate
		* \param py pixel y coordinate
		* \param finalHit the hit information if there is coverage
		* \return true if there is coverage.
		*/
		bool hitTest(int px, int py, RayHit & finalHit);

		/** Test if the UV-space mesh approximately covers a pixel of the texture map, by sampling a neighborhood in uv-space.
		* \param px pixel x coordinate
		* \param py pixel y coordinate
		* \param hit the hit information if there is coverage
		* \return true if there is coverage.
		*/
		bool sampleNeighborhood(int px, int py, RayHit& hit);

		/** Compute the interpolated position and normal at the intersection point on the initial mesh.
		* \param hit the intersection information
		* \param vertex will contain the interpolated position
		* \param normal will contain the interpolated normal
		*/
		void interpolate(const sibr::RayHit & hit, sibr::Vector3f & vertex, sibr::Vector3f & normal) const;

		sibr::ImageRGB32F _accum; ///< Color accumulator.
		sibr::ImageL8 _mask; ///< Mask indicating which regions of the texture map have been covered.

		sibr::Mesh::Ptr _mesh; ///< The original world-space mesh.
		sibr::Raycaster _worldRaycaster; ///< The world-space mesh raycaster.
		sibr::Raycaster _uvsRaycaster; ///< The uv-space mesh raycaster.


	};

}
