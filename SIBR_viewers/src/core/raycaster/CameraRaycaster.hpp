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

# include <array>
# include <core/graphics/Image.hpp>
# include <core/assets/InputCamera.hpp>
# include "core/raycaster/Config.hpp"
# include "core/raycaster/Raycaster.hpp"

namespace sibr
{

	/** Used to process casted rays from image pixels. Implement
	 this interface and write your custom behavior.
	 (e.g. see CameraRaycasterProcessor.hpp for built-in processor)
	 \ingroup sibr_raycaster
	*/
	class SIBR_RAYCASTER_EXPORT ICameraRaycasterProcessor
	{
	public:

		/// Destructor.
		virtual ~ICameraRaycasterProcessor( void ) {}

		/** Called for each casted ray (that hit or not).
		\param px pixel source pixel x coordinate
		\param py pixel source pixel y coordinate
		\param hit the (potential) hit information
		*/
		virtual void	onCast( uint px, uint py, const RayHit& hit ) = 0;

	};

	/**  Used for casting each pixel of an image into a raycaster scene.
	 \ingroup sibr_raycaster
	*/
	class SIBR_RAYCASTER_EXPORT CameraRaycaster
	{
	public:

		/// Constructor.
		CameraRaycaster( void ) { }

		/// Initialize (will be done when adding a mesh).
		bool	init( void );

		/** Add a mesh to the raycaster
		 \param mesh the mesh
		 */
		void	addMesh( const sibr::Mesh& mesh );

		/** For each image pixel, send a ray and compute data using the provided processors.
		\param cam the source camera
		\param processors a list of processors to call for each cast ray
		\param nbProcessors the number of processors in the list
		\param optLogMessage log message
		*/
		void	castForEachPixel( const sibr::InputCamera& cam, ICameraRaycasterProcessor* processors[], uint nbProcessors,
									const std::string& optLogMessage="Executing camera raycasting");

		/** This function returns the step (in both x- and y-coordinates) between each pixel in the world
		 space. Thus, if go through each pixel of an can image but you need their 3d world position,
		 using this function you can get it using:
				pixel3d = dx*pixel2d.x + dy*pixel2d.y + upLeftOffset
		 where   dx is the step between each horizontal pixel,
		         dy is the step between each vertical pixel,
		\param cam the source camera
		\param dx will contain the horizontal step
		\param dy will contain the vertical step
		\param upLeftOffset will contain the 3D coordinates of the top-left pixel
		*/
		static void	computePixelDerivatives( const sibr::InputCamera& cam, sibr::Vector3f& dx, sibr::Vector3f& dy, sibr::Vector3f& upLeftOffset );

		/**	Compute the ray direction from the camera position to a given pixel.
		\param cam the source camera
		\param pixel the pixel in [0,w-1]x[0,h-1]
		\return the ray direction from the camera position to the center of the input pixel.
		*/
		static sibr::Vector3f computeRayDir( const sibr::InputCamera& cam, const sibr::Vector2f & pixel );

		/** Estimate the clipping planes for a set of cameras so that the mesh is entirely visible in each camera.
		\param mesh the mesh to visualize
		\param cams the list of cameras
		\param nearsFars will contain the near and far plane of each camera
		*/
		static void computeClippingPlanes(const sibr::Mesh & mesh, std::vector<InputCamera::Ptr>& cams, std::vector<sibr::Vector2f> & nearsFars);

		/// \return the internal raycaster
		Raycaster&			raycaster( void )			{ return _raycaster; }
		/// \return the internal raycaster
		const Raycaster&	raycaster( void ) const 	{ return _raycaster; }

	private:

		Raycaster									_raycaster; ///< Internal raycaster.
	};

	/** A raycasting camera is an input camera augmented with additional casting and frustum helpers.
	\ingroup sibr_raycaster
	*/
	class SIBR_RAYCASTER_EXPORT RaycastingCamera : public sibr::InputCamera {
		SIBR_CLASS_PTR(RaycastingCamera);
	public:
		using HPlane = Eigen::Hyperplane<float, 3>;
		using Line3 = Eigen::ParametrizedLine<float, 3>;

		/** Constructor from an InputCamera
		\param cam the camera
		*/
		RaycastingCamera(const sibr::InputCamera & cam);

		/**	Compute the unormalized ray direction from the camera position to a given pixel.
		\param pixel the pixel in [0,w-1]x[0,h-1]
		\return the ray direction from the camera position to the center of the input pixel.
		*/
		sibr::Vector3f rayDirNotNormalized(const sibr::Vector2f & pixel) const;

		/**	Compute the normalized ray direction from the camera position to a given pixel.
		\param pixel the pixel in [0,w-1]x[0,h-1]
		\return the ray direction from the camera position to the center of the input pixel.
		*/
		sibr::Vector3f rayDir(const sibr::Vector2f & pixel) const;

		/**	Generate the ray going from the camera position to a given pixel.
		\param pixel the pixel in [0,w-1]x[0,h-1]
		\return the ray from the camera position to the center of the input pixel.
		*/
		Ray getRay(const sibr::Vector2f & pixel) const;

		/** Compute the (up to) two intersections of a oriented line with the camera frustum.
		\param line the parametrized oriented line to test
		\return the intersection parameters of the two intersection points with the frustum.
		*/
		sibr::Vector2f rayProjection(const Line3 & line) const;

		/** Check if a point is in the camera frustum.
		\param pt the 3D point to test
		\param eps the tolerance threshold
		\return true if the point is inside
		*/
		bool isInsideFrustum(const sibr::Vector3f & pt, float eps = 0.0001) const;

		/** Project a 3D point on the image plane, including points behind the camera (horizontal flip).
		\param pt3d the 3d point
		\return the pixel coordinates in [0,w]x(0,h]
		*/
		sibr::Vector2f projectImg_outside_frustum_correction(const Vector3f& pt3d) const;

		sibr::Vector3f dx, dy, upLeftOffsetMinusPos; ///< Camera raycasting parameters.

		std::vector<HPlane> frustum_planes; ///< Frustum planes: near, far, top, bottom, left, right	
	};

	///// DEFINITIONS /////

} // namespace sibr
