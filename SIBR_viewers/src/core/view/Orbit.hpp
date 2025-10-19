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

#include <memory>
#include <fstream>

#include "Config.hpp"
#include "core/graphics/Shader.hpp"
#include "core/assets/InputCamera.hpp"
#include "ICameraHandler.hpp"


namespace sibr {

	class Viewport;
	class Mesh;
	class Input;
	class Raycaster;

	/**
	 * Interactive camera that allow the user to roate around an object using the keypad.
	 * Commands:

		to enable/disable the orbit (note that using at least once ( atl + click ) to retrieve a 3D point on the proxy is mandatory before enabling the orbit) :
		b
		in static mode (default mode) :
		5 to flip the orbit (might be the first thing to do if all commands seem broken/reversed, it is needed because there is an ambiguity when using the normal of the plan containing the input cameras and the clicked point)
		4 or 6 to rotate towards current camera x-axis
		2 or 8 to rotate towards current camera y-axis
		7 or 9 to rotate towards current camera z-axis
		1 or 3 to zoom in or out
		in dynamic mode ( rotates without interruption around an axis )  :
		alt + ( 4 or 6 ) to rotate towards current camera x-axis
		alt + ( 2 or 8 ) to rotate towards current camera y-axis
		alt + ( 7 or 9 ) to rotate towards current camera z-axis
		5 to inverse the direction (same axis)
		0 to switch back to static mode with initial camera
		. to switch back to static mode with current camera
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT Orbit : public ICameraHandler
	{
	
	public:

		/**
		 Create an orbit centered at (0,0,0) with basic initial parameters.
		 It is recommended to call fromCamera after that to setup the orbit with a valid set of parameters.
		*/
		Orbit( void );

		/**
		 Setup the orbit so that its camera has the same pose as the argument camera. A raycaster is required to find the center of the orbit.
		\param cam the reference camera
		\param raycaster raycaster to use for centering intersection tests.
		*/
		void fromCamera(const sibr::InputCamera & cam, const std::shared_ptr<sibr::Raycaster> raycaster);

		/**
			Update the orbit camera based on the user input (keyboard). Can require a raycaster if the user is alt-clicking to select a new orbit center.
		\param input user input
		\param raycaster optional raycaster
		*/
		void update( const sibr::Input & input, const std::shared_ptr<sibr::Raycaster> raycaster = std::shared_ptr<sibr::Raycaster>());

		/** Update the camera handler state.
		\param input user input
		\param deltaTime time elapsed since last udpate
		\param viewport view viewport
		*/
		virtual void update(const sibr::Input & input, const float deltaTime, const Viewport & viewport) override;
		
		/** \return the current camera. */
		virtual const sibr::InputCamera & getCamera( void ) const override;

	private:

		/** Internal orbit parameters. */
		struct OrbitParameters
		{
			/** Motion direction: Clockwise, AntiClockWis e*/
			enum OrbitDirection { CW = 1, ACW = -1 };
			/** Orbit current motion status. */
			enum OrbitStatus { STATIC, FORWARD_X, FORWARD_Y, FORWARD_Z };

			/** Default constructor. */
			OrbitParameters(void) : factor(0), status(STATIC),
				center(sibr::Vector3f(0.0f, 0.0f, 0.0f)), radius(1.0f), theta(0), phi(0), roll(0), direction(CW), keepCamera(false)
			{}

			/** Flip motion. */
			void flip(void) {
				yAxis = -yAxis;
				xAxis = yAxis.cross(zAxis);
			}

			bool								keepCamera; ///< ?
			int									factor; ///< Interpolation ID.

			OrbitStatus							status; ///< Current status.
			OrbitDirection						direction; ///< Current motion direction.

			sibr::Vector3f						center; ///< Orbit center.
			sibr::Vector3f						xAxis; ///< Orbit X axis.
			sibr::Vector3f						yAxis; ///< Orbit Y axis.
			sibr::Vector3f						zAxis; ///< Orbit Z axis.

			float								radius; ///< Orbit radius.
			float								theta, phi, roll; ///< Orbit angles.

			sibr::Camera						initialCamera; ///< Starting camera.
			sibr::Vector4f						planePointCams; ///< Fitted plane points.

		};

		/**
		*	Compute new camera pose from current orbit parameters.
		*/
		void interpolateOrbit();

		/**
		*	Updates the orbit's center and camera pose, by casting a ray from the clicked point (in Input) to the mesh.
		*	\param input user input
		*	\param raycaster scene raycaster
		*/
		void updateOrbitParameters(const sibr::Input& input, const std::shared_ptr<sibr::Raycaster> raycaster);

		/**
		*	Updates the orbit's center and camera pose, by casting a ray from the center of the screen to the mesh.
		*	\param raycaster scene raycaster
		*/
		void updateOrbitParametersCentered(const std::shared_ptr<sibr::Raycaster> raycaster);

		/** 
		*	Compute the best fitting plane of the clicked points plus the input cams positions.
		*	\param clickedPoint point clicked by the user
		*	\param cams reference cameras
		*/
		static sibr::Vector4f computeFittingPlaneCameras(sibr::Vector3f& clickedPoint, const std::vector<InputCamera::Ptr>& cams);

		bool _hasBeenInitialized; ///< Has the orbit been initialized.
		bool _orbitPointClicked; ///< Has the user clicked on a point in the scene.
		sibr::InputCamera _currentCamera; ///< Current camera.
		OrbitParameters _orbit; ///< Parameters.
	
	};
}
