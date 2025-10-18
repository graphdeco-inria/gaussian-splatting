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

#include "core/view/IBRBasicUtils.hpp"
#include "core/view/FPSCamera.hpp"
#include "core/view/Orbit.hpp"
#include "core/view/TrackBall.h"
#include "core/assets/CameraRecorder.hpp"
#include "core/graphics/Viewport.hpp"
#include "core/graphics/Mesh.hpp"
#include "ICameraHandler.hpp"

namespace sibr {
	class Mesh;
	class Input;
	class Raycaster;

	/**
		The InteractiveCameraHandler gathers various types of camera interactions and
		allows the user to switch between them, keeping them in sync.
		It can also perform camera interpolation along a path.
		\ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT InteractiveCameraHandler : public ICameraHandler
	{

	public:

		SIBR_CLASS_PTR(InteractiveCameraHandler);

		/** Current handler interaction mode. */
		enum InteractionMode {
			FPS = 0, ORBIT = 1, INTERPOLATION = 2, TRACKBALL = 3, NONE=4
		};

		/** Constructor.
		 *\param supportRecording can this handler record camera paths.
		 *\todo Do we really need this option?
		 */
		InteractiveCameraHandler(const bool supportRecording = true);

		/** \deprecated Resolution is deprecated and will be removed in the near future.
		 *	See setup(const std::vector<InputCamera::Ptr>&, const sibr::Viewport&, std::shared_ptr<sibr::Raycaster>,...) instead. */
		void setup(const std::vector<InputCamera::Ptr> & cams, const sibr::Vector2u & resolution, const sibr::Viewport & viewport, const std::shared_ptr<sibr::Raycaster> raycaster);

		/** Setup an interactive camera handler from an existing camera.
		The interactive camera will be initialized at the position of the argument camera.
		\param cam initialization camera
		\param viewport the window viewport
		\param raycaster raycaster containing the mesh displayed (used for the trackball centering), can be nullptr
		*/
		void setup(const sibr::InputCamera & cam, const sibr::Viewport & viewport, const std::shared_ptr<sibr::Raycaster> raycaster);

		/** Setup an interactive camera handler from an area of interest.
		The interactive camera will be initialized so that the area is completely visible.
		\param areaOfInterest the region of space to show
		\param viewport the window viewport
		\param raycaster raycaster containing the mesh displayed (used for the trackball centering), can be nullptr
		*/
		void setup(const Eigen::AlignedBox<float, 3>& areaOfInterest, const sibr::Viewport & viewport, const std::shared_ptr<sibr::Raycaster> raycaster);

		/** Setup an interactive camera handler from a series of existing cameras and mesh. 
		The interactive camera will be initialized at the position of the first camera from the list.
		\param cams a list of cameras (used for interpolation path)
		\param viewport the window viewport
		\param raycaster raycaster containing the mesh displayed (used for the trackball centering), can be nullptr
		\param clippingPlanes optional clipping planes to enforce
		*/
		void setup(const std::vector<InputCamera::Ptr>& cams, const sibr::Viewport& viewport, std::shared_ptr<sibr::Raycaster> raycaster, const sibr::Vector2f & clippingPlanes = {-1.0f,-1.0f});

		/** Setup an interactive camera handler from a mesh.
		The interactive camera will be initialized so that the mesh is completely visible.
		\param mesh the mesh to display
		\param viewport the window viewport
		\note a raycaster will be set up internally
		*/
		void setup(std::shared_ptr<sibr::Mesh> mesh, const sibr::Viewport& viewport);

		/** Setup a camera path for the interpolation mode. 
		 * \param cameras to interpolate along
		 */
		void setupInterpolationPath(const std::vector<InputCamera::Ptr> & cameras);

		/** Move the interactive camera to a new position and change its internal parameters.
		\param cam the cameras the parameters and pose should be copied from
		\param interpolate smooth interpolation between the current pose and the new one
		\param updateResolution should the resolution of the camera be updated or not. Can be disabled if the new cam has a size incompatible with the current viewport.
		*/
		void fromCamera(const sibr::InputCamera & cam, bool interpolate = true, bool updateResolution = true);
		
		/** Move the interactive camera to a new position.
		\param transform the transform the orientation and pose should be copied from
		\param interpolate smooth interpolation between the current pose and the new one
		\param updateResolution should the resolution of the camera be updated or not. Can be disabled if the new cam has a size incompatible with the current viewport.
		*/
		void fromTransform(const Transform3f & transform, bool interpolate = true, bool updateResolution = true);

		/** Set the clipping planes.
		 *\param znear near plane
		 *\param zfar far plane
		 */
		void setClippingPlanes(float znear, float zfar);

		/** Find the camera in a list closest to the current interactive camera position
		\param inputCameras the list to search in
		\return the index of the closest camera in the list, or -1
		\note This function ignores cameras that are not 'active' in the list.
		*/
		int	findNearestCamera(const std::vector<InputCamera::Ptr>& inputCameras, const bool& useRotation = true) const;

		/** Toggle camera motion smoothing. */
		void switchSmoothing() { _shouldSmooth = !_shouldSmooth; SIBR_LOG << "Smoothing " << (_shouldSmooth ? "enabled" : "disabled") << std::endl; }

		/** Toggle automatic snapping when getting close to a camera from the interpolation path. */
		void switchSnapping() { _shouldSnap = !_shouldSnap; SIBR_LOG << "Snapping " << (_shouldSnap ? "enabled" : "disabled") << std::endl; }

		/** Switch the interaction mode (trackball, fps,...). 
			\param mode the new mode
		*/
		void switchMode(const InteractionMode mode);

		/** Save the current camera as a binary file to a standard location.
		 * \param datasetPath destination directory
		 * \note "default_camera.bin" will be appended to the path.
		 */
		void saveDefaultCamera(const std::string& datasetPath);

		/** Load a camera parameters from a binary file at a standard location.
		 *\param cam the camera to use if loading fails
		 * \param datasetPath source directory
		 * \note "default_camera.bin" will be appended to the path.
		 */
		void loadDefaultCamera(const sibr::InputCamera& cam, const std::string& datasetPath);

		/** \return the current interaction mode. */
		InteractionMode getMode() const { return _currentMode; }

		/** Set the speed of the FPS camera.
		\param speed the new speed
		*/
		void setFPSCameraSpeed(const float speed);

		/// ICameraHandler interface.
		/** Update function, call at every tick.
		\param input the input object for the current view.
		\param deltaTime time elapsed since last frame
		\param viewport optional window viewport (can be used by the trackball for instance)
		*/
		virtual void update(const sibr::Input & input, float deltaTime, const sibr::Viewport & viewport = Viewport(0.0f, 0.0f, 0.0f, 0.0f)) override;

		/** \return the current camera. */
		virtual const sibr::InputCamera & getCamera(void) const override;

		/** Render additional information on screen (trackball gizmo).
		\param viewport the window viewport
		*/
		virtual void onRender(const sibr::Viewport & viewport) override;

		/** Show the GUI. 
		\param suffix additional GUI name suffix to avoid collisions when having multiple handlers. 
		*/
		virtual void onGUI(const std::string & suffix) override;

		/** \return the camera recorder */
		sibr::CameraRecorder & getCameraRecorder() { return _cameraRecorder; };
		
		/** \return the camera trackball */
		sibr::TrackBall & getTrackball() { return _trackball; }

		/** Snap the interactive camera to one of the interpolation path cameras.
		\param id the index of the camera to snap to. if -1, the closest camera. 
		*/
		void snapToCamera(int id = -1);

		/** \return the handler raycaster.
			\warning Can be nullptr 
		*/
		std::shared_ptr<Raycaster> & getRaycaster() { return _raycaster; }

		/** \return true if the handler has been entirely setup */
		bool isSetup() const { return _isSetup; }

		/** \return the handler viewport */
		const sibr::Viewport & getViewport() const { return _viewport; }

		/** \return radius used for trackball*/
		float & getRadius() { return _radius; }

		float getInterpolatedHeight(const std::vector<InputCamera::Ptr>& inputCameras);

	private:

		int _currentCamId; ///< Current snapped camera ID.

		bool _shouldSmooth; ///< Motion smoothing.
		bool _shouldSnap; ///< Currently snapping.
		bool _altitudeInterp = false; ///< Interpolate altitude on the cameras height or not.

		float _bottom_vp = 1.f;

		sibr::FPSCamera _fpsCamera; ///< FPS handler.
		sibr::Orbit _orbit; ///< Orbit handler.
		sibr::TrackBall _trackball; ///< Trackball handler.

		InteractionMode _currentMode; ///< Current handler mode.

		float _radius; ///< Trackball radius property (for GUI).

		std::shared_ptr<sibr::Raycaster> _raycaster; ///< Raycaster (for trackball).
		sibr::Viewport _viewport;  ///< Current viewport.

		sibr::InputCamera _previousCamera; ///< Previous camera (for interpolation).
		sibr::InputCamera _currentCamera; ///< Current camera.

		/// Parameters for path interpolation.
		uint _startCam; ///< Start camera index in the list.
		uint _interpFactor; ///< Current interpolation factor between cam _startCam and _startCam+1.
		std::vector<InputCamera::Ptr> _interpPath; ///< Cameras along the path.

		sibr::CameraRecorder _cameraRecorder; ///< Camera recorder.
		bool _supportRecording; ///< Does the camera support recording (uneeded).
		std::vector<InputCamera::Ptr> _keyCameras; ///< Key cameras saved punctually. \note This could be merged with the camera recorder.

		sibr::Vector2f _clippingPlanes; ///< Clipping planes parameter (for GUI).
		bool _saveFrame; ///< Should the frame be saved as an image.
		bool _saveFrameVideo; ///< Should the frame be saved as part of a video.
		bool _triggerCameraUpdate; ///< Should the camera be updated (delayed if info is missing).
		bool _isSetup; ///< Is the handler setup.
		float _cameraFovDeg = 0.0f; ///< Camera field of view in degrees (for GUI).
		bool _fribrExport = false; ///< Switch to FRIBR compatible export mode for paths.

		/** Interpolate along the path. */
		void interpolate();

	};
}
