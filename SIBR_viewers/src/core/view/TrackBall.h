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

	
	/** Provide a handler to interact using a trackball (based on mouse motions).
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT TrackBall : public ICameraHandler
	{
	public:
		/** Constructor
		\param verbose log updates and infos. 
		*/
		TrackBall( bool verbose = false );

		/** Load a trackball settings from a file on disk.
		\param filePath path to the file
		\param viewport current viewport
		\return a success boolean
		\note The viewport is needed to fill-in missing info.
		*/
		bool load( std::string & filePath ,  const Viewport & viewport);

		/** Save trackball settings to a file on disk.
		\param filePath file path
		*/
		void save( std::string & filePath ) const ;

		/** Update the trackball pose from a reference camera.
		 \param cam the reference camera
		 \param viewport the viewport to use
		 \param radius the default trackball radius to use
		**/
		void fromCamera( const InputCamera & cam , const Viewport & viewport , const float & radius = 100.0f );

		/** Setup the trackball so that a mesh if visible and centered.
		\param mesh the mesh to show
		\param viewport the view viewport
		*/
		bool fromMesh( const Mesh & mesh, const Viewport & viewport );

		/** Setup the trackball so that a region of space if visible and centered.
		\param box the region of space to cover
		\param viewport the view viewport
		*/
		bool fromBoundingBox(const Eigen::AlignedBox<float, 3>& box, const Viewport & viewport);

		/** Update the trackball handler state. the raycaster is used when the user is clicking to center the trackball or panning.
			\param input user input
			\param viewport view viewport
			\param raycaster an optional raycaster
		*/
		void update( const Input & input , const Viewport & viewport, std::shared_ptr<Raycaster> raycaster = std::shared_ptr<Raycaster>());

		/** update the internal aspect ratio.
		\param viewport the new viewport
		*/
		void updateAspectWithViewport(const Viewport & viewport);

		bool drawThis; ///< Should the trackball overlay be displayed.

		/** \return true if the trackball has been initialized */
		bool initialized() const { return hasBeenInitialized; }

		/// ICameraHandler interface
		/** Update the camera handler state.
		\param input user input
		\param deltaTime time elapsed since last udpate
		\param viewport view viewport
		*/
		virtual void update(const sibr::Input & input, const float deltaTime, const Viewport & viewport) override;

		/** \return the current camera. */
		virtual const InputCamera & getCamera(void) const override;

		/** Render on top of the associated view(s).
		\param viewport the rendering region
		*/
		virtual void onRender(const sibr::Viewport & viewport) override;

	private:

		/** Trackball interaction status. */
		enum class TrackBallState { IDLE, TRANSLATION_PLANE, TRANSLATION_Z, ROTATION_SPHERE, ROTATION_ROLL };

		/** Map a pixel to a point on the sphere.
		\param pos2D pixel position
		\param viewport view viewport
		\return a 3D point on the sphere
		*/
		Vector3f mapToSphere( const Vector2i & pos2D, const Viewport & viewport ) const;


		/** Map a pixel to a point on the plane.
		\param pos2D pixel position
		\return a 3D point on the plane
		*/
		Vector3f mapTo3Dplane( const Vector2i & pos2D ) const;

		/** update near and far planes.
		\param input user input
		*/
		void updateZnearZFar( const Input & input );

		/** update the trackball pivot center.
		\param input user input
		\param raycaster the scene raycaster
		*/
		void updateBallCenter( const Input & input,  std::shared_ptr<Raycaster> raycaster );

		/** Update the trackball radius.
		\param input user input
		*/
		void updateRadius( const Input & input );

		/** Update the rotation parameters.
		\param input user input
		\param viewport view viewport
		*/
		void updateRotationSphere( const Input & input , const Viewport & viewport );

		/** Update the rotation parameters.
		\param input user input
		\param viewport view viewport
		*/
		void updateRotationRoll( const Input & input , const Viewport & viewport );

		/** Update the translation parameters.
		\param input user input
		\param viewport view viewport
		\param raycaster optional scene raycaster
		*/
		void updateTranslationPlane( const Input & input , const Viewport & viewport, std::shared_ptr<Raycaster> raycaster = std::shared_ptr<Raycaster>() );

		/** Update the translation parameters.
		\param input user input
		\param viewport view viewport
		*/
		void updateTranslationZ( const Input & input , const Viewport & viewport );

		/** Update based on keys input
		\param input user input
		*/
		void updateFromKeyboard(const Input & input);

		/** Check if a point is in the central trackball region.
		\param pos2D position in the view
		\param viewport view viewport
		\return true if it falls inside
		*/
		bool isInTrackBall2dRegion( const Vector2i & pos2D, const Viewport & viewport ) const;

		/**
		 * Check if three points are in clockwise order.
		 * \param a first point
		 * \param b second point
		 * \param c third point
		 * \return true if their order is clockwise.
		 */
		bool areClockWise( const Vector2f & a, const Vector2f & b, const Vector2f & c ) const;

		/** Log a message.
		\param msg essage string
		*/
		void printMessage( const std::string & msg ) const;

		/** Save vector to output stream.
		\param s stream
		\param v vector
		*/
		void saveVectorInFile( std::ofstream & s , const Vector3f & v ) const ;

		/** Setup trackball shader. */
		void initTrackBallShader( void );

		/** Set camera attributes
		\param viewport the viewport to use
		*/
		void setCameraAttributes( const Viewport & viewport );

		/** Update camera size.
		\param viewport the viewport to use
		*/
		void updateTrackBallCameraSize(const Viewport & viewport);

		/** Update trackball interaction status.
		  \param input user input
		  \param viewport the viewport to use
		*/
		void updateTrackBallStatus( const Input & input, const Viewport & viewport );

		/** Main update function.
		  \param input user input
		  \param viewport the viewport to use
		  \param raycaster optional scene raycaster
		*/
		void updateTrackBallCamera( const Input & input, const Viewport & viewport ,  std::shared_ptr<Raycaster> raycaster = std::shared_ptr<Raycaster>() );


		InputCamera					fixedCamera;	///< Reference camera.
		InputCamera					tempCamera;	///< Temp camera.

		Vector3f					fixedCenter; ///< Current center.
		Vector3f					tempCenter; ///< Temp center.

		Vector2i					lastPoint2D; ///< Last clicked 2D point.
		Vector2i					currentPoint2D; ///< Current clicked point.

		Eigen::Hyperplane<float,3>	trackballPlane; ///< Trackball translation plane.

		TrackBallState				state; ///< Current status.

		bool						hasBeenInitialized; ///< Initialized or not.
		bool						verbose; ///< verbose or not.

		float						zoom=1.0f;//zoom factor used for ortho cams
		//members used for interaction drawing
		std::shared_ptr<Mesh>		quadMesh; ///< Supporting mesh for the overlay.
		GLShader					trackBallShader; ///< Overlay shader.
		GLParameter					ratioTrackBall2Dgpu; ///< Aspect ratio.
		GLParameter					trackBallStateGPU; ///< Trackball state uniform.
		bool						shadersCompiled; ///< Are the sahders ready.
		static float				ratioTrackBall2D; ///< 2D ratio parameter.

	};

}