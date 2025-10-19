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

# include "core/graphics/Config.hpp"
# include "core/system/Transform3.hpp"


namespace sibr
{
	/** Represent a basic camera.
	\note In practice, InputCamera is used most of the time
	* \ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT Camera
	{
	public:
		SIBR_CLASS_PTR(Camera);
		typedef Transform3<float>		Transform3f;

	public:

		/// Default constructor.
		Camera( void ):
			_matViewProj(Matrix4f::Identity()), _invMatViewProj(Matrix4f::Identity()),
			_dirtyViewProj(true), _savePath(""), _debugVideoFrames(false),
			_fov(70.f/180.0f*float(M_PI)), _aspect(1.f), _znear(0.01f), _zfar(1000.f), _right(1.0f), _top(1.0f), _isOrtho(false), _p(0.5f, 0.5f) { }

		/** Set the camera pose.
		\param translation the camera translation
		\param rotation the camera rotation
		*/
		void					set( const Vector3f& translation, const Quaternionf& rotation );

		/** Set the camera pose based on two points and a up vector.
		\param eye the camera position
		\param center the camera target point
		\param up the camera up vector
		*/
		void					setLookAt( const Vector3f& eye, const Vector3f& center, const Vector3f& up );

		/** Translate the camera.
		\param v the translation
		*/
		void					translate( const Vector3f& v );

		/** Translate the camera with respect to a reference frame.
		\param v the translation
		\param ref the reference frame
		*/
		void					translate( const Vector3f& v, const Transform3f& ref );

		/** Set the camera position.
		\param v the new position
		*/
		void					position( const Vector3f& v );

		/** \return the camer position. */
		const Vector3f&			position( void ) const;

		/** Rotate the camera.
		\param rotation the quaternion rotation to apply
		*/
		void					rotate( const Quaternionf& rotation );

		/** Rotate the camera.
		\param v the euler angles to apply
		*/
		void					rotate( const Vector3f& v );
		
		/** Rotate the camera with respect to a reference frame.
		\param v the rotation euler angles
		\param ref the reference frame
		*/
		void					rotate( const Vector3f& v, const Transform3f& ref );

		/** Set the camera rotation.
		\param v the new rotation euler angles
		*/
		void					rotation( const Vector3f& v );

		/** Set the camera rotation.
		\param q the new rotation
		*/
		void					rotation( const Quaternionf& q );
		
		/** \return the camera rotation. */
		const Quaternionf&		rotation( void ) const;

		/** Set the camera transform.
		\param t the new transform
		*/
		void 					transform( const Transform3f& t );
		
		/** \return the camera transform. */
		const Transform3f&		transform( void ) const;

		/////////////////////////////////////////////////////////////////
		///// ==================== Projection  ==================== /////
		/////////////////////////////////////////////////////////////////

		/** Set the vertical field of view (in radians).
		\param value the new value
		*/
		void				fovy( float value );

		/** \return the vertical field of view (in radians). */
		float				fovy( void ) const;

		/** Set the aspect ratio.
		\param value the new value
		*/
		void				aspect( float value );

		/** \return the aspect ratio. */
		float				aspect( void ) const;

		/** Set the near plane.
		\param value the new value
		*/
		void				znear( float value );

		/** \return the near plane distance */
		float				znear( void ) const;

		/** Set the far plane.
		\param value the new value
		*/
		void				zfar( float value );

		/** \return the far plane distance */
		float				zfar( void ) const;

		/** Set the right frustum extent.
		\param value the new value
		*/
		void				orthoRight( float value );

		/** \return the right frustum distance */
		float				orthoRight( void ) const;

		/** Set the top frustum extent.
		\param value the new value
		*/
		void				orthoTop( float value );

		/** \return the top frustum distance */
		float				orthoTop( void ) const;

		/** \return true if the camera is orthographic. */
		bool				ortho(void) const;

		/** \return the camera direction vector. */
		Vector3f			dir( void ) const;

		/** \return the camera up vector. */
		Vector3f			up( void ) const;

		/** \return the camera right vector. */
		Vector3f			right( void ) const;

		/** Project 3D point using perspective projection.
		* \param point3d 3D point
		* \return pixel coordinates in [-1,1] and depth in [-1,1]
		*/
		Vector3f			project( const Vector3f& point3d ) const;

		/** Back-project pixel coordinates and depth.
		* \param pixel2d pixel coordinates p[0],p[1] in [-1,1] and depth p[2] in [-1,1]
		* \return 3D point
		*/
		Vector3f			unproject( const Vector3f& pixel2d ) const;

		/** Update the projection parameters of the camera.
		\param fovRad the vertical field ov view in radians
		\param ratio the aspect ratio
		\param znear the near plane distance
		\param zfar the far plane distance
		*/
		void				perspective( float fovRad, float ratio, float znear, float zfar );

		/** Check if a point falls inside the camera frustum.
		\param position3d the point location in 3D
		\return true if the point falls inside
		*/
		bool				frustumTest(const Vector3f& position3d) const;
		
		/** Check if a point falls inside the camera frustum. Use this version if you already have the projected point.
		\param position3d the point location in 3D
		\param pixel2d the projection location of the point in image space ([-1,1])
		\return true if the point falls inside
		*/
		bool				frustumTest(const Vector3f& position3d, const Vector2f& pixel2d) const;

		/** \return the camera model matrix (for camera stub rendering for instance). */
		Matrix4f			model( void ) const { return _transform.matrix(); }

		/** \return the camera view matrix. */
		Matrix4f			view( void ) const { return _transform.invMatrix(); }

		/** \return the camera projection matrix. */
		virtual Matrix4f	proj( void ) const;

		/** \return the camera view-proj matrix (cached). */
		const Matrix4f&		viewproj( void ) const;

		/** \return the camera inverse view-proj matrix (cached). */
		const Matrix4f&		invViewproj( void ) const;

		/** Set the camera principal point. 
		\param p the principal point, expressed in [0,1] 
		*/
		void principalPoint(const sibr::Vector2f & p);

		/** Interpolate between two cameras.
		\param from start camera
		\param to end camera
		\param dist01 the interpolation factor
		\return a camera with interpolated parameters
		*/
		static Camera		interpolate( const Camera& from, const Camera& to, float dist01 );

		/** Set stereo camera projection parameters.
		\param isLeft is the camera for the left eye (else right)
		\param focal the focal distance
		\param iod the inter ocular distance
		*/
		void 				setStereoCam(bool isLeft, float focal, float iod);

		/** Set orthographic camera projection parameters.
		\param right the right frustum extent
		\param top the top frustum extent
		*/
		void				setOrthoCam(float right, float top);

		/** \return true if the rendering generated with the camera be saved. */
		bool				needSave() const { return _savePath!=""; }

		/**\return true if the rendering generated with the camera be saved as a frame. */
		bool				needVideoSave() const { return _debugVideoFrames; }

		/** \return the save destination path for renderings */
		std::string			savePath() const { return _savePath; }

		/** Set the save destination path for renderings.
		\param savePath the new path
		*/
		void				setSavePath(std::string savePath) { _savePath = savePath; }

		/** Toggle video saving.
		\todo Cleanup naming.
		\param debug if true, saving frames
		*/
		void				setDebugVideo(const bool debug) { _debugVideoFrames = debug; }
		
	protected:

		/** Trigger a viewproj matrix udpate. */
		void				forceUpdateViewProj( void ) const;

		std::string				_savePath; ///< Save destination path when reocrding images.
		bool					_debugVideoFrames; ///< Is video saving enabled or not. \todo Cleanup.
		mutable Matrix4f		_matViewProj; ///< View projection matrix.
		mutable Matrix4f		_invMatViewProj; ///< Inverse projection matrix.
		mutable bool			_dirtyViewProj; ///< Does the camera matrix need an update.

		Transform3f		_transform; ///< The camera pose.
		float			_fov; ///< The vertical field of view (radians)
		float			_aspect; ///< Aspect ratio.
		float			_znear; ///< Near plane.
		float			_zfar; ///< Far plane.
		float			_right; ///< Frustum half width.
		float			_top; ///< Frustum half height.
		sibr::Vector2f   _p = {0.5f, 0.5}; ///< Principal point.
		bool			_isOrtho; ///< Is the camera orthographic.
	};

	/** Write a camera to a byte stream.
	\param stream the stream to write to
	\param c the camera
	\return the stream (for chaining).
	*/
	SIBR_GRAPHICS_EXPORT ByteStream&		operator << (ByteStream& stream, const Camera& c );

	/** Read a camera from a byte stream.
	\param stream the stream to read from
	\param c the camera
	\return the stream (for chaining).
	*/
	SIBR_GRAPHICS_EXPORT ByteStream&		operator >> (ByteStream& stream, Camera& c );

	///// DEFINITIONS /////

	/////////////////////////////////////////////////////////////////
	inline const Transform3f&		Camera::transform( void ) const {
		return _transform;
	}

	inline void				Camera::set( const Vector3f& translation, const Quaternionf& rotation ) {
		_dirtyViewProj = true; _transform.set(translation, rotation);
	}

	inline void				Camera::setLookAt( const Vector3f& eye, const Vector3f& at, const Vector3f& up ) {
		const Vector3f zAxis( (eye - at).normalized() );
		const Vector3f xAxis( (up.normalized().cross(zAxis)).normalized() );
		const Vector3f yAxis( zAxis.cross(xAxis).normalized() );

		Eigen::Matrix3f rotation;
		rotation << xAxis, yAxis, zAxis;
		Quaternionf q(rotation);

		_transform.set(eye,q);
		forceUpdateViewProj();
	}

	inline void				Camera::translate( const Vector3f& v ) {
		_dirtyViewProj = true; _transform.translate(v);
	}
	inline void				Camera::translate( const Vector3f& v, const Transform3f& ref ) {
		_dirtyViewProj = true; _transform.translate(v, ref);
	}
	inline void				Camera::position( const Vector3f& v ) {
		_dirtyViewProj = true; _transform.position(v);
	}
	inline const Vector3f&		Camera::position( void ) const {
		return _transform.position();
	}

	inline void					Camera::rotate( const Quaternionf& rotation ) {
		_dirtyViewProj = true; _transform.rotate(rotation);
	}
	inline void					Camera::rotate( const Vector3f& v ) {
		_dirtyViewProj = true; _transform.rotate(v);
	}
	inline void					Camera::rotate( const Vector3f& v, const Transform3f& ref ) {
		_dirtyViewProj = true; _transform.rotate(v, ref);
	}

	inline void					Camera::rotation( const Vector3f& v ) {
		_dirtyViewProj = true; _transform.rotation(v);
	}
	inline void					Camera::rotation( const Quaternionf& q ) {
		_dirtyViewProj = true; _transform.rotation(q);
	}

	inline const Quaternionf&		Camera::rotation( void ) const {
		return _transform.rotation();
	}

	/////////////////////////////////////////////////////////////////

	inline void	Camera::fovy( float value ) {
		_fov = value; _dirtyViewProj = true;
	}
	inline float	Camera::fovy( void ) const {
		return _fov;
	}

	inline void	Camera::aspect( float value ) {
		_aspect = value; _dirtyViewProj = true;
	}
	inline float	Camera::aspect( void ) const {
		return _aspect;
	}

	inline void	Camera::znear( float value ) {
		_znear = value; _dirtyViewProj = true;
	}
	inline float	Camera::znear( void ) const {
		return _znear;
	}

	inline void	Camera::zfar( float value ) {
		_zfar = value; _dirtyViewProj = true;
	}
	inline float	Camera::zfar( void ) const {
		return _zfar;
	}

	inline void Camera::principalPoint(const sibr::Vector2f & p) {
		_p = p; _dirtyViewProj = true;
	}

	inline void	Camera::orthoRight( float value ) {
		_right = value; _dirtyViewProj = true;
	}
	inline float	Camera::orthoRight( void ) const {
		return _right;
	}

	inline void	Camera::orthoTop( float value ) {
		_top = value; _dirtyViewProj = true;
	}
	inline float	Camera::orthoTop( void ) const {
		return _top;
	}
	inline bool	Camera::ortho(void) const {
		return _isOrtho;
	}


	inline const Matrix4f&			Camera::viewproj( void ) const {
		if (_dirtyViewProj)
			forceUpdateViewProj();

		return _matViewProj;
	}

	inline const Matrix4f&			Camera::invViewproj( void ) const {
		if (_dirtyViewProj)
			forceUpdateViewProj();

		return _invMatViewProj;
	}

} // namespace sibr
