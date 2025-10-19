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

# include "core/system/ByteStream.hpp"
# include "core/system/Config.hpp"
# include "core/system/Matrix.hpp"
# include "core/system/Vector.hpp"
# include "core/system/Quaternion.hpp"


namespace sibr
{
	/**
	 * Represent a 3D transformation composed of a rotation and translation.
	* \ingroup sibr_system
	*/
	template <typename T>
	class Transform3
	{
	public:
		typedef Eigen::Matrix<T,3, 1, Eigen::DontAlign>		Vector3;
		typedef Eigen::Quaternion<T>						Quaternion;

	public:

		/** Constructor: identity transform. */
		Transform3( void ) : _position(0, 0, 0), _scale(1., 1., 1.) {
			_rotation.setIdentity();
		}

		/** Set the transformation parameters.
		 *\param translation the translation vector
		 *\param rotation the rotation quaternion
		 */
		void	       set( const Vector3& translation, const Quaternion& rotation ) {
				_position = translation;
				_rotation = rotation;
		}

		/** Apply a translation.
		 *\param x x shift
		 *\param y y shift
		 *\param z z shift
		 **/
		void				translate( float x, float y, float z );

		/** Apply a translation that is itself rotated by another transformation.
		 *\param x x shift
		 *\param y y shift
		 *\param z z shift
		 *\param ref additional rotation trnasofrmation to apply to the translation vector.
		 **/
		void				translate( float x, float y, float z, const Transform3& ref);

		/** Apply a translation.
		 *\param v translation vector
		 **/
		void				translate( const Vector3& v );
		
		/** Apply a translation that is itself rotated by another transformation.
		 *\param v translation vector
		 *\param ref additional rotation trnasofrmation to apply to the translation vector.
		 **/
		void				translate( const Vector3& v, const Transform3& ref );


		void				scale(const float& s);
		/** Set the position.
		 *\param x x position
		 *\param y y position
		 *\param z z position
		 **/
		void				position( float x, float y, float z );

		/** Set the position.
		 *\param v position
		 **/
		void				position( const Vector3& v );

		/** \return the position */
		const Vector3&	position( void ) const;


		/** Apply a rotation.
		 *\param rotation quaternion rotation
		 */
		void					rotate( const Quaternion& rotation );

		/** Apply a rotation using Euler angles.
		 *\param x yaw
		 *\param y pitch
		 *\param z roll
		 *\todo Clarify the angles order.
		 *\sa quatFromEulerAngles
		 */
		void					rotate( float x, float y, float z );

		/** Apply a rotation using Euler angles and composite with an additional transformation.
		 *\param x yaw
		 *\param y pitch
		 *\param z roll
		 *\param ref additional rotation
		 *\todo Clarify the angles order.
		 *\sa quatFromEulerAngles
		 */
		void					rotate( float x, float y, float z,
											const Transform3& ref);

		/** Apply a rotation using Euler angles.
		 *\param v angles
		 *\todo Clarify the angles order.
		 *\sa quatFromEulerAngles
		 */
		void					rotate( const Vector3& v );

		/** Apply a rotation using Euler angles and composite with an additional transformation.
		 *\param v angles
		 *\param ref additional rotation
		 *\todo Clarify the angles order.
		 *\sa quatFromEulerAngles
		 */
		void					rotate( const Vector3& v, const Transform3& ref );

		/** Set the rotation from Euler angles.
		 *\param x yaw
		 *\param y pitch
		 *\param z roll
		 *\todo Clarify the angles order.
		 *\sa quatFromEulerAngles
		 */
		void					rotation( float x, float y, float z );

		/** Set the rotation from Euler angles.
		 *\param v angles
		 *\todo Clarify the angles order.
		 *\sa quatFromEulerAngles
		 */
		void					rotation( const Vector3& v );

		/** Set the rotation.
		 *\param q quaternion rotation
		 */
		void					rotation( const Quaternion& q );

		/// \return the rotation
		const Quaternion&	rotation( void ) const;

		/// \return the transformation matrix
		Matrix4f		matrix( void ) const;
		/// \return the inverse of the transformation matrix
		Matrix4f		invMatrix( void ) const;

		/** Interpolate between two transformations.
		 *\param from source transformation
		 *\param to destination transformation
		 *\param dist01 interpolation factor
		 *\return the interpolated transformation
		 */
		static Transform3<T>	interpolate( const Transform3<T>& from, const Transform3<T>& to, float dist01 ) {
			dist01 = std::max(0.f, std::min(1.f, dist01)); // clamp
			
			Transform3<T> out;
			out.position((1.0f-dist01)*from.position() + dist01*to.position());
			out.rotation(from.rotation().slerp(dist01, to.rotation()));
			return out;
		}

		/** Linearly extrapolate based on two transformations, by reapplying the delta between the two transformations to the current one 
		 * and interpolating between the current and the new estimate.
		 *\param previous source transformation
		 *\param current current transformation
		 *\param dist01 extrapolation factor
		 *\return the extrapolated transformation
		 *\note dist01 should still be in 0,1
		 */
		static Transform3<T>	extrapolate(const Transform3<T>& previous, const Transform3<T>& current, float dist01) {

			Vector3f deltaPosition = current.position() - previous.position();
			Quaternion deltaRotation = previous.rotation().inverse() * current.rotation();

			Transform3<T> t = current;
			t.rotate(deltaRotation);
			t.translate(deltaPosition);
			return interpolate(current, t, dist01);
		}

		/** Compute a trnasformation made by compsoiting a parent and child transformations.
		 * \param parentTr the parent
		 * \param childTr the child
		 * \return the composite transformation
		 */
		static Transform3<T>	computeFinal( const Transform3<T>& parentTr, const Transform3<T>& childTr ) {
			Transform3<T>		finalTr;
			finalTr.position(parentTr.position() + parentTr.rotation() * childTr.position());
			finalTr.rotation(parentTr.rotation() * childTr.rotation());
			return finalTr;
		}

		/** Equality operator with a 1e-3 tolerance.
		 *\param other transformation to test equality with
		 *\return true if other is equal
		 */
		bool operator==(const Transform3 & other) const {
			static const float eps = 1e-3f;
			return (_position-other._position).norm()/ _position.norm() < eps && std::abs(_rotation.dot(other._rotation)) > ( 1 - eps);
		}

		/** Difference operator.
		 *\param other transformation to test difference with
		 *\return true if other is different
		 **/
		bool operator!=(const Transform3 & other) const {
			return !(*this == other);
		}

	private:
		Vector3		    _position;
		Quaternion		_rotation;
		Vector3			_scale;
	};

	/// Helper def.
	typedef Transform3<float> Transform3f;

	/** Write transformation to a byte stream.
	 *\param stream the byte stream
	 *\param t the transform
	 *\return the stream for compositing
	 \ingroup sibr_system
	 */
	template <typename T>
	ByteStream&		operator << (ByteStream& stream, const Transform3<T>& t ) {
		typename Transform3<T>::Vector3 v = t.position();
		typename Transform3<T>::Quaternion q = t.rotation();
		return stream
			<< v.x() << v.y() << v.z()
			<< q.x() << q.y() << q.z() << q.w();
	}

	/** Read transformation from a byte stream.
	 *\param stream the byte stream
	 *\param t the transform
	 *\return the stream for compositing
	 \ingroup sibr_system
	 */
	template <typename T>
	ByteStream&		operator >> (ByteStream& stream, Transform3<T>& t ) {
		typename Transform3<T>::Vector3 v;
		typename Transform3<T>::Quaternion q;
		stream
			>> v.x() >> v.y() >> v.z()
			>> q.x() >> q.y() >> q.z() >> q.w();
		t.position(v);
		t.rotation(q);
		return stream;
	}

	//==================================================================//
	// Inlines
	//==================================================================//

	template <typename T>
	void		Transform3<T>::translate( float x, float y, float z ) {
		_position.x() += x; _position.y() += y; _position.z() += z;
	}

	template <typename T>
	void		Transform3<T>::translate( float x, float y, float z,
		const Transform3<T>& ref) {
			translate( Vector3( x, y, z ), ref );
	}

	template <typename T>
	void		Transform3<T>::translate( const Vector3& v ) {
		_position.x() += v.x(); _position.y() += v.y(); _position.z() += v.z();
	}

	template <typename T>
	void		Transform3<T>::translate( const Vector3& v, const Transform3& ref ) {
		translate( ref.rotation().operator*(v) );
	}

	template<typename T>
	inline void Transform3<T>::scale(const float& s)
	{
		_scale = Vector3(s, s, s);
	}

	template <typename T>
	void		Transform3<T>::position( float x, float y, float z ) {
		_position.x() = x; _position.y() = y; _position.z() = z;
	}

	template <typename T>
	void		Transform3<T>::position( const Vector3& v ) {
		_position.x() = v.x(); _position.y() = v.y(); _position.z() = v.z();
	}

	template <typename T>
	const typename Transform3<T>::Vector3&	Transform3<T>::position( void ) const {
		return _position;
	}


	template <typename T>
	void		Transform3<T>::rotate( const Quaternion& rotation ) {
		_rotation = rotation * _rotation;
		_rotation.normalize();
	}

	template <typename T>
	void		Transform3<T>::rotate( float x, float y, float z ) {
		Quaternion q = quatFromEulerAngles(Vector3(x, y, z));
		q.normalize();
		rotate(q);
	}

	template <typename T>
	void		Transform3<T>::rotate( const Vector3& v ) {
		rotate( v.x(), v.y(), v.z() );
	}

	template <typename T>
	void		Transform3<T>::rotate( const Vector3& v, const Transform3& ref ) {
		rotate( v.x(), v.y(), v.z(), ref );
	}

	template <typename T>
	void		Transform3<T>::rotation( float x, float y, float z ) {
		_rotation = quatFromEulerAngles(Vector3(x, y, z));
	}

	template <typename T>
	void		Transform3<T>::rotation( const Vector3& v ) {
		rotation( v.x(), v.y(), v.z() );
	}

	template <typename T>
	void		Transform3<T>::rotation( const Quaternion& q ) {
		_rotation = q;
	}

	template <typename T>
	const typename Transform3<T>::Quaternion&	Transform3<T>::rotation( void ) const {
		return _rotation;
	}

	template <typename T>
	Matrix4f Transform3<T>::matrix( void ) const {
		Matrix4f trans = matFromQuat(_rotation);
		Matrix4f scaleMat = Matrix4f::Identity();
		scaleMat(0, 0) = _scale.x();
		scaleMat(1, 1) = _scale.y();
		scaleMat(2, 2) = _scale.z();

		trans = matFromTranslation(_position) * trans * scaleMat; // Opti (direct)

		return trans;
	}

	template <typename T>
	Matrix4f Transform3<T>::invMatrix( void ) const {
		// This is wrapped so we can (in the future) add a policy class
		// to enable caching this inv matrix
		return matrix().inverse();
	}

	template <typename T>
	void		Transform3<T>::rotate( float x, float y, float z,
		const Transform3<T>& ref)
	{
		Quaternion q = quatFromEulerAngles(Vector3(x, y, z));
		q.normalize();

		if ( &ref == this ) // Local Rotation
		{
			_rotation = _rotation * q;
			_rotation.normalize();
		}
		else
		{
			Quaternion refConj = ref.rotation();
			refConj.conjugate();

			// 1) Apply global rotation of ref on 'q' (ref * q)
			// 2) Apply local rotation of ref.conj (~inv) on 'q' (q*ref.conj)
			// 3) The rotation is converted and can be applied using rotate
			rotate((ref.rotation() * q) * refConj);
		}
	}


} // namespace sibr
