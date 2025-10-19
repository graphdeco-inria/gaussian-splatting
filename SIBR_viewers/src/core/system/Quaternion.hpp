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

# include <Eigen/Core>
# include <Eigen/Geometry>
# include "core/system/Config.hpp"
# include "core/system/Matrix.hpp"
# include "core/system/Vector.hpp"

namespace sibr
{
	/**
	\addtogroup sibr_system
	@{
	*/

	/** Build a quaternion from a rotation matrix
	 *\param m the rotation matrix
	 *\return the quaternion
	 *\todo Seems to be different from sibr::Quaternion(rotationmatrix)
	 */
	template <typename T, int Options>
	Eigen::Quaternion<T, 0>	quatFromMatrix(const Eigen::Matrix<T, 3, 3, Options, 3, 3>& m) {
		Eigen::Quaternion<T, 0> q;
		float trace = m(0, 0) + m(1, 1) + m(2, 2) + 1.f;
		if (trace > 0)
		{
			float s = 0.5f / sqrtf(trace);
			q.x() = (m(1, 2) - m(2, 1)) * s;
			q.y() = (m(2, 0) - m(0, 2)) * s;
			q.z() = (m(0, 1) - m(1, 0)) * s;
			q.w() = 0.25f / s;
		}
		else
		{
			if ((m(0, 0) > m(1, 1)) && (m(0, 0) > m(2, 2)))
			{
				float s = sqrtf(1.f + m(0, 0) - m(1, 1) - m(2, 2)) * 2.f;
				q.x() = 0.5f / s;
				q.y() = (m(1, 0) + m(0, 1)) / s;
				q.z() = (m(2, 0) + m(0, 2)) / s;
				q.w() = (m(2, 1) + m(1, 2)) / s;
			}
			else if (m(1, 1) > m(2, 2))
			{
				float s = sqrtf(1.f - m(0, 0) + m(1, 1) - m(2, 2)) * 2.f;
				q.x() = (m(1, 0) + m(0, 1)) / s;
				q.y() = 0.5f / s;
				q.z() = (m(2, 1) + m(1, 2)) / s;
				q.w() = (m(2, 0) + m(0, 2)) / s;
			}
			else
			{
				float s = sqrtf(1.f - m(0, 0) - m(1, 1) + m(2, 2)) * 2.f;
				q.x() = (m(2, 0) + m(0, 2)) / s;
				q.y() = (m(2, 1) + m(1, 2)) / s;
				q.z() = 0.5f / s;
				q.w() = (m(1, 0) + m(0, 1)) / s;
			}
		}
		return q;
	}

	/** Build a quaternion from a rotation matrix
	 *\param m the rotation matrix
	 *\return the quaternion
	 */
	template <typename T, int Options>
	Eigen::Quaternion<T, 0>	quatFromMatrix( const Eigen::Matrix<T, 4,4, Options, 4,4>& m ) {
		Eigen::Quaternion<T, 0> q;
		float trace = m(0, 0) + m(1, 1) + m(2, 2) + 1.f;
		if (trace > 0)
		{
			float s = 0.5f / sqrtf(trace);
			q.x() = (m(1, 2) - m(2, 1)) * s;
			q.y() = (m(2, 0) - m(0, 2)) * s;
			q.z() = (m(0, 1) - m(1, 0)) * s;
			q.w() = 0.25f / s;
		}
		else
		{
			if ((m(0, 0) > m(1, 1)) && (m(0, 0) > m(2, 2)))
			{
				float s = sqrtf(1.f + m(0, 0) - m(1, 1) - m(2, 2)) * 2.f;
				q.x() = 0.5f / s;
				q.y() = (m(1, 0) + m(0, 1)) / s;
				q.z() = (m(2, 0) + m(0, 2)) / s;
				q.w() = (m(2, 1) + m(1, 2)) / s;
			}
			else if (m(1, 1) > m(2, 2))
			{
				float s = sqrtf(1.f - m(0, 0) + m(1, 1) - m(2, 2)) * 2.f;
				q.x() = (m(1, 0) + m(0, 1)) / s;
				q.y() = 0.5f / s;
				q.z() = (m(2, 1) + m(1, 2)) / s;
				q.w() = (m(2, 0) + m(0, 2)) / s;
			}
			else
			{
				float s = sqrtf(1.f - m(0, 0) - m(1, 1) + m(2, 2)) * 2.f;
				q.x() = (m(2, 0) + m(0, 2)) / s;
				q.y() = (m(2, 1) + m(1, 2)) / s;
				q.z() = 0.5f / s;
				q.w() = (m(1, 0) + m(0, 1)) / s;
			}
		}
		return q;
	}

	/** Build a quaternion from rotation euler angles.
	 *\param deg the rotation angles
	 *\return the quaternion
	 *\todo Explicit the angles order (yaw, pitch, roll?)
	 */
	template <typename T, int Options>
	Eigen::Quaternion<T, 0>	quatFromEulerAngles( const Eigen::Matrix<T, 3, 1,Options>& deg ) {
		Vector3f v(SIBR_DEGTORAD(deg.x()), SIBR_DEGTORAD(deg.y()), SIBR_DEGTORAD(deg.z()));
		Vector3f halfAngles( v.x() * 0.5f, v.y() * 0.5f, v.z() * 0.5f );

		const float cx = cosf (halfAngles.x());
		const float sx = sinf (halfAngles.x());
		const float cy = cosf (halfAngles.y());
		const float sy = sinf (halfAngles.y());
		const float cz = cosf (halfAngles.z());
		const float sz = sinf (halfAngles.z());

		const float cxcz = cx*cz;
		const float cxsz = cx*sz;
		const float sxcz = sx*cz;
		const float sxsz = sx*sz;

		Eigen::Quaternion<T, 0> dst;
		dst.vec().x() = (cy * sxcz) - (sy * cxsz);
		dst.vec().y() = (cy * sxsz) + (sy * cxcz);
		dst.vec().z() = (cy * cxsz) - (sy * sxcz);
		dst.w() = (cy * cxcz) + (sy * sxsz);
		return dst;
	}

	/** Rotate a vector using a quaternion.
	 *\param rotation the quaternion
	 *\param vec the vector
	 *\return the rotated vector.
	 */
	template <typename T, int Options>
	Eigen::Matrix<T, 3, 1, Options>	quatRotateVec(
		const Eigen::Quaternion<T, 0>& rotation, const Eigen::Matrix<T, 3, 1, Options>& vec ) {
		return rotation._transformVector(vec);
	}

	/** Quaternion product.
	 * \param q1 first quaternion
	 * \param q2 second quaternion
	 * \return the result quaternion
	 */
	template <typename T>
	inline static Eigen::Quaternion<T> dot( const Eigen::Quaternion<T>& q1, const Eigen::Quaternion<T>& q2 ) {
		return q1.vec().dot(q2.vec()) + q1.w()*q2.w();
	}

	/** Compute the delta angle between two quaternions.
	 *\param q1 first quaternion
	 *\param q2 second quaternion
	 *\return the angle in radians
	 *\note Will return the smallest angle possible
	 */
	template <typename T>
	inline static float		angleRadian( const Eigen::Quaternion<T>& q1, const Eigen::Quaternion<T>& q2 ) {
		const float mid = 3.14159f;
		const float angle = q1.angularDistance(q2);
		return angle > mid? mid-angle : angle; // be sure to return the shortest angle
	}

    /** Linear quaternion interpolation
     *\param q1 first quaternion
	 *\param q2 second quaternion
	 *\param t interpolation factor
	 *\return the interpolated quaternion
	 */
	template <typename T>
    inline static Eigen::Quaternion<T> lerp( const Eigen::Quaternion<T>& q1, const Eigen::Quaternion<T>& q2, float t ) {
		return (q1*(1-t) + q2*t).normalized();
	}

	/** Spherical quaternion interpolation
	*\param q1 first quaternion
	*\param q2 second quaternion
	*\param t interpolation factor
	*\return the interpolated quaternion
	*/
	template <typename T>
	static Eigen::Quaternion<T> slerp( const Eigen::Quaternion<T>& q1, const Eigen::Quaternion<T>& q2, float t ) {
		Eigen::Quaternion<T> q3;
		float dot = q1.dot(q2);// Eigen::Quaternion<T>::dot(q1, q2);
		// dot = cos(theta)
		// 	 if (dot < 0), q1 and q2 are more than 90 degrees apart,
		// 	 so we can invert one to reduce spinning
		if (dot < 0)
		{
			dot = -dot;
			q3 = -q2;
		} else q3 = q2;
		if (dot < 0.95f)
		{
			float angle = acosf(dot);
			return (q1*sinf(angle*(1-t)) + q3*sinf(angle*t))/sinf(angle);
		} else // if the angle is small, use linear interpolation
			return lerp(q1,q3,t);
	}

	typedef	Eigen::Quaternion<unsigned>		Quaternionu;
	typedef	Eigen::Quaternion<int>			Quaternioni;
	typedef	Eigen::Quaternion<float>		Quaternionf;
	typedef	Eigen::Quaternion<double>		Quaterniond;

	/** }@ */

} // namespace sibr
