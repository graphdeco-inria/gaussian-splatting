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

# include "core/system/Config.hpp"



namespace Eigen
{
	/**
	* \addtogroup sibr_system
	* @{
	*/

	// The following operators work with Eigen structs, so
	// they must be declared in the namespace Eigen (or
	// you would have to do sibr::operator < (left, right)
	// instead of simple left < right)

	/** Lexicographic comparison (from left to right).
	 *\param left first element
	 *\param right second element
	 *\return true if left is lexicographically smaller than right.
	 */
	template <typename T, int N, int Options>
	bool operator<(const Eigen::Matrix<T, N, 1, Options>& left, const Eigen::Matrix<T, N, 1, Options>& right) {

		for (int c = 0; c < N; c++) {
			if (left[c] < right[c]) return true;
			else if (left[c] > right[c]) return false;
		}
		return false; //case where they are equal

	}

	// stream

	/** Output matrix to a stream.
	 *\param s stream
	 *\param t matrix
	 *\return the stream for chaining
	 */
	template <typename T, int N, int Options>
	std::ostream& operator<<( std::ostream& s, const Eigen::Matrix<T, N, 1, Options>& t ) {
		s << '(';
		for (uint i=0; i<N; i++) { s << t[i]; if (i < N-1) s << ','; }
		s << ')';
		return (s);
	}

	/** Read matrix from a stream.
	 *\param s stream
	 *\param t matrix
	 *\return the stream for chaining
	 */
	template <typename T, int N, int Options>
	std::istream& operator>>( std::istream& s, Eigen::Matrix<T, N, 1, Options>& t ) {
		char tmp = 0;
		s >> tmp; // (
		for (int i = 0; i < N; ++i)
		{
			s >> t [i];
			s >> tmp; //, or )
		}

		return s;
	}

	/** @} */
}

namespace sibr
{

	/**
	* \addtogroup sibr_system
	* @{
	*/

	template <typename T, int N>
	using Vector = Eigen::Matrix<T, N, 1, Eigen::DontAlign>;

	/** Fractional part of each component.
	 *\param A vector
	 *\return the fractional matrix
	 **/
	template <typename T, int N, int Options>
	Eigen::Matrix<T, N, 1, Options>			frac( const Eigen::Matrix<T, N, 1, Options>& A ) {
		Eigen::Matrix<T, N, 1, Options> out = A;
		for (int i = 0; i < N; ++i)
			out[i] = out[i] - floor(out[i]);
		return out;
	}

	/** Distance between two vectors
	 *\param A first vector
	 *\param B second vector
	 *\return norm(A-B)
	 */
	template <typename T, int N, int Options>
	inline T			distance( const Eigen::Matrix<T, N, 1, Options>& A, const Eigen::Matrix<T, N, 1, Options>& B ) {
		return (A-B).norm();
	}

	/** Return the length of a vector.
	 *\param A vector
	 *\return norm(A)
	 */
	template <typename T, int N, int Options>
	inline T			length( const Eigen::Matrix<T, N, 1, Options>& A ) {
		return A.norm();
	}

	/** Return the squared length of a vector.
	 *\param A vector
	 *\return norm(A)^2
	 */
	template <typename T, int N, int Options>
	inline T			sqLength( const Eigen::Matrix<T, N, 1, Options>& A ) {
		return A.squaredNorm();
	}

	/** Compute the dot product of two vectors
	 *\param A first vector
	 *\param B second vector
	 *\return A.B
	 */
	template <typename T, int N, int Options>
	inline T			dot( const Eigen::Matrix<T, N, 1, Options>& A, const Eigen::Matrix<T, N, 1, Options>& B ) {
		return A.dot(B);
	}

	/** Compute the cross product of two vectors
	 *\param A first vector
	 *\param B second vector
	 *\return AxB
	 */
	template <typename T, int N, int Options>
	inline Eigen::Matrix<T, N, 1, Options>	cross( const Eigen::Matrix<T, N, 1, Options>& A, const Eigen::Matrix<T, N, 1, Options>& B ) {
		return A.cross(B);
	}

	/** Clamp each component of a vector between two values.
	 * \param A vector
	 * \param min min values vector
	 * \param max max values vector
	 * \return min(max(A, min), max)
	 */
	template <typename T, int N>
	inline Vector<T,N> clamp(const Vector<T, N>& A, const Vector<T, N> & min, const Vector<T, N> & max) {
		return A.cwiseMax(min).cwiseMin(max);
	}

	/** Compute the cotangent of the angle between two vectors.
	 *\param A first vector
	 *\param B second vector
	 *\return the cotangent
	 */
	template <typename T, int N, int Options>
	inline T cotan(const Eigen::Matrix<T, N, 1, Options>& A, const Eigen::Matrix<T, N, 1, Options>& B) {
		return A.dot(B) / A.cross(B).norm();
	}

	/** Convert an unsigned char color in [0,255] to a float color in [0,1].
	 *\param colorUB the color vector
	 *\return the [0,1] float vector
	 */
	SIBR_SYSTEM_EXPORT Eigen::Matrix<float, 3,	1, Eigen::DontAlign>  toColorFloat( Vector<unsigned char, 3> & colorUB );

	/** Convert a float color in [0,1] to an unsigned char color in [0,255].
	 *\param colorFloat the color vector
	 *\return the [0,255] float vector
	 */
	SIBR_SYSTEM_EXPORT Eigen::Matrix<unsigned char, 3,1,Eigen::DontAlign> toColorUB( Vector<float,3> & colorFloat );

	// Typedefs.

	typedef	Eigen::Matrix<float, 1,			1,Eigen::DontAlign>			Vector1f;
	typedef	Eigen::Matrix<int, 1,			1,Eigen::DontAlign>			Vector1i;

	typedef	Eigen::Matrix<unsigned, 2,		1,Eigen::DontAlign>			Vector2u;
	typedef	Eigen::Matrix<unsigned char, 2,	1,Eigen::DontAlign>			Vector2ub;
	typedef	Eigen::Matrix<int, 2,			1,Eigen::DontAlign>			Vector2i;
	typedef	Eigen::Matrix<float, 2,			1,Eigen::DontAlign>			Vector2f;
	typedef	Eigen::Matrix<double, 2,		1,Eigen::DontAlign>			Vector2d;

	typedef	Eigen::Matrix<unsigned, 3,		1,Eigen::DontAlign>			Vector3u;
	typedef	Eigen::Matrix<unsigned char, 3,	1,Eigen::DontAlign>			Vector3ub;
	typedef	Eigen::Matrix<unsigned short int, 3, 1,Eigen::DontAlign>	Vector3s;
	typedef	Eigen::Matrix<int, 3,			1,Eigen::DontAlign>			Vector3i;
	typedef	Eigen::Matrix<float, 3,			1,Eigen::DontAlign>			Vector3f;
	typedef	Eigen::Matrix<double, 3,		1,Eigen::DontAlign>			Vector3d;

	typedef	Eigen::Matrix<unsigned, 4,		1,Eigen::DontAlign>			Vector4u;
	typedef	Eigen::Matrix<unsigned char, 4,	1,Eigen::DontAlign>			Vector4ub;
	typedef	Eigen::Matrix<int, 4,			1,Eigen::DontAlign>			Vector4i;
	typedef	Eigen::Matrix<float, 4,			1,Eigen::DontAlign>			Vector4f;
	typedef	Eigen::Matrix<double, 4,		1,Eigen::DontAlign>			Vector4d;

	/**
		Return a 4x4 3D rotation matrix that aligns the first vector onto the second one.
		\param from source vector, current direction
		\param to destination vector, target direction
		\return the rotation matrix
	*/
	SIBR_SYSTEM_EXPORT Eigen::Matrix<float, 4, 4, Eigen::DontAlign> alignRotationMatrix(const sibr::Vector3f & from, const sibr::Vector3f & to);

	/** @} */
} // namespace sibr

