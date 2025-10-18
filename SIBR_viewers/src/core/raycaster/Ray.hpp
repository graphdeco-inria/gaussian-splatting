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

# include <string>
# include <vector>
# include <core/system/Vector.hpp>
# include "core/raycaster/Config.hpp"

namespace sibr
{

	///
	/// Represents a simple ray
	/// \ingroup sibr_raycaster
	///
	class SIBR_RAYCASTER_EXPORT Ray
	{
	public:
		/** Construct a ray from parameters.
		\param orig ray origin
		\param dir ray direction
		\note The direction will be normalized.
		*/
		Ray( const sibr::Vector3f& orig = sibr::Vector3f(0.f, 0.f, 0.f),
			const sibr::Vector3f& dir = sibr::Vector3f(0.f, 0.f, -1.f) );

		/** Set the position from where the ray starts.
		\param o the new origin
		*/
		inline void		orig( const sibr::Vector3f& o );
		
		/// \return the ray origin
		inline const sibr::Vector3f&	orig( void ) const;

		/** Set the direction of the ray. Additionally,
		 you can precise if you want this direction to be automatically
		 normalized or not.
		 \param d the new direction
		 \param normalizeIt should normalization be applied
		 */
		inline void		dir( const sibr::Vector3f& d, bool normalizeIt=true );

		/// \return the direction of the ray.
		inline const sibr::Vector3f&	dir( void ) const;

		/** Return the 3D point such that p = orig + t * dir;
		\param t the distance along the ray
		\return the 3D point
		*/
		Vector3f at(float t) const;

	private:
		sibr::Vector3f		_orig;	///< Position from where the ray starts
		sibr::Vector3f		_dir;	///< Direction where the ray goes
	};

	///
	/// Contains information about a ray hit
	/// \ingroup sibr_raycaster
	///
	class SIBR_RAYCASTER_EXPORT RayHit
	{
	public:
		static const float	InfinityDist;

		/// Infos about the object that was hit
		struct Primitive
		{
			uint triID;		///< triangle id of the mesh that was hit
			uint geomID;	///< mesh id loaded in the raycaster
			uint instID;	///< id of the instance loaded in the raycaster
		};

		/// Barycentric coordinates
		struct BCCoord
		{
			float u;	///< u-coordinates (ranging from 0.0 to 1.0)
			float v;	///< v-coordinates (ranging from 0.0 to 1.0)
		};
		
		/** Construct a hit record.
		\param r the ray
		\param dist intersection distance
		\param coord barycentric coordinates
		\param normal surface normal
		\param prim intersected primitive
		*/
		RayHit( const Ray& r, float dist, const BCCoord& coord,
			const sibr::Vector3f& normal, const Primitive& prim );

		/// Non-hit constructor.
		RayHit() {};

		/// \return the ray that was casted
		inline const Ray&			ray( void ) const;

		/// \return the distance from the ray origin to the hit
		inline float				dist( void ) const;

		/// \return the barycentric coordinates of the hit point on the triangle that was hit
		inline const BCCoord&		barycentricCoord( void ) const;
		
		/** Return the proper barycentric factors for interpolating information stored
		 at each vertex of a triangle.
		 e.g: get fragment color using
		   color = factor[0]*colorVert0 + factor[1]*colorVert1 + factor[2]*colorVert2
		 It consider the following triangle: https://embree.github.io/images/triangle_uv.png
		\return the barycentric coordinates
		*/
		sibr::Vector3f			interpolateUV( void ) const;

		/// \return the normal of the triangle that was hit.
		inline const sibr::Vector3f&			normal( void ) const;

		/// \return information about the primitive that was hit.
		inline const Primitive&		primitive( void ) const;

		/// \return true if an object was hit.
		inline bool	hitSomething( void ) const;

	private:
		Ray			_ray;		///< casted ray
		float		_dist;		///< distance from the ray's origin to the hit
		BCCoord		_coord;		///< barycentric coordinate on the triangle that was hit
		sibr::Vector3f	_normal;///< normal of the triangle that was hit
		Primitive	_prim;		///< infos about the primitive that was hit
	};

	///// DEFINITION /////
	
	void		Ray::orig( const sibr::Vector3f& o ) {
		_orig = o;
	}
	const sibr::Vector3f&	Ray::orig( void ) const {
		return _orig;
	}

	void		Ray::dir( const sibr::Vector3f& d, bool normalizeIt) {
		_dir = (normalizeIt)? sibr::Vector3f(d.normalized()) : d;
	}
	const sibr::Vector3f&	Ray::dir( void ) const {
		return _dir;
	}

	inline Vector3f Ray::at(float t) const
	{
		return orig() + t * dir();
	}



	const Ray&			RayHit::ray( void ) const {
		return _ray;
	}
	float				RayHit::dist( void ) const {
		return _dist;
	}
	const RayHit::BCCoord&		RayHit::barycentricCoord( void ) const {
		return _coord;
	}
	const sibr::Vector3f&			RayHit::normal( void ) const {
		return _normal;
	}
	const RayHit::Primitive&		RayHit::primitive( void ) const {
		return _prim;
	}

	bool	RayHit::hitSomething( void ) const {
		return (_dist != RayHit::InfinityDist);
	}


} // namespace sibr
