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

#include "core/raycaster/Config.hpp"
#include <vector>
#include <core/system/Vector.hpp>
#include <core/assets/InputCamera.hpp>



/** Struct representing a 3D quad, along with load/save utilities.
	\todo move in proper namespace without breaking anything.
	\ingroup sibr_raycaster
*/
struct  quad {

	sibr::Vector3f q1;
	sibr::Vector3f q2;
	sibr::Vector3f q3;
	sibr::Vector3f q4;

	/** Save quad to file on disk.
	\param path destination file.
	*/
	void save(std::string path) {
		sibr::ByteStream bs;
		bs << q1.x() << q1.y() << q1.z()
			<< q2.x() << q2.y() << q2.z()
			<< q3.x() << q3.y() << q3.z()
			<< q4.x() << q4.y() << q4.z();
		bs.saveToFile(path);
	}

	/** Load quad from file on disk.
	\param path source file.
	*/
	void load(std::string path) {
		sibr::ByteStream bs;
		bs.load(path);
		bs >> q1.x() >> q1.y() >> q1.z()
			>> q2.x() >> q2.y() >> q2.z()
			>> q3.x() >> q3.y() >> q3.z()
			>> q4.x() >> q4.y() >> q4.z();
	}

};

namespace sibr {


	/** This class provides utilities to compute point/line/triangle/quad intersections.
	\ingroup sibr_raycaster
	*/
	class SIBR_RAYCASTER_EXPORT Intersector2D
	{

	public:
		/// Constructor.
		Intersector2D(void) = delete;

		/// Destructor
		~Intersector2D(void) = delete;

		/**
		Having defined a straight line in the 2D plane, this method can be used to know in which half-space (defined by the line) a point lies.
		\param p1 the 2D point to locate wrt to the line.
		\param p2 a 2D point on the line.
		\param p3 another 2D point on the line.
		\return a signed value indicating on which side of the line the point is.
		*/
		static float sign(sibr::Vector2f p1, sibr::Vector2f p2, sibr::Vector2f p3);

		/**
		Tests if a point falls inside a triangle, in 2D space.
		\param pt the point to test.
		\param v1 first triangle vertex.
		\param v2 second triangle vertex.
		\param v3 third triangle vertex.
		\return a boolean denoting if the point belong to the triangle or not.
		*/
		static bool PointInTriangle(sibr::Vector2f pt, sibr::Vector2f v1, sibr::Vector2f v2, sibr::Vector2f v3);

		/**
		Tests if a line intersects another line, in 2D space.
		\param a first point on the first line.
		\param b second point on the first line.
		\param c first point on the second line.
		\param d second point on the second line.
		\return a boolean denoting if the lines intersects.
		*/
		static bool LineLineIntersect(sibr::Vector2f a, sibr::Vector2f b, sibr::Vector2f c, sibr::Vector2f d);

		/**
		Tests if two triangles overlap, in 2D space.
		\param t0_0 first vertex of the first triangle.
		\param t0_1 second vertex of the first triangle.
		\param t0_2 third vertex of the first triangle.
		\param t1_0 first vertex of the second triangle.
		\param t1_1 second vertex of the second triangle.
		\param t1_2 third vertex of the second triangle.
		\return a boolean denoting if the triangles overlap.
		*/
		static bool TriTriIntersect(sibr::Vector2f t0_0, sibr::Vector2f t0_1, sibr::Vector2f t0_2,
			sibr::Vector2f t1_0, sibr::Vector2f t1_1, sibr::Vector2f t1_2);

		/**
		Tests if two quads overlap, in 2D space.
		\param q0_0 first vertex of the first quad.
		\param q0_1 second vertex of the first quad.
		\param q0_2 third vertex of the first quad.
		\param q0_3 fourth vertex of the first quad.
		\param q1_0 first vertex of the second quad.
		\param q1_1 second vertex of the second quad.
		\param q1_2 third vertex of the second quad.
		\param q1_3 fourth vertex of the second quad.
		\return a boolean denoting if the quads overlap.
		*/
		static bool QuadQuadIntersect(sibr::Vector2f q0_0, sibr::Vector2f q0_1, sibr::Vector2f q0_2, sibr::Vector2f q0_3,
			sibr::Vector2f q1_0, sibr::Vector2f q1_1, sibr::Vector2f q1_2, sibr::Vector2f q1_3);

		/**
		Perform multiple quads/camera frusta intersections at once.
		\warning Requires an existing and current OpenGL context.
		\param quads an array of quads to test against each camera frustum.
		\param cams an array of cameras against which frusta the intersections tests should be performed.
		\return a double-array of booleans denoting, for each camera, for each quad, if the quad intersects the frustum volume.
		*/
		static std::vector<std::vector<bool>> frustrumQuadsIntersect(std::vector<quad> & quads, const std::vector<InputCamera::Ptr> & cams);

	};

}