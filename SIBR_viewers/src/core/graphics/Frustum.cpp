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


// Partially based on:
// http://www.lighthouse3d.com/tutorials/view-frustum-culling

#include "core/graphics/Camera.hpp"
#include "core/graphics/Frustum.hpp"

namespace sibr
{
	Frustum::Frustum(const Camera& cam)
	{
		float ratio = cam.aspect();
		float angle = cam.fovy();
		float nearD = cam.znear();
		float farD = cam.zfar();

		// compute width and height of the near and far plane sections
		float tang = (float)tan(SIBR_DEGTORAD(angle) * 0.5);
		float nh = nearD * tang;
		float nw = nh * ratio;
		float fh = farD  * tang;
		float fw = fh * ratio;

		Vector3f nc, fc, X, Y, Z;
		const Vector3f& p = cam.position();

		// compute the Z axis of camera
		// this axis points in the opposite direction from
		// the looking direction
		Z = -cam.dir();

		// X axis of camera with given "up" vector and Z axis
		X = cam.up().cross(Z);
		X.normalize();

		// the real "up" vector is the cross product of Z and X
		Y = cam.up();

		// compute the centers of the near and far planes
		nc = p - Z * nearD;
		fc = p - Z * farD;

		_planes[NEARP].buildFrom( -Z, nc );
		_planes[FARP].buildFrom( Z, fc );

		Vector3f aux, normal;

		aux = (nc + Y*nh) - p;
		aux.normalize();
		normal = aux.cross(X);
		_planes[TOP].buildFrom( normal, nc + Y*nh );

		aux = (nc - Y*nh) - p;
		aux.normalize();
		normal = X.cross(aux);
		_planes[BOTTOM].buildFrom( normal, nc - Y*nh );

		aux = (nc - X*nw) - p;
		aux.normalize();
		normal = aux.cross(Y);
		_planes[LEFT].buildFrom( normal, nc - X*nw );

		aux = (nc + X*nw) - p;
		aux.normalize();
		normal = Y.cross(aux);
		_planes[RIGHT].buildFrom( normal, nc + X*nw );
	}

	Frustum::TestResult	Frustum::testSphere(const Vector3f& p, float radius)
	{
		float distance;
		TestResult result = INSIDE;

		for (int i = 0; i < 6; i++) {
			distance = _planes[i].distanceWithPoint(p);
			if (distance < -radius)
				return OUTSIDE;
			else if (distance < radius)
				result = INTERSECT;
		}
		return result;
	}

	float	Frustum::Plane::distanceWithPoint(const Vector3f& p)
	{
		// dist = A*rx + B*ry + C*rz + D = n . r  + D
		return A*p.x() + B*p.y() + C*p.z() + D;
	}
	void	Frustum::Plane::buildFrom(const Vector3f& normal, const Vector3f& point)
	{
		Vector3f n = normal;
		n.normalize();
		A = n.x();
		B = n.y();
		C = n.z();
		D = -n.dot(point);
	}


} // namespace sibr
