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



#include "core/system/Transform3.hpp"

namespace sibr
{
	Matrix4f perspective( float fovRadian, float ratio, float zn, float zf, const sibr::Vector2f & p)
	{
		const float yScale = float(1.0)/std::tan(fovRadian/2.0f);
		const float xScale = yScale/ratio;

		Matrix4f m;
		const float dx = 2.0f * p.x() - 1.0f;
		const float dy = 2.0f * p.y() - 1.0f;
		m << 
			xScale,    0,          dx,             0,
			0,    yScale,          dy,             0,
			0,         0, (zn+zf)/(zn-zf), 2*zn*zf/(zn-zf),
			0,         0,         -1,             0;

		return m;
	}

	Matrix4f perspectiveOffCenter(
			float left, float right, float bottom, float top, float mynear, float myfar )
	{      
		float x =  (2.0f * mynear) / (right - left);
		float y =  (2.0f * mynear) / (top - bottom);
		float a =  (right + left) / (right - left);

		float b =  (top + bottom) / (top - bottom);
		float c = -(myfar + mynear) / (myfar - mynear);
		float d = -(2.0f * myfar * mynear) / (myfar - mynear);
		float e = -1.0f;

		Matrix4f m;

		m << 
			x, 0, 0, 0, 
			0, y, 0, 0,
			a, b, c, e,
			0, 0, d, 0;

		return m;
	}

	Matrix4f perspectiveStereo(
			float fovRadian, float aspect, float zn, float zf, float focalDistance, float eyeDistance, bool isLeftEye )
	{

		float left, right;
		float a = float(1.0f)/std::tan(fovRadian/2.0f);
		float b = zf / focalDistance;

		if (isLeftEye)          // left camera
		{
			left  = - aspect * a + (eyeDistance) * b;
			right =   aspect * a + (eyeDistance) * b;
		}
		else                 // right camera
		{
			left  = - aspect * a - (eyeDistance) * b;
			right =   aspect * a - (eyeDistance) * b;
		}

		return perspectiveOffCenter(left, right, -a, a, zn, zf);
	}

	Matrix4f orthographic(float right, float top, float mynear, float myfar)
	{

		Matrix4f m;

		m <<
			1.0f/right, 0.0f,		0.0f,					0.0f,
			0.0f,		1.0f/top,	0.0f,					0.0f,
			0.0f,		0.0f,		-2.0f/(myfar-mynear),	-(myfar + mynear) / (myfar - mynear),
			0.0f,		0.0f,		0.0f,					1.0f;

		return m;
	}

	Matrix4f	lookAt(
			const Vector3f& eye,
			const Vector3f& center,
			const Vector3f& up )
	{
		const sibr::Vector3f f = (center - eye).normalized();
		sibr::Vector3f u = up.normalized();
		const sibr::Vector3f s = f.cross(u).normalized();
		u = s.cross(f);

		Eigen::Matrix<float, 4, 4, 0, 4, 4> res;
		res <<  s.x(),s.y(),s.z(),-s.dot(eye),
			u.x(),u.y(),u.z(),-u.dot(eye),
			-f.x(),-f.y(),-f.z(),f.dot(eye),
			0,0,0,1;

		return res;
	}

	void 	operator<< (std::ofstream& outfile, const Matrix4f& m)
	{
		outfile << m(0,0) << " " << m(0,1) << " " << m(0,2) << " " << m(0,3) 
			<< " " << m(1,0) << " " << m(1,1) << " " << m(1,2) << " " << m(1,3) 
			<< " " << m(2,0) << " " << m(2,1) << " " << m(2,2) << " " << m(2,3) 
			<< " " << m(3,0) << " " << m(3,1) << " " << m(3,2) << " " << m(3,3)  ;

	}

	void 	operator>>( std::ifstream& infile, Matrix4f& out)
	{
		float m[16];
		infile >> m[0] >> m[1] >> m[2] >> m[3] 
			>> m[4] >> m[5] >> m[6] >> m[7] 
			>> m[8] >> m[9] >> m[10] >> m[11]
			>> m[12] >> m[13] >> m[14] >> m[15];

			out << m[0] , m[1] , m[2] , m[3] 
				, m[4] , m[5] , m[6] , m[7] 
				, m[8] , m[9] , m[10] , m[11]
				, m[12] , m[13] , m[14] , m[15];
	}

} // namespace sibr
