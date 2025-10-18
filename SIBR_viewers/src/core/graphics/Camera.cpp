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


#include "core/graphics/Camera.hpp"

namespace sibr
{
	
	ByteStream&		operator << (ByteStream& stream, const Camera& c )
	{
		Camera::Transform3f t = c.transform();
		float fovy = c.fovy();
		float aspect = c.aspect();
		float znear = c.znear();
		float zfar = c.zfar();
		return stream
			<< t << fovy << aspect << znear << zfar;
	}

	ByteStream&		operator >> (ByteStream& stream, Camera& c )
	{
		Camera::Transform3f t;
		float fovy = 0.f;
		float aspect = 0.f;
		float znear = 0.f;
		float zfar = 0.f;
		stream
			>> t >> fovy >> aspect >> znear >> zfar;
		c.transform(t);
		c.fovy(fovy);
		c.aspect(aspect);
		c.znear(znear);
		c.zfar(zfar);
		return stream;
	}

	void	Camera::perspective( float fovRad, float ratio, float znear, float zfar )
	{
		_fov = fovRad;
		_aspect = ratio;
		_znear = znear;
		_zfar = zfar;
		_dirtyViewProj = true;
	}
	
	Vector3f			Camera::project( const Vector3f& p3d ) const
	{
		Vector4f p4d;
		p4d[0] = p3d[0]; p4d[1] = p3d[1]; p4d[2] = p3d[2]; p4d[3] = 1.0;
		Vector4f p3d_t = viewproj() * p4d;
		p3d_t = p3d_t / p3d_t[3];
		//p3d_t[2] = p3d_t[2]*0.5f + 0.5f; // [-1;1] to [0;1] // not used
		return Vector3f(p3d_t[0], p3d_t[1], p3d_t[2]); // so return [-1;1]
		
		//p3d_t[2] = p3d_t[2]*0.5f + 0.5f; // [-1;1] to [0;1] // not used
		//return Vector3f(p3d_t[0], p3d_t[1], p3d_t[2]); // so return [-1;1]
	}

	Vector3f			Camera::unproject( const Vector3f& p3d ) const
	{
		Vector4f p4d;
		p4d[0] = p3d[0]; p4d[1] = p3d[1]; p4d[2] = p3d[2]; p4d[3] = 1.0;
		//p4d[2] = p4d[2]*2.f - 1.f; // [0;1] to [-1;1]  // not used
		Vector4f p3d_t = invViewproj() * p4d;//;viewproj().inverse() * p4d;
		return Vector3f(p3d_t[0],p3d_t[1],p3d_t[2])/p3d_t[3];
	}

	bool		Camera::frustumTest(const Vector3f& position3d, const Vector2f& pixel2d) const
	{
		return (pixel2d.cwiseAbs().array() < (1.0f-1e-5f) ).all() && (dir().dot(position3d - position()) > 0);
	}

	bool		Camera::frustumTest(const Vector3f& position3d) const
	{
		return frustumTest(position3d, project(position3d).xy());
	}

	void		Camera::forceUpdateViewProj( void ) const
	{
		_matViewProj = sibr::Matrix4f(proj()*view());
		//_matViewProj = proj()*view();
		_invMatViewProj = _matViewProj.inverse();
		_dirtyViewProj = false;
	}

	Vector3f			Camera::dir( void ) const
	{ 
		return quatRotateVec(rotation(), Vector3f( 0.f, 0.f,-1.f));
	}

	Vector3f			Camera::up( void ) const	
	{ 
		return quatRotateVec(rotation(), Vector3f( 0.f, 1.f, 0.f));
	}

	Vector3f			Camera::right( void ) const	
	{ 
		return quatRotateVec(rotation(), Vector3f( 1.f, 0.f, 0.f));
	}

	Matrix4f	Camera::proj( void ) const
	{
		//std::cout << "FOV: " << _fov << "Aspect" << _aspect << "ZNEAR" << _znear << "ZFAR" << _zfar << std::endl << std::flush;
		if (ortho())
			return sibr::orthographic(_right, _top, _znear, _zfar);
		else
			return sibr::perspective(_fov, _aspect, _znear, _zfar, _p);
	}

	/*static*/ Camera	Camera::interpolate( const Camera& from, const Camera& to, float dist01 )
	{
		dist01 = std::max(0.f, std::min(1.f, dist01));
		Transform3f t = Transform3f::interpolate(from._transform, to._transform, dist01);
		Camera out = from;
		out._transform = t;
		out.fovy(dist01*from.fovy() + (1.0f-dist01)*to.fovy());
		out.aspect(dist01*from.aspect() + (1.0f-dist01)*to.aspect());
		out.zfar(dist01*from.zfar() + (1.0f-dist01)*to.zfar());
		out.znear(dist01*from.znear() + (1.0f-dist01)*to.znear());
		if (from.ortho()) {
			out.orthoRight(dist01*from.orthoRight() + (1.0f-dist01)*to.orthoRight());
			out.orthoTop(dist01*from.orthoTop() + (1.0f-dist01)*to.orthoTop());
		}
		return out;
	}

	void Camera::setStereoCam(bool isLeft, float focal, float iod) 
	{
		_matViewProj = sibr::perspectiveStereo(_fov, _aspect, _znear, _zfar, focal, iod, isLeft)*view();
		_invMatViewProj = _matViewProj.inverse();
	}

	void Camera::setOrthoCam(float right, float top)
	{
		_matViewProj = sibr::orthographic(right,top,_znear,_zfar)*view();
		_invMatViewProj = _matViewProj.inverse();
		_dirtyViewProj = false;
		_isOrtho = true;
		_right = right;
		_top = top;
	}
	
	void 					Camera::transform( const Transform3f& t )
	{
		_transform = t;
		_dirtyViewProj = true;
	}

} // namespace sibr