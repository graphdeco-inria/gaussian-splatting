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
# include "core/system/Vector.hpp"
# include "core/graphics/Image.hpp"


namespace sibr
{
	/**
	* \addtogroup sibr_graphics
	* @{
	*/

	/** Generate a random color.
	\return a random RGB triplet
	*/
	template<typename T_Type>
	static Eigen::Matrix<T_Type, 3, 1, Eigen::DontAlign> randomColor(){
		// We just use rand here, we don't need 'proper' PRNG.
		const uint8_t r = uint8((std::rand() % 255 + 192) * 0.5f);
		const uint8_t g = uint8((std::rand() % 255 + 192) * 0.5f);
		const uint8_t b = uint8((std::rand() % 255 + 192) * 0.5f);
		const sibr::Vector3ub output(r, g,b);
		return output.unaryExpr([](float f) { return f * sibr::opencv::imageTypeRange<T_Type>(); }).template cast<T_Type>();
	}

	/** Generate a color for a given scalar score, using the jet color map.
	\param gray the probability value
	\return the associated jet color.
	*/
	template<typename T_Type>
	static Eigen::Matrix<T_Type, 3, 1, Eigen::DontAlign> jetColor(float gray)
	{
		sibr::Vector3f output(1, 1, 1);
		float g = std::min(1.0f, std::max(0.0f, gray));
		float dg = 0.25f;
		float d = 4.0f;
		if (g < dg) {
			output.x() = 0.0f; 
			output.y() = d*g;
		} else if (g < 2.0f*dg) {
			output.x() = 0.0f; 
			output.z() = 1.0f + d*(dg - g);
		} else if (g < 3.0f*dg) {
			output.x() = d*(g - 0.5f); 
			output.z() = 0.0f;
		} else {
			output.y() = 1.0f + d*(0.75f - g);  
			output.z() = 0.0f;
		}

		return output.unaryExpr([](float f) { return f * sibr::opencv::imageTypeRange<T_Type>(); }).template cast<T_Type>();
	}

	/** Generate a jet color associated to the input probability, as a 3-channels cv::Scalar.
	\param gray the probability value
	\return the associated jet color.
	*/
	SIBR_GRAPHICS_EXPORT cv::Scalar jetColor(float gray);

	/** Generate a color for a given scalar score, using a reversible mapping.
	\param proba the probability value
	\return the associated color
	*/
	SIBR_GRAPHICS_EXPORT sibr::Vector3ub getLinearColorFromProbaV(double proba);

	/** Convert a color to the associated scalar score, using a reversible mapping.
	\param color the color
	\return the probability value
	*/
	SIBR_GRAPHICS_EXPORT double getProbaFromLinearColor(const sibr::Vector3ub & color);

	/** Convert a direction from cartesian to spherical coordinates.
	\param dir a direction in cartesian 3D space
	\return the spherical coordinates [phi,theta] in [-pi,pi]x[0,pi]
	\warning dir is assumed to be normalized
	*/
	SIBR_GRAPHICS_EXPORT sibr::Vector2d cartesianToSpherical(const sibr::Vector3d & dir); 

	/** Convert a direction from cartesian to spherical UVs.
	\param dir a direction in cartesian 3D space
	\return the spherical UVs [u,v] in [0,1]^2
	\warning dir is assumed to be normalized
	*/
	SIBR_GRAPHICS_EXPORT sibr::Vector2d cartesianToSphericalUVs(const sibr::Vector3d & dir);

	/**Inplace conversion of float image from sRGB space to linear.
	\param img the image to convert
	*/
	SIBR_GRAPHICS_EXPORT void sRGB2Lin(sibr::ImageRGB32F& img);

	/** Inplace conversion of a float image from linear space to sRGB.
	\param img the image to convert
	*/
	SIBR_GRAPHICS_EXPORT void lin2sRGB(sibr::ImageRGB32F& img);

	/** Debug helper: wrap a rendering task in an openGL debug group (visible in Renderdoc).
	\param s debug group name
	\param f the task to wrap
	\param args the task arguments
	*/
	template<typename FunType, typename ...ArgsType>
	void renderTask(const std::string & s, FunType && f, ArgsType && ... args) {
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, s.c_str());
		f(args...);
		glPopDebugGroup();
	};

	/** Interpolate between two values.
	\param A first value
	\param B second value
	\param fac interpolation factor
	\return A+fac*(B-A)
	*/
	inline float lerp( float A, float B, float fac ) {
		return A*(1.f-fac)+B*fac;
	}

	/** Express a value as the linear combination of two other values.
	\param from first value
	\param to second value
	\param current value to express as a combination
	\return the interpolation factor
	*/
	inline float inverseLerp( float from, float to, float current ) {
		return (current - from)/(to - from);
	}

	/*** @} */

} // namespace sibr

