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



#include "core/graphics/Utils.hpp"

namespace sibr
{


	cv::Scalar jetColor(float gray) {
		const sibr::Vector3ub col = jetColor<uchar>(gray);
		return toOpenCV<uchar, uchar, 3>(col);
	}

	sibr::Vector3ub getLinearColorFromProbaV(double proba) {
		const double scProba = 3.0 * proba;
		const unsigned char red = double(sibr::clamp(scProba, 0.0, 1.0)) * 255;
		const unsigned char green = double(sibr::clamp(scProba - 1, 0.0, 1.0)) * 255;
		const unsigned char blue = double(sibr::clamp(scProba - 2, 0.0, 1.0)) * 255;

		return sibr::Vector3ub(red, green, blue);
	}

	double getProbaFromLinearColor(const sibr::Vector3ub & color) {
		const double red = double(color[0]) / 255.0;
		const double green = double(color[1]) / 255.0;
		const double blue = double(color[2]) / 255.0;
		return (red + green + blue) / 3.0;
	}

	sibr::Vector2d cartesianToSpherical(const sibr::Vector3d & dir)
	{
		double theta = std::acos(dir.z());

		double phi = 0;
		if (dir.x() != 0 && dir.y() != 0) {
			phi = std::atan2(dir.y(), dir.x());
		}

		return sibr::Vector2d(phi, theta);
	}

	sibr::Vector2d cartesianToSphericalUVs(const sibr::Vector3d & dir)
	{
		const sibr::Vector2d angles = cartesianToSpherical(dir);
		const double & phi = angles[0];
		const double & theta = angles[1];

		return sibr::Vector2d(0.5*(phi / M_PI + 1.0), theta / M_PI);
	}

	float sRGB2LinF(float inF) {
		if (inF < 0.04045f) {
			return inF / 12.92f;
		}
		else {
			return std::pow((inF + 0.055f) / (1.055f), 2.4f);
		}
	}

	float lin2sRGBF(float inF) {

		if (inF < 0.0031308f) {
			return std::max(0.0f, std::min(1.0f, 12.92f*inF));
		}
		else {
			return std::max(0.0f, std::min(1.0f, 1.055f*std::pow(inF, 1.0f / 2.4f) - 0.055f));
		}

	}

	void sRGB2Lin(sibr::ImageRGB32F& img) {
#pragma omp parallel for
		for (int j = 0; j < int(img.h()); j++) {
			for (int i = 0; i < int(img.w()); i++) {
				for (int c = 0; c < 3; c++) {
					img(i, j)[c] = sRGB2LinF(img(i, j)[c]);
				}
			}
		}
	}

	void lin2sRGB(sibr::ImageRGB32F& img) {
#pragma omp parallel for
		for (int j = 0; j < int(img.h()); j++) {
			for (int i = 0; i < int(img.w()); i++) {
				for (int c = 0; c < 3; c++) {
					img(i, j)[c] = lin2sRGBF(img(i, j)[c]);
				}
			}
		}

	}

} // namespace sibr
