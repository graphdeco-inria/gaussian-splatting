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



#include "core/graphics/Image.hpp"
#include <fstream>

namespace sibr
{
	namespace opencv
	{


		float			imageTypeCVRange(int cvDepth)
		{
			// keep in mind
			//enum { CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3, CV_32S=4, CV_32F=5, CV_64F=6 };
			static float ranges[] = {
				imageTypeRange<uint8>(),
				imageTypeRange<int8>(),
				imageTypeRange<uint16>(),
				imageTypeRange<int16>(),
				imageTypeRange<int32>(),
				imageTypeRange<float>(),
				imageTypeRange<double>()
			};
			return ranges[cvDepth];
		}

		void			convertBGR2RGB(cv::Mat& img)
		{
			switch (img.channels())
			{
			case 3:
				cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				break;
			case 4:
				cv::cvtColor(img, img,  cv::COLOR_BGRA2RGBA);
				break;
			default:
				break;
			}
		}

		void			convertRGB2BGR(cv::Mat& img)
		{
			switch (img.channels())
			{
			case 3:
				cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
				break;
			case 4:
				cv::cvtColor(img, img, cv::COLOR_RGBA2BGRA);
				break;
			default:
				break;
			}
		}

	} // namespace opencv

	sibr::ImageRGBA convertL32FtoRGBA(const sibr::ImageL32F & imgF)
	{
		sibr::ImageRGBA out(imgF.w(), imgF.h());
		for (uint y = 0; y < out.h(); ++y) {
			for (uint x = 0; x < out.w(); ++x) {
				unsigned char const * p = reinterpret_cast<unsigned char const *>(&imgF(x, y).x());
				for (std::size_t i = 0; i != sizeof(float); ++i) {
					out(x, y)[i] = p[i];
				}
			}
		}
		return out;
	}

	sibr::ImageL32F convertRGBAtoL32F(const sibr::ImageRGBA & imgRGBA)
	{
		sibr::ImageL32F out(imgRGBA.w(), imgRGBA.h());
#pragma omp parallel for
		for (int y = 0; y < int(out.h()); ++y) {
			for (uint x = 0; x < out.w(); ++x) {
				unsigned char * p = reinterpret_cast<unsigned char *>(&out(x, y).x());
				for (std::size_t i = 0; i != sizeof(float); ++i) {
					p[i] = imgRGBA(x, y)[i];
				}
			}
		}
		return out;
	}

	sibr::ImageRGBA convertRGB32FtoRGBA(const sibr::ImageRGB32F & imgF)
	{
		sibr::ImageRGBA out(3*imgF.w(), imgF.h());
#pragma omp parallel for
		for (int y = 0; y < int(imgF.h()); ++y) {
			for (uint x = 0; x < imgF.w(); ++x) {
				for (int k = 0; k < 3; k++) {
					unsigned char const * p = reinterpret_cast<unsigned char const *>(&imgF(x, y)[k]);
					for (std::size_t i = 0; i != sizeof(float); ++i) {
						out(k*imgF.w() + x, y)[i] = p[i];
					}
				}
			}
		}
		return out;
	}

	 sibr::ImageRGB32F convertRGBAtoRGB32F(const sibr::ImageRGBA& imgRGBA)
	{
		sibr::ImageRGB32F out(imgRGBA.w() / 3, imgRGBA.h());
#pragma omp parallel for
		for (int y = 0; y < int(out.h()); ++y) {
			for (uint x = 0; x < out.w(); ++x) {
				for (int k = 0; k < 3; k++) {
					unsigned char* p = reinterpret_cast<unsigned char*>(&out(x, y)[k]);
					for (std::size_t i = 0; i != sizeof(float); ++i) {
						p[i] = imgRGBA(k * out.w() + x, y)[i];
					}
				}
			}
		}
		return out;
	}

	 sibr::ImageRGBA convertNormalMapToSphericalHalf(const sibr::ImageRGB32F & imgF)
	{
		uint phi_uint; 
		uint theta_uint;
		unsigned char * phi_ptr = reinterpret_cast<unsigned char *>(&phi_uint);
		unsigned char * theta_ptr = reinterpret_cast<unsigned char *>(&theta_uint);

		ImageRGBA out(imgF.w(),imgF.h());

		for (uint i = 0; i < out.h(); ++i) {
			for (uint j = 0; j < out.w(); ++j) {		
				const double phi = std::acos((double)imgF(j, i)[2]);
				const double theta = std::atan2((double)imgF(j, i)[1], (double)imgF(j, i)[0]);
				phi_uint = (uint)((phi / M_PI) * (1 << 16));
				theta_uint = (uint)((0.5*(theta / M_PI + 1.0)) * (1 << 16));

				unsigned char * out_ptr = reinterpret_cast<unsigned char *>(&out(j, i)[0]);
				out_ptr[0] = phi_ptr[0];
				out_ptr[1] = phi_ptr[1];
				out_ptr[2] = theta_ptr[0];
				out_ptr[3] = theta_ptr[1];
			}
		}

		return out;
	}

	 sibr::ImageRGB32F convertSphericalHalfToNormalMap(const sibr::ImageRGBA & imgRGBA)
	{	
		uint phi_uint;
		uint theta_uint; 
		unsigned char * phi_ptr = reinterpret_cast<unsigned char *>(&phi_uint);
		unsigned char * theta_ptr = reinterpret_cast<unsigned char *>(&theta_uint);

		ImageRGB32F out(imgRGBA.w(), imgRGBA.h());

		for (uint i = 0; i < out.h(); ++i) {
			for (uint j = 0; j < out.w(); ++j) {	
				unsigned char const * out_ptr = reinterpret_cast<unsigned char const *>(&imgRGBA(j, i)[0]);
				phi_ptr[0] = out_ptr[0];
				phi_ptr[1] = out_ptr[1];
				theta_ptr[2] = out_ptr[0];
				theta_ptr[3] = out_ptr[1];

				float theta = ((float)phi_uint*2.0f / (1 << 16) - 1.0f)*float(M_PI);
				float phi = ((float)theta_uint /(1 << 16))*float(M_PI);
				float sin_t = std::sin(theta);
				float cos_t = std::cos(theta);
				float sin_p = std::sin(phi);
				float cos_p = std::cos(phi);
				out(j, i) = sibr::Vector3f(sin_t*cos_p, sin_t*sin_p, cos_t);
			}
		}

		return out;
	}

	Image<unsigned char, 3> coloredClass(const Image<unsigned char, 1>::Ptr imClass) { 
		
		const int color_list[25][3] = {
			{255, 179, 0},{128, 62, 117},{166, 189, 215} ,{193, 0, 32},{0,128,255},{0, 125, 52},
			{246, 118, 142},{0, 83, 138},{255, 122, 92} ,{0, 255, 0},{255, 142, 0},{179, 40, 81},
			{244, 200, 0},{127, 24, 13},{147, 170, 0} ,{89, 51, 21},{241, 58, 19},{35, 44, 22},
			{83, 55, 122},{255,0,128},{128,255,0} ,{128,0,255},{206, 162, 98},{128,128,128},{255,255,255}
		};

		std::vector<Vector3ub> colors(256);
		for (int i = 0; i < 255; i++) {
			colors[i] = Vector3ub(color_list[i % 25][0], color_list[i % 25][1], color_list[i % 25][2]);
		}
		colors[255] = Vector3ub(0, 0, 0);
		Image<unsigned char, 3> imClassColor(imClass->w(), imClass->h());

		for (unsigned int i = 0; i < imClass->w(); i++) {
			for (unsigned int j = 0; j < imClass->h(); j++) {
				imClassColor(i, j) = colors[imClass(i, j).x() % 256];
			}
		}
		return imClassColor;
	}

	Image<unsigned char, 3> coloredClass(const Image<int, 1>::Ptr imClass) { 
																					   
		const int color_list[25][3] = {
			{ 255, 179, 0 },{ 128, 62, 117 },{ 166, 189, 215 } ,{ 193, 0, 32 },{ 0,128,255 },{ 0, 125, 52 },
			{ 246, 118, 142 },{ 0, 83, 138 },{ 255, 122, 92 } ,{ 0, 255, 0 },{ 255, 142, 0 },{ 179, 40, 81 },
			{ 244, 200, 0 },{ 127, 24, 13 },{ 147, 170, 0 } ,{ 89, 51, 21 },{ 241, 58, 19 },{ 35, 44, 22 },
			{ 83, 55, 122 },{ 255,0,128 },{ 128,255,0 } ,{ 128,0,255 },{ 206, 162, 98 },{ 128,128,128 },{ 255,255,255 }
		};

		std::vector<Vector3ub> colors(256);
		for (int i = 0; i < 255; i++) {
			colors[i] = Vector3ub(color_list[i % 25][0], color_list[i % 25][1], color_list[i % 25][2]);
		}
		colors[255] = Vector3ub(0, 0, 0);
		Image<unsigned char, 3> imClassColor(imClass->w(), imClass->h());

		for (unsigned int j = 0; j < imClass->h(); j++) {
			for (unsigned int i = 0; i < imClass->w(); i++) {

				Vector3ub color;
				if (imClass(i, j).x() < 0)
					color = colors[255];
				else
					color = colors[imClass(i, j).x() % 256];

				imClassColor(i, j) = color;
			}
		}
		return imClassColor;
	}

	void showFloat(const Image<float, 1> & im, bool logScale, double min, double max) {
		Image<unsigned char, 1> imIntensity(im.w(), im.h());
		Image<unsigned char, 3> imColor(im.w(), im.h());

		if (min == -DBL_MAX && max == DBL_MAX) {
			cv::minMaxLoc(im.toOpenCV(), &min, &max);
		}
		else if (min == -DBL_MAX) {
			double drop;
			cv::minMaxLoc(im.toOpenCV(), &min, &drop);
		}
		else if (max == DBL_MAX) {
			double drop;
			cv::minMaxLoc(im.toOpenCV(), &drop, &max);
		}

		if (logScale) {
			min = log(min);
			max = log(max);
		}

		for (unsigned int j = 0; j < im.h(); j++) {
			for (unsigned int i = 0; i < im.w(); i++) {
				if (logScale)
					imIntensity(i, j).x() = static_cast<unsigned char>(std::max(0.0, std::min((log(im(i, j).x()) - min) * 255 / (max - min), 255.0)));
				else
					imIntensity(i, j).x() = static_cast<unsigned char>(std::max(0.0, std::min((im(i, j).x() - min) * 255 / (max - min), 255.0)));
			}
		}
		cv::Mat colorMat;
		cv::applyColorMap(imIntensity.toOpenCV(), colorMat, cv::COLORMAP_PARULA);
		imColor.fromOpenCVBGR(colorMat);
		show(imColor);
	}

	cv::Mat duplicate3(cv::Mat c) {
		cv::Mat out;
		cv::Mat in[] = { c, c, c };
		cv::merge(in, 3, out);
		return out;
	}

	// Adopted from http://www.64lines.com/jpeg-width-height. Gets the JPEG size from the file stream passed
	// to the function, file reference: http://www.obrador.com/essentialjpeg/headerinfo.htm
	sibr::Vector2i IImage::get_jpeg_size(std::ifstream& file)
	{
		// Check for valid JPG
		if (file.get() != 0xFF || file.get() != 0xD8)
			return Eigen::Vector2i(-1, -1);
		file.get(); file.get(); // Skip the rest of JPG identifier.

		std::streampos block_length = static_cast<std::streampos>(file.get() * 256 + file.get() - 2);
		for (;;)
		{
			// Skip the first block since it doesn't contain the resolution
			file.seekg(file.tellg() + block_length);

			// Check if we are at the start of another block
			if (!file.good() || file.get() != 0xFF)
				break;

			// If the block is not the "Start of frame", skip to the next block
			if (file.get() != 0xC0)
			{
				block_length = static_cast<std::streampos>(file.get() * 256 + file.get() - 2);
				continue;
			}

			// Found the appropriate block. Extract the dimensions.
			for (int i = 0; i < 3; ++i) file.get();

			int height = file.get() * 256 + file.get();
			int width = file.get() * 256 + file.get();
			return sibr::Vector2i(width, height);
		}
		return sibr::Vector2i(-1, -1);
	}

	// Adopted from http://stackoverflow.com/questions/22638755/image-dimensions-without-loading
	// kudos to Lukas (http://stackoverflow.com/users/643315/lukas).
	sibr::Vector2i IImage::imageResolution(const std::string& file_path)
	{
		enum ValidFormats {
			PNG = 0,
			BMP,
			TGA,
			JPG,
			JPEG,
			VALID_COUNT
		};

		std::string valid_extensions[] = {
			"png",
			"bmp",
			"tga",
			"jpg",
			"jpeg"
		};

		std::string extension = sibr::to_lower(sibr::getExtension(file_path));
		
		int extension_id = 0;
		while (extension_id < VALID_COUNT &&
			extension != valid_extensions[extension_id])
			extension_id++;

		if (extension_id == VALID_COUNT)
			return Eigen::Vector2i(-1, -1);

		std::ifstream file(file_path, std::ios::binary);
		if (!file.good())
			return Eigen::Vector2i(-1, -1);

		uint32_t temp = 0;
		int32_t  width = -1;
		int32_t  height = -1;
		switch (extension_id)
		{
		case PNG:
			file.seekg(16);
			file.read(reinterpret_cast<char*>(&width), 4);
			file.read(reinterpret_cast<char*>(&height), 4);
			width = sibr::ByteStream::ntohl(width);
			height = sibr::ByteStream::ntohl(height);
			break;
		case BMP:
			file.seekg(14);
			file.read(reinterpret_cast<char*>(&temp), 4);
			if (temp == 40) // Windows Format
			{
				file.read(reinterpret_cast<char*>(&width), 4);
				file.read(reinterpret_cast<char*>(&height), 4);
			}
			else if (temp == 20) // MAC Format
			{
				file.read(reinterpret_cast<char*>(&width), 2);
				file.read(reinterpret_cast<char*>(&height), 2);
			}
			break;
		case TGA:
			file.seekg(12);
			file.read(reinterpret_cast<char*>(&width), 2);
			file.read(reinterpret_cast<char*>(&height), 2);
			break;
		case JPG:
		case JPEG:
			return get_jpeg_size(file);
			break;
		}
		return sibr::Vector2i(width, height);
	}

} // namespace sibr
