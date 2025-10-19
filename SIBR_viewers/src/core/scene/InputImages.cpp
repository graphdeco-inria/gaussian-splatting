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


#include "InputImages.hpp"


namespace sibr
{
	void InputImages::loadFromData(const IParseData::Ptr & data)
	{
		//InputImages out;
		_inputImages.resize(data->imgInfos().size());

		if (data->imgInfos().empty() == false)
		{
//			#pragma omp parallel for
			for (int i = 0; i < data->imgInfos().size(); ++i) {
				if (data->activeImages()[i]) {
					_inputImages[i] = std::make_shared<ImageRGB>();
					_inputImages[i]->load(data->imgPath() + "/" + data->imgInfos().at(i).filename, false);
				}
				else {
					_inputImages[i] = std::make_shared<ImageRGB>(16,16, 0);
				}
			}
									
		}
		else
			SIBR_WRG << "cannot load images (ImageListFile is empty. Did you use ImageListFile::load(...) before ?";

		std::cout << std::endl;
		
		return;
	}

	void InputImages::loadFromExisting(const std::vector<sibr::ImageRGB> & imgs)
	{
		_inputImages.resize(imgs.size());
		for (size_t i = 0; i < imgs.size(); ++i) {
			_inputImages[i].reset(new ImageRGB(imgs[i].clone()));
		}
	}

	/// \todo UN-TESTED code!!!!
	void InputImages::loadFromPath(const IParseData::Ptr & data, const std::string & prefix, const std::string & postfix)
	{
		_inputImages.resize(data->imgInfos().size());

		#pragma omp parallel for
		for (int i = 0; i < data->imgInfos().size(); ++i) {
			if (data->activeImages()[i]) {
				std::string imgPath = data->basePathName()+ "/images/" + prefix + sibr::imageIdToString(i) + postfix;
				if (!_inputImages[i]->load(imgPath, false)) {
					SIBR_WRG << "could not load input image : " << imgPath << std::endl;
				}
			}
		}
	}


	void InputImages::alphaBlendInputImages(const std::vector<sibr::ImageRGB>& back, std::vector<sibr::ImageRGB>& alphas)
	{
		for (uint i = 0; i< _inputImages.size(); i++) {
			// check size
			if (_inputImages[i]->w() != alphas[i].w() ||
				_inputImages[i]->h() != alphas[i].h())
				alphas[i] = alphas[i].resized(_inputImages[i]->w(), _inputImages[i]->h());
			for (uint x = 0; x<_inputImages[i]->w(); x++)
				for (uint y = 0; y<_inputImages[i]->h(); y++) {
					ImageRGB::Pixel p = _inputImages[i](x, y);
					ImageRGB::Pixel bp = back[i](x, y);
					ImageRGB::Pixel a = alphas[i](x, y);
					Vector3f alpha(float(a[0] / 255.), float(a[1] / 255.), float(a[2] / 255.));
					float al = alpha[0]; // assume grey for now
					Vector3f val = Vector3f(float(p[0]), float(p[1]), float(p[2]));
					Vector3f bval = Vector3f(float(bp[0]), float(bp[1]), float(bp[2]));
					Vector3f out;
					if (alpha[0] > 0.4) {
						out = (val - bval + al*bval) / al; // foreground solving for a

														   // clamp
#define NZ(x)		((x)<0.f?0.f:x)
#define UP(x)		((x)>255.f?255.f:x)
						out[0] = UP(NZ(out[0]));
						out[1] = UP(NZ(out[1]));
						out[2] = UP(NZ(out[2]));
					}
					else
						out = Vector3f(0, 0, 0);

					_inputImages[i](x, y) = ImageRGB::Pixel((uint)out[0], (uint)out[1], (uint)out[2]);
				}
		}
	}
}
	
