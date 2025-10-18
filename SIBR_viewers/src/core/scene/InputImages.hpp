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

#include "core/scene/IInputImages.hpp"
#include "core/scene/Config.hpp"

namespace sibr
{
	/** 
	\ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT InputImages : public IInputImages {
		SIBR_DISALLOW_COPY(InputImages);
	public:


		typedef std::shared_ptr<InputImages>				Ptr;

		InputImages(){};
		void												loadFromData(const IParseData::Ptr & data) override;
		virtual void										loadFromExisting(const std::vector<sibr::ImageRGB::Ptr> & imgs) override;
		void												loadFromExisting(const std::vector<sibr::ImageRGB> & imgs) override;
		void												loadFromPath(const IParseData::Ptr & data, const std::string & prefix, const std::string & postfix) override;

		// Alpha blend and modify input images -- for fences
		void												alphaBlendInputImages(const std::vector<sibr::ImageRGB>& back, std::vector<sibr::ImageRGB>& alphas) override;

		const std::vector<sibr::ImageRGB::Ptr>&				inputImages(void) const override;
		const sibr::ImageRGB& 								image(uint i) override	{	return *_inputImages[i]; }

		~InputImages(){};

	protected:

		std::vector<sibr::ImageRGB::Ptr>							_inputImages;

	};

	inline void InputImages::loadFromExisting(const std::vector<sibr::ImageRGB::Ptr>& imgs)
	{
		_inputImages = imgs;
	}

	inline const std::vector<sibr::ImageRGB::Ptr>& InputImages::inputImages(void) const {
		return _inputImages;
	}

}
