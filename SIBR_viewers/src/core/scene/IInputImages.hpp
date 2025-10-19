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

#include "core/scene/Config.hpp"
#include "core/scene/IParseData.hpp"

namespace sibr
{
	/** 
	\ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT IInputImages {
		SIBR_DISALLOW_COPY(IInputImages);
	public:


		typedef std::shared_ptr<IInputImages>				Ptr;

		virtual void										loadFromData(const IParseData::Ptr & data) = 0;
		virtual void										loadFromExisting(const std::vector<sibr::ImageRGB::Ptr> & imgs) = 0;
		virtual void										loadFromExisting(const std::vector<sibr::ImageRGB> & imgs) = 0;
		virtual void										loadFromPath(const IParseData::Ptr & data, const std::string & prefix, const std::string & postfix) = 0;

		// Alpha blend and modify input images -- for fences
		virtual void										alphaBlendInputImages(const std::vector<sibr::ImageRGB>& back, std::vector<sibr::ImageRGB>& alphas) = 0;

		virtual const std::vector<sibr::ImageRGB::Ptr>&		inputImages(void) const = 0;
		virtual const	sibr::ImageRGB& 						image(uint i) = 0;

	protected:
		IInputImages() {};

	};

}
