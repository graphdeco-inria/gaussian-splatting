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

#include "core/view/Config.hpp"
#include "core/graphics/Utils.hpp"
#include "core/graphics/Camera.hpp"
#include "core/graphics/Image.hpp"
#include "core/assets/InputCamera.hpp"
#include <map>

namespace sibr{

	/** Provide basic IBR utilities. 
	 \ingroup sibr_view
	 */
	class SIBR_VIEW_EXPORT IBRBasicUtils
	{

	public:

		/** Select cameras for a given viewpoint.
		\param cams cameras to select from
		\param eye novel viewpoint
		\param count number of cameras to select
		\return a list of selected camera indices.
		\warning The number of cameras selected might be lower than count
		\sa selectCamerasAngleWeight
		*/
		static std::vector<uint> selectCameras(const std::vector<InputCamera::Ptr>& cams, const sibr::Camera& eye, uint count);

		/** Select cameras based on distance to a given viewpoint. Cameras with an orientation that is more than 45� off compared to the reference are ignored.
		\param cams cameras to select from
		\param eye novel viewpoint
		\param count number of cameras to select
		\return a list of selected camera indices.
		\warning The number of cameras selected might be lower than count
		*/
		static std::vector<uint> selectCamerasSimpleDist(const std::vector<InputCamera::Ptr>& cams, const sibr::Camera& eye, uint count, const bool& distOnly = false);

	
		/** Select cameras based on distance and orientation to a given viewpoint. Cameras with an orientation that is more than 45� off compared to the reference are ignored.
		\param cams cameras to select from
		\param eye novel viewpoint
		\param count number of cameras to select
		\return a list of selected camera indices.
		\warning The number of cameras selected might be lower than count
		*/
		static std::vector<uint> selectCamerasAngleWeight(const std::vector<InputCamera::Ptr>& cams, const sibr::Camera& eye, uint count);

	};

}
