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


#include "IBRBasicUtils.hpp"

namespace sibr {
	std::vector<uint> IBRBasicUtils::selectCameras(const std::vector<InputCamera::Ptr>& cams, const Camera & eye, uint count)
	{
		// Select one method
		return selectCamerasAngleWeight(cams, eye, count);
		//return selectCamerasSimpleDist(cams, eye, count);
	}

	std::vector<uint> IBRBasicUtils::selectCamerasSimpleDist(const std::vector<InputCamera::Ptr>& cams, const sibr::Camera & eye, uint count, const bool& distOnly)
	{
		std::vector<uint> warped_img_id;
		std::multimap<float, uint> dist;                 // distance wise closest input cameras

		for (uint i = 0; i < cams.size(); ++i)
		{
			if (cams.at(i)->isActive())
			{
				float d = sibr::distance(cams[i]->position(), eye.position());
				float a = sibr::dot(cams[i]->dir(), eye.dir());
				if (distOnly) {
					dist.insert(std::make_pair(d, i));
				}
				else if (a > 0.707) {					// cameras with 45 degrees
					dist.insert(std::make_pair(d, i));	// sort distances in increasing order
				}
			}
		}

		std::multimap<float, uint>::const_iterator	d_it(dist.begin());
		for (uint i = 0; d_it != dist.end() && i < count; ++d_it, ++i)
			warped_img_id.push_back(d_it->second);

		SIBR_ASSERT(warped_img_id.size() <= count);

		return warped_img_id;
	}

	std::vector<uint> IBRBasicUtils::selectCamerasAngleWeight(const std::vector<InputCamera::Ptr>& cams, const sibr::Camera & eye, uint count)
	{
		const Vector3f& position = eye.position();
		const Quaternionf& rotation = eye.rotation();
		float angleWeight = 0.3f;

		float maxdist = 0.f;
		std::vector<float>	sqrDists(cams.size(), 0.f);

		for (uint i = 0; i < cams.size(); ++i)
		{
			if (cams.at(i)->isActive())
			{
				float sqrDist = (cams[i]->position() - position).squaredNorm();
				sqrDists[i] = sqrDist;
				maxdist = std::max(sqrDist, maxdist);
			}
		}

		std::multimap<float, uint>	factors;
		for (uint i = 0; i < cams.size(); ++i)
		{
			if (cams.at(i)->isActive())
			{
				float a = sibr::dot(cams[i]->dir(), eye.dir());
				if (a > 0.707)							// cameras with 45 degrees
				{
					const float midAngle = 4.71239f; // = 270 degree
					float sqrDist = sqrDists[i];
					float currNormalDist = inverseLerp(0.f, maxdist, sqrDist);
					float currNormalAngle = inverseLerp(0.f, midAngle, angleRadian(rotation, cams[i]->rotation()));
					float factor = currNormalDist*(1.f - angleWeight) + currNormalAngle*angleWeight;

					factors.insert(std::make_pair(factor, i));	// sort distances in increasing order
				}
			}
		}

		std::vector<uint> warped_img_id;
		std::multimap<float, uint>::const_iterator	d_it(factors.begin());
		for (uint i = 0; d_it != factors.end() && i < count; ++d_it, ++i)
			warped_img_id.push_back(d_it->second);

		SIBR_ASSERT(warped_img_id.size() <= count);

		return warped_img_id;
	}
}
