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


/// \todo TODO: make shorter
#include "Config.hpp"
#include <core/assets/Resources.hpp>
#include <projects/ulr/renderer/ULRV2View.hpp>
#include <core/system/Vector.hpp>
#include <core/graphics/Texture.hpp>
#include <core/graphics/GUI.hpp>
#include <map>

namespace sibr { 
	ULRV2View::~ULRV2View( )
{
	_altMesh.reset();
}

ULRV2View::ULRV2View( const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h ) :
	_scene(ibrScene),
		sibr::ViewBase(render_w, render_h), 
	_renderMode(ULRV2View::RenderMode::NORMAL), _singleCamId(0)
{
	_altMesh.reset();
	_altMesh = nullptr;
	_numDistUlr = 4, _numAnglUlr = 0;
    std::cerr << "[ULR] setting number of images to blend "<< _numDistUlr << " " << _numAnglUlr << std::endl;

	_ulr.reset(new ULRV2Renderer(ibrScene->cameras()->inputCameras(), render_w, render_h, _numDistUlr + _numAnglUlr));
	uint w = render_w;
	uint h = render_h;
	_poissonRT.reset(new RenderTargetRGBA(w, h, SIBR_CLAMP_UVS));
	_blendRT.reset(new RenderTargetRGBA(w, h, SIBR_CLAMP_UVS));
	_poisson.reset(new PoissonRenderer(w,h));
	_poisson->enableFix() = true;
	_inputRTs = ibrScene->renderTargets()->inputImagesRT();

	testAltlULRShader = false;
}

void ULRV2View::onRenderIBR( sibr::IRenderTarget& dst, const sibr::Camera& eye ) {
    // Select subset of input images for ULR
	//std::vector<uint> imgs_ulr = chosen_cameras(eye);
	std::vector<uint> imgs_ulr = chosen_cameras_angdist(eye);
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
	//std::cout << imgs_ulr.size() << " " << std::flush;

	if (_renderMode == RenderMode::ONLY_ONE_CAM) {
		for (auto i : imgs_ulr) {
			//std::cout << i << " ";
		}
		int id_cam = std::max(0, std::min((int)imgs_ulr.size()-1, _singleCamId));
		int cam = imgs_ulr[id_cam];
		imgs_ulr = std::vector<uint>(1, cam);
		//std::cout << " -> ulr debug single cam, id : " << _singleCamId << ", cam : ";
		//for (auto i : imgs_ulr) {
			//std::cout << i << " ";
		//}
		//std::cout << std::endl;
	} else if(_renderMode == RenderMode::LEAVE_ONE_OUT) {
		std::vector<uint> new_imgs_ulr;
		for(const auto & i : imgs_ulr) {
				if(int(i) != _singleCamId) {
					new_imgs_ulr.emplace_back(i);
				}
		}
		imgs_ulr = new_imgs_ulr;
	}

	if (_noPoissonBlend) {
		_ulr->process(
			imgs_ulr,
			eye,
			_scene,
			_altMesh,
			_inputRTs,
			dst);

	}  else {
			_ulr->process(
				/* input -- images chosen */ imgs_ulr, 
				/* input -- camera position */ eye, 
				/* input -- scene */ _scene, 
				/* input -- alt mesh if available */ _altMesh, 
				/* input -- input RTs -- can be RGB or alpha */  _inputRTs,
				/* output */ *_blendRT);

			_poisson->process(
					_blendRT,
					_poissonRT);


			blit(*_poissonRT, dst);
	}
}

void ULRV2View::onUpdate(Input & input)
{
	if (input.key().isReleased(sibr::Key::Tab)) {
		testAltlULRShader = !testAltlULRShader;
		if (testAltlULRShader) {
			_ulr->setupULRshader("ulr_v2_alt");
		} else {
			_ulr->setupULRshader();
		}
		std::cout << "ULR using " << (testAltlULRShader ? "all cams" : "standard ulr") << std::endl;
	}
}

void ULRV2View::onGUI() {
		const std::string guiName = "ULRV2 Settings (" + name() + ")";
		if(ImGui::Begin(guiName.c_str())) {
			
			ImGui::PushScaledItemWidth(80);
			const bool v1_changed = ImGui::InputInt("#Dist", &_numDistUlr, 1, 10);
			ImGui::SameLine();
			const bool v2_changed = ImGui::InputInt("#Angle", &_numAnglUlr, 1, 10);
			ImGui::PopItemWidth();

			if (v1_changed || v2_changed) {
				setNumBlend(_numDistUlr, _numAnglUlr);
			}

			

			ImGui::Checkbox("Disable Poisson", &_noPoissonBlend);
			ImGui::Checkbox("Poisson fix", &_poisson->enableFix());

			ImGui::PushScaledItemWidth(120);
			ImGui::InputFloat("Epsilon occlusion", &_ulr->epsilonOcclusion(), 0.001f, 0.01f);
			ImGui::Combo("Rendering mode", (int*)(&_renderMode), "Standard\0One image\0Leave one out\0\0");
			if (ImGui::InputInt("Selected image", &_singleCamId, 1, 10)) {
				_renderMode = RenderMode::ONLY_ONE_CAM;
			}
			_singleCamId = sibr::clamp(_singleCamId, 0, (int)_scene->cameras()->inputCameras().size() - 1);
			//ImGui::SliderInt("Selected image", &_singleCamId, 0, scene().inputCameras().size() - 1);
			ImGui::PopItemWidth();

		}
		ImGui::End();
}

void ULRV2View::computeVisibilityMap(const sibr::ImageL32F & depthMap, sibr::ImageRGBA & out)
{
	const float threshold_3d = 2.5f;
	const std::vector<sibr::Vector2i> shifts = { { 1,0 },{ 0,1 },{ -1,0 },{ 0,-1 } };

	sibr::ImageL8 edgeMap(depthMap.w(), depthMap.h(), 255);
	for (uint i = 0; i < depthMap.h(); i++) {
		for (uint j = 0; j < depthMap.w(); j++) {
			sibr::Vector2i pos(j, i);
			float currentDepth = depthMap(pos).x();
			for (const auto & shift : shifts) {
				Vector2i npos = pos + shift;
				if (!depthMap.isInRange(npos)) { continue; }
				if (std::abs(depthMap(npos).x() - currentDepth) > threshold_3d) {
					edgeMap(pos).x() = 0;
					break;
				}
			}
		}
	}

	cv::Mat distance(depthMap.h(), depthMap.w(), CV_32FC1);
	cv::distanceTransform(edgeMap.toOpenCVnonConst(), distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);

	sibr::ImageL32F outF;
	outF.fromOpenCV(distance);
	out = sibr::convertL32FtoRGBA(outF);

}

	// -----------------------------------------------------------------------

std::vector<uint> ULRV2View::chosen_cameras(const sibr::Camera& eye) {
    std::vector<uint> imgs_id;
    std::multimap<float,uint> distMap;									// distance wise closest input cameras
	std::multimap<float,uint> dang;									// angular distance from inputs to novel camera
    for (uint i=0; i< _scene->cameras()->inputCameras().size(); i++ ) {
        const sibr::InputCamera& inputCam = *_scene->cameras()->inputCameras()[i];
        if (inputCam.isActive()) {
			// Convert following to Eigen versions
            float dist = sibr::distance(inputCam.position(), eye.position());
            float angle = sibr::dot(inputCam.dir(),eye.dir());
                distMap.insert(std::make_pair(dist,i));					// sort distances in increasing order
				dang.insert(std::make_pair( acos(angle),i));				// sort angles in increasing order
        }
    }
	for (uint i=0; i< _scene->cameras()->inputCameras().size(); i++) {
        const sibr::InputCamera& inputCam = *_scene->cameras()->inputCameras()[i];
        if (inputCam.isActive() && distMap.size() <= (_numDistUlr+_numAnglUlr)/2 ) {
            float dist = sibr::distance(inputCam.position(),eye.position());
            distMap.insert(std::make_pair(dist,i));					// sort distances in increasing order
			}
	}

    std::multimap<float,uint>::const_iterator d_it(distMap.begin());	// select the _numDistUlr closest cameras
	for (int i=0; d_it!=distMap.end() && i<_numDistUlr; d_it++,i++) {
        imgs_id.push_back(d_it->second);
    }

	std::multimap<float,uint>::const_iterator a_it(dang.begin());    // select the NUM_ANG_ULR closest cameras
	for (int i=0; a_it!=dang.end() && i<_numAnglUlr; a_it++,i++) {
        imgs_id.push_back(a_it->second);
    }

	std::sort( imgs_id.begin(), imgs_id.end() );				// Avoid repetitions
	imgs_id.erase( std::unique( imgs_id.begin(), imgs_id.end() ), imgs_id.end() );

	SIBR_ASSERT(imgs_id.size() <= _numDistUlr + _numAnglUlr);
    return imgs_id;
}

std::vector<uint> ULRV2View::chosen_cameras_angdist(const sibr::Camera & eye)
{
	const auto & cams = _scene->cameras()->inputCameras();
	std::vector<uint> out;

	// sort angle / dist combined
	struct camAng
	{
		camAng() {}
		camAng(float a, float d, int i) : ang(a), dist(d), id(i) {}
		float ang, dist;
		int id;
		static bool compare(const camAng & a, const camAng & b) { return a.ang / a.dist > b.ang / b.dist;  }
	};

	int total_size = _numAnglUlr + _numDistUlr;

	std::vector<camAng> allAng;
	for (int id = 0; id < (int)cams.size(); ++id) {
		const auto & cam = *cams[id];
        float angle = sibr::dot(cam.dir(),eye.dir());
		// reject back facing 
		if( angle > 0.001 && cam.isActive()) {
			float dist =  (cam.position() - eye.position()).norm();
			allAng.push_back(camAng(angle, dist, id));		
		}
	}

	std::vector<bool> wasChosen(cams.size(), false);

	std::sort(allAng.begin(), allAng.end(), camAng::compare);
	for (int id = 0; id < std::min((int)allAng.size(), total_size); ++id) {
		out.push_back(allAng[id].id);
		wasChosen[allAng[id].id] = true;
	}

	for (int id = 0; id < (int)cams.size(); ++id) {
		if (!wasChosen[id] && out.size() < total_size && cams[id]->isActive()) {
			out.push_back(id);
		}
	}

	return out;
}

std::vector<uint> ULRV2View::chosen_camerasNew(const sibr::Camera & eye)
{
	const auto & cams = _scene->cameras()->inputCameras();

	struct camDist
	{
		camDist() {}
		camDist(float d, int i) : dist(d), id(i) {}
		float dist;
		int id;
		static bool compare(const camDist & a, const camDist & b) { return a.dist < b.dist;  }
	};

	std::vector<camDist> allDist;
	for (int id = 0; id < (int)cams.size(); ++id) {
		const auto & cam = *cams[id];
		allDist.push_back(camDist((cam.position() - eye.position()).norm(), id));
	}
	std::sort(allDist.begin(), allDist.end(), camDist::compare);
	std::vector<uint> out;
	for (int id = 0; id < std::min((int)cams.size(),(int)_numDistUlr); ++id) {
		out.push_back(allDist[id].id);
	}
	return out;
}

void ULRV2View::setNumBlend(short int dist, short int angle)
{
	// Backup masks.
	auto copyMasks = _ulr->getMasks();

	_numDistUlr = dist, _numAnglUlr = angle;
	std::cerr << "[ULR] setting number of images to blend " << _numDistUlr << " " << _numAnglUlr << std::endl;
	_ulr.reset(new ULRV2Renderer(_scene->cameras()->inputCameras(), _scene->cameras()->inputCameras()[0]->w(), _scene->cameras()->inputCameras()[0]->h(), _numDistUlr + _numAnglUlr));
	_ulr->setMasks(copyMasks);
	
}

void ULRV2View::loadMasks(const sibr::BasicIBRScene::Ptr& ibrScene, int w, int h, const std::string& maskDir, const std::string& preFileName, const std::string& postFileName
) {
	std::string finalMaskDir = (maskDir == "" ? ibrScene->data()->basePathName() + "/masks/" : maskDir);
	std::string finalPostFileName = (postFileName == "" ? "-mask.jpg" : postFileName);
	_ulr->loadMasks(ibrScene, finalMaskDir, preFileName, finalPostFileName, w, h);
}

void ULRV2View::setMasks( const std::vector<RenderTargetLum::Ptr>& masks ) {
		_ulr->setMasks(masks);
}

} /*namespace sibr*/ 
