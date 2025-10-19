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


#include <projects/ulr/renderer/ULRV3View.hpp>
#include <core/graphics/GUI.hpp>

sibr::ULRV3View::ULRV3View(const sibr::BasicIBRScene::Ptr & ibrScene, uint render_w, uint render_h) :
	_scene(ibrScene),
	sibr::ViewBase(render_w, render_h)
{
	const uint w = render_w;
	const uint h = render_h;

	//  Renderers.
	_ulrRenderer.reset(new ULRV3Renderer(ibrScene->cameras()->inputCameras(), w, h));
	_poissonRenderer.reset(new PoissonRenderer(w, h));
	_poissonRenderer->enableFix() = true;

	// Rendertargets.
	_poissonRT.reset(new RenderTargetRGBA(w, h, SIBR_CLAMP_UVS));
	_blendRT.reset(new RenderTargetRGBA(w, h, SIBR_CLAMP_UVS));

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto & cams = ibrScene->cameras()->inputCameras();
	for(size_t cid = 0; cid < cams.size(); ++cid) {
		if(cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::ULRV3View::setScene(const sibr::BasicIBRScene::Ptr & newScene) {
	_scene = newScene;
	const uint w = getResolution().x();
	const uint h = getResolution().y();

	std::string shaderName = "ulr_v3";
	if (_weightsMode == VARIANCE_BASED_W) {
		shaderName = "ulr_v3_alt";
	}
	else if (_weightsMode == ULR_FAST) {
		shaderName = "ulr_v3_fast";
	}

	_ulrRenderer.reset(new ULRV3Renderer(newScene->cameras()->inputCameras(), w, h, shaderName));

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto & cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::ULRV3View::setMode(const WeightsMode mode) {
	_weightsMode = mode;
	if (_weightsMode == VARIANCE_BASED_W) {
		_ulrRenderer->setupShaders("ulr/ulr_v3_alt");
	}
	else if (_weightsMode == ULR_FAST) {
		_ulrRenderer->setupShaders("ulr/ulr_v3_fast");
	}
	else {
		_ulrRenderer->setupShaders();
	}
}

void sibr::ULRV3View::onRenderIBR(sibr::IRenderTarget & dst, const sibr::Camera & eye)
{
	// Perform ULR rendering, either directly to the destination RT, or to the intermediate RT when poisson blending is enabled.
	_ulrRenderer->process(
			_scene->proxies()->proxy(),
			eye, 
			_poissonBlend ? *_blendRT : dst,
			_scene->renderTargets()->getInputRGBTextureArrayPtr(),
		_scene->renderTargets()->getInputDepthMapArrayPtr()
		);

	// Perform Poisson blending if enabled and copy to the destination RT.
	if (_poissonBlend) {
		_poissonRenderer->process(_blendRT, _poissonRT);
		blit(*_poissonRT, dst);
	}

}

void sibr::ULRV3View::onUpdate(Input & input)
{
}

void sibr::ULRV3View::onGUI()
{
	const std::string guiName = "ULRV3 Settings (" + name() + ")";
	if (ImGui::Begin(guiName.c_str())) {

		// Poisson settings.
		ImGui::Checkbox("Poisson ", &_poissonBlend); ImGui::SameLine();
		ImGui::Checkbox("Poisson fix", &_poissonRenderer->enableFix());

		// Other settings.
		ImGui::Checkbox("Flip RGB ", &getULRrenderer()->flipRGBs());
		ImGui::PushScaledItemWidth(150);
		ImGui::InputFloat("Epsilon occlusion", &_ulrRenderer->epsilonOcclusion(), 0.001f, 0.01f);

		ImGui::Separator();
		// Rendering mode selection.
		if(ImGui::Combo("Rendering mode", (int*)(&_renderMode), "Standard\0One image\0Leave one out\0Every N\0\0")) {
			updateCameras(true);
		}

		// Get the desired index, make sure it falls in the cameras range.
		if (_renderMode == ONE_CAM || _renderMode == LEAVE_ONE_OUT) {
			const bool changedIndex = ImGui::InputInt("Selected image", &_singleCamId, 1, 10);
			_singleCamId = sibr::clamp(_singleCamId, 0, (int)_scene->cameras()->inputCameras().size() - 1);
			if (changedIndex) {
				// If we are in "leave one out" or "one camera only" mode, we have to update the list of enabled cameras.
				updateCameras(false);
			}
		}

		if (_renderMode == EVERY_N_CAM) {
			if (ImGui::InputInt("Selection step", &_everyNCamStep, 1, 10)) {
				_everyNCamStep = std::max(1, _everyNCamStep);
				updateCameras(false);
			}
		}
		ImGui::Separator();
		// Switch the shaders for ULR rendering.
		if (ImGui::Combo("Weights mode", (int*)(&_weightsMode), "Standard ULR\0Variance based\0Fast ULR\0\0")) {
			setMode(_weightsMode);
		}
		
		ImGui::Checkbox("Occlusion Testing", &_ulrRenderer->occTest());
		ImGui::Checkbox("Debug weights", &_ulrRenderer->showWeights());
		ImGui::Checkbox("Gamma correction", &_ulrRenderer->gammaCorrection());
		ImGui::PopItemWidth();
	}
	ImGui::End();
}

void sibr::ULRV3View::updateCameras(bool allowResetToDefault) {
	// If we are here, the rendering mode or the selected index have changed, we need to update the enabled cameras.
	std::vector<uint> imgs_ulr;
	const auto & cams = _scene->cameras()->inputCameras();

	// Compute the cameras indices based on the new mode.
	if (_renderMode == RenderMode::ONE_CAM) {
		// We only use the given camera (if it is active).
		if (cams[_singleCamId]->isActive()) {
			imgs_ulr = { (uint)_singleCamId };
		} else {
			std::cerr << "The camera is not active, using all cameras." << std::endl;
		}
	} else if (_renderMode == RenderMode::LEAVE_ONE_OUT) {
		// We use all active cameras apart from the one given.
		for (size_t cid = 0; cid < cams.size(); ++cid) {
			if (cid != (size_t)_singleCamId && cams[cid]->isActive()) {
				imgs_ulr.push_back(uint(cid));
			}
		}
	}
	else if (_renderMode == RenderMode::EVERY_N_CAM) {
		// We pick one camera every N
		for (size_t cid = 0; cid < cams.size(); ++cid) {
			if ((cid % _everyNCamStep == 0) && cams[cid]->isActive()) {
				imgs_ulr.push_back(uint(cid));
			}
		}
	} else if(allowResetToDefault){
		// We use all active cameras.
		for (size_t cid = 0; cid < cams.size(); ++cid) {
			if (cams[cid]->isActive()) {
				imgs_ulr.push_back(uint(cid));
			}
		}
	}
	// Only update if there is at least one camera enabled.
	if(!imgs_ulr.empty()) {
		// Update the shader informations in the renderer.
		_ulrRenderer->updateCameras(imgs_ulr);
		// Tell the scene which cameras we are using for debug visualization.
		_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
	}
	
}
