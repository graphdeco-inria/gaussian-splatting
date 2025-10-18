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


#include <fstream>
#include <memory>

#include "core/view/FPSCounter.hpp"
#include "core/assets/Resources.hpp"

#include <imgui/imgui.h>
#include "core/graphics/GUI.hpp"
#include "imgui/imgui_internal.h"

#define SIBR_FPS_SMOOTHING 60


namespace sibr
{

	int FPSCounter::_count = 0;

	FPSCounter::FPSCounter(const bool overlayed){
		_frameTimes = std::vector<float>(SIBR_FPS_SMOOTHING, 0.0f);
		_frameIndex = 0;
		_frameTimeSum = 0.0f;
		_lastFrameTime = std::chrono::high_resolution_clock::now();
		_position = sibr::Vector2f(-1, -1);
		if (overlayed) {
			_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
		} else {
			_flags = 0;
		}
		_hidden = false;
		_name = "Metrics##" + std::to_string(_count);
		++_count;
	}

	void FPSCounter::init(const sibr::Vector2f & position){
		_position = position;
	}
	
	void FPSCounter::render(){

		if (_hidden) {
			return;
		}

		if (_position.x() != -1) {
			ImGui::SetNextWindowPos(ImVec2(_position.x(), _position.y()));
			ImGui::SetNextWindowSize(ImVec2(0, ImGui::GetTitleBarHeight()), ImGuiCond_FirstUseEver);
		}
		
		ImGui::SetNextWindowBgAlpha(0.5f);
		if (ImGui::Begin(_name.c_str(), nullptr, _flags))
		{
			ImGui::SetWindowFontScale(1.8);
			const float frameTime = _frameTimeSum / float(SIBR_FPS_SMOOTHING);
			ImGui::Text("%.2f (%.2f ms)", 1.0f/ frameTime, frameTime*1000.0f);
			ImGui::SetWindowFontScale(1);
		}

		ImGui::End();
	}
	
	void FPSCounter::update(float deltaTime){
		_frameTimeSum -= _frameTimes[_frameIndex];
		_frameTimeSum += deltaTime;
		_frameTimes[_frameIndex] = deltaTime;
		_frameIndex = (_frameIndex + 1) % SIBR_FPS_SMOOTHING;
	}
	
	void FPSCounter::update(bool doRender) {
		auto now = std::chrono::high_resolution_clock::now();
		float deltaTime = std::chrono::duration<float>(now - _lastFrameTime).count();
		update(deltaTime);
		if (doRender) {
			render();
		}
		_lastFrameTime = now;
		
	}

} // namespace sibr 
