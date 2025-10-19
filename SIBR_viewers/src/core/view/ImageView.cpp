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



#include "ImageView.hpp"

#include <imgui/imgui.h>

namespace sibr {

	ImageView::ImageView(bool interactiveMode)
	{
		_display.init("Display", sibr::loadFile(
			sibr::getShadersDirectory("core") + "/image_viewer.vert"),
			sibr::loadFile(sibr::getShadersDirectory("core") + "/image_viewer.frag"));
		
		_minVal.init(_display, "minVal");
		_maxVal.init(_display, "maxVal");
		_channels.init(_display, "channels");
		_size.init(_display, "size");
		_pos.init(_display, "pos");
		_scale.init(_display, "scale");
		_correctRatio.init(_display, "correctRatio");
		
		_minVal = { 0.0f, 0.0f, 0.0f, 0.0f };
		_maxVal = {1.0f, 1.0f, 1.0f, 1.0f};
		_showChannels[0] = _showChannels[1] = _showChannels[2] = _showChannels[3] = true;
		_bgColor = {0.25f, 0.25f, 0.25f};
		_pos = {0.0f, 0.0f};
		_scale = 1.0f;
		// When in "fixed" mode, don't respect the aspect ratio, to make sure that the full image is visible to the viewer.
		_correctRatio = interactiveMode;
		_showGUI = interactiveMode;
		_allowInteraction = interactiveMode;
	}

	void ImageView::onUpdate(Input& input, const Viewport & vp) {
		if(!_allowInteraction) {
			return;
		}
		_scale = std::max(_scale - float(input.mouseScroll()) * 0.05f, 0.001f);
		if(input.mouseButton().isActivated(Mouse::Left)) {
			sibr::Vector2f delta = input.mouseDeltaPosition().cast<float>().cwiseQuotient(vp.finalSize());
			delta[1] *= -1.0f;
			_pos = _pos.get() + delta;
		}
	}

	void ImageView::onGUI() {

		if(!_showGUI) {
			return;
		}
		const std::string guiName = name() + " options";
		if(ImGui::Begin(guiName.c_str())) {

			ImGui::Text("Size: %dx%d. Scale: %.2f%%", int(_size.get()[0]), int(_size.get()[1]), 100.0f * _scale);
			
			if (ImGui::Button("Reset view")) {
				_pos = sibr::Vector2f(0.0f, 0.0f);
				_scale = 1.0f;
			}
			ImGui::SameLine();
			ImGui::Checkbox("Correct aspect ratio", &_correctRatio.get());
			
			ImGui::Separator();

			ImGui::Text("Channels"); ImGui::SameLine();
			ImGui::Checkbox("R", &_showChannels[0]); ImGui::SameLine();
			ImGui::Checkbox("G", &_showChannels[1]); ImGui::SameLine();
			ImGui::Checkbox("B", &_showChannels[2]); ImGui::SameLine();
			ImGui::Checkbox("A", &_showChannels[3]);

			ImGui::ColorEdit3("Background", &_bgColor[0]);
			
			ImGui::Separator();
			
			const float dragSpeed = 0.05f;
			bool editBounds = false;
			
			if(_lockChannels) {
				// Only display one value and ensure synchronisation between the RGB components.
				editBounds = ImGui::DragFloat("Min.", &_minVal.get()[0], dragSpeed) || editBounds;
				editBounds = ImGui::DragFloat("Max.", &_maxVal.get()[0], dragSpeed) || editBounds;
			} else {
				editBounds = ImGui::DragFloat4("Min.", &_minVal.get()[0], dragSpeed) || editBounds;
				editBounds = ImGui::DragFloat4("Max.", &_maxVal.get()[0], dragSpeed) || editBounds;
			}
			// Ensure internal state consistency.
			if(editBounds && _lockChannels) {
				_minVal.get()[3] = _minVal.get()[2] = _minVal.get()[1] = _minVal.get()[0];
				_maxVal.get()[3] = _maxVal.get()[2] = _maxVal.get()[1] = _maxVal.get()[0];
			}
			// Ensure ordering.
			if(editBounds) {
				const sibr::Vector4f temp = _minVal;
				_minVal = temp.cwiseMin(_maxVal.get());
				_maxVal = temp.cwiseMax(_maxVal.get());
			}
			
			ImGui::Checkbox("Lock values", &_lockChannels);
			ImGui::SameLine();
			if (ImGui::Button("Reset values")) {
				_minVal = sibr::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
				_maxVal = sibr::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);
			}
		}
		ImGui::End();
	}

	void ImageView::setRenderTarget(const IRenderTarget& rt, uint handle) {
		_tex = nullptr;
		_texHandle = rt.handle(handle);
		_size.get()[0] = float(rt.w()); _size.get()[1] = float(rt.h());
	}

	void ImageView::setTexture(const ITexture2D& tex)
	{
		_tex = nullptr;
		_texHandle = tex.handle();
		_size.get()[0] = float(tex.w()); _size.get()[1] = float(tex.h());
	}

	
	void ImageView::onRender(const Viewport & vpRender){

		vpRender.bind();
		vpRender.clear(_bgColor);
		if (_texHandle == 0) {
			return;
		}

		// Update channels flags.
		_channels.get()[0] = float(_showChannels[0]);
		_channels.get()[1] = float(_showChannels[1]);
		_channels.get()[2] = float(_showChannels[2]);
		_channels.get()[3] = float(_showChannels[3]);

		_display.begin();

		if(_showChannels[3]) {
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glBlendEquation(GL_FUNC_ADD);
		}
		
		_maxVal.send();
		_minVal.send();
		_channels.send();
		_scale.send();
		_pos.send();
		_size.send();
		_correctRatio.send();
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, _texHandle);
		RenderUtility::renderScreenQuad();

		glDisable(GL_BLEND);
		_display.end();

		CHECK_GL_ERROR;
	}

}