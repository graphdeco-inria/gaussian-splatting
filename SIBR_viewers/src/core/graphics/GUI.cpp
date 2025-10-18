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



#include "core/graphics/Window.hpp"
#include "core/graphics/GUI.hpp"
#include "core/graphics/Mesh.hpp"

// We extend ImGui functionality so we need the internal definitions.
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui/imgui_internal.h>

namespace sibr
{
	
	bool		showImGuiWindow(const std::string& windowTitle, const IRenderTarget& rt, ImGuiWindowFlags flags, Viewport & viewport,  bool invalidTexture,  bool updateLayout, int handle )
	{
		bool isWindowFocused = false;
		// If we are asked to, we need to update the viewport at launch.
		if (updateLayout) {
			ImGui::SetNextWindowPos(ImVec2(viewport.finalLeft(), viewport.finalTop()));
			ImGui::SetNextWindowSize(ImVec2(0, 0));
			ImGui::SetNextWindowContentSize(ImVec2(viewport.finalWidth(), viewport.finalHeight()));
		}

		if (::ImGui::Begin(windowTitle.c_str(), NULL, flags))
		{
			// Get the current cursor position (where your window is)
			ImVec2 pos = /*ImGui::GetItemRectMin() + */::ImGui::GetCursorScreenPos();
			Vector2f offset, size;
			Vector2i availRegionSize(::ImGui::GetContentRegionAvail().x, ::ImGui::GetContentRegionAvail().y);
			
			fitImageToDisplayRegion(viewport.finalSize(), availRegionSize, offset, size);
			
			size = size.cwiseMax( sibr::Vector2f( 1.0f,1.0f) );

				
			pos.x += offset.x();
			pos.y += offset.y();

			
			ImGui::SetCursorPos(ImVec2(offset.x(), ImGui::GetTitleBarHeight()+offset.y()));
			ImGui::InvisibleButton((windowTitle + "--TEXTURE-INVISIBLE_BUTTON").c_str(), ImVec2(size.x(), size.y()));
			if (!invalidTexture) {
				::ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)(rt.handle(handle)),
					pos, ImVec2(pos.x + size.x(), pos.y + size.y()),
					ImVec2(0, 1), ImVec2(1, 0));
			}
			
			isWindowFocused = ImGui::IsWindowFocused();

			viewport = Viewport(pos.x, pos.y, pos.x+size.x(), pos.y+size.y());

			// Hand back the inputs to sibr.
			if (ImGui::IsItemHovered()) {
				ImGui::CaptureKeyboardFromApp(false);
				ImGui::CaptureMouseFromApp(false);
			}
		}
		::ImGui::End();

		return isWindowFocused;
	}

	Mesh::Ptr generateMeshForText(const std::string & text, unsigned int & separationIndex){
		// Technically we don't care if we already are in the middle of a ImGui frame.
		// as long as we clear the draw list. ImGui will detect the empty draw lists and cull them.
		ImGui::PushID(1234567809);
		ImGui::SetNextWindowPos(ImVec2(0,0));
		ImGui::Begin(text.c_str(), nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs);
		ImGui::SetWindowFontScale(ImGui::GetIO().FontGlobalScale);

		ImGui::Text(text.c_str());
		// Get back the draw list.
		ImDrawList * drawlist = ImGui::GetWindowDrawList();
		const int vertCount = drawlist->VtxBuffer.Size;
		const int indexCount = drawlist->IdxBuffer.Size;
		// We generate one mesh from the draw list.
		std::vector<sibr::Vector3f> vertices(vertCount);
		std::vector<sibr::Vector2f> uvs(vertCount);
		std::vector<sibr::Vector3f> colors(vertCount);
		std::vector<sibr::Vector3u> faces(indexCount / 3);

		sibr::Vector3f centroid(0.0f, 0.0f, 0.0f);
		for (int k = 0; k < vertCount; ++k) {
			const auto & vtx = drawlist->VtxBuffer[k];
			vertices[k][0] = (vtx.pos.x)*2.0f;
			vertices[k][1] = -vtx.pos.y*2.0f;
			uvs[k][0] = vtx.uv.x; uvs[k][1] = vtx.uv.y;
			ImVec4 col = ImGui::ColorConvertU32ToFloat4(vtx.col);
			colors[k][0] = col.x; colors[k][1] = col.y;
			colors[k][2] = col.z; vertices[k][2] = col.w;
			centroid += vertices[k];
		}
		for (int k = 0; k < indexCount; k += 3) {
			faces[k / 3][0] = (unsigned int)drawlist->IdxBuffer[k];
			faces[k / 3][1] = (unsigned int)drawlist->IdxBuffer[k + 1];
			faces[k / 3][2] = (unsigned int)drawlist->IdxBuffer[k + 2];
		}
		// Center the mesh?
		centroid /= float(vertices.size());
		for (int k = 0; k < vertices.size(); ++k) {
			vertices[k] -= centroid;
		}
		Mesh::Ptr mesh = std::make_shared<Mesh>();
		mesh->vertices(vertices);
		mesh->colors(colors);
		mesh->texCoords(uvs);
		mesh->triangles(faces);
		// Store the separation idnex between the background and the text foreground.
		separationIndex = drawlist->CmdBuffer[0].ElemCount;
		
		// Finish the window, then clear the draw list.
		ImGui::End();
		ImGui::PopID();
		drawlist->Clear();
		return mesh;
	}


	void 			fitImageToDisplayRegion(const Vector2f & imgSize, const Vector2i & regionSize, Vector2f& offset, Vector2f& size)
	{
		
		Vector2f ratios = imgSize.cwiseQuotient(regionSize.cast<float>());
		if (ratios.x() < ratios.y())
		{
			float aspect = imgSize.x() / imgSize.y();
			size.y() = float(regionSize.y());
			size.x() = size.y() * aspect;
		}
		else
		{
			float aspect = imgSize.y() / imgSize.x();
			size.x() = float(regionSize.x());
			size.y() = size.x() * aspect;
		}
		offset = regionSize.cast<float>() / 2 - size / 2;
	}
	


	sibr::Vector2f ZoomData::topLeft()		const { return center - diagonal; }
	sibr::Vector2f ZoomData::bottomRight()	const { return center + diagonal; }

	sibr::Vector2f ZoomData::uvFromBoxPos(const sibr::Vector2f& pos) const
	{
		return topLeft() + 2.0f*diagonal.cwiseProduct(pos);
	}

	ZoomData ZoomData::scaled(const sibr::Vector2f& size) const 
	{
		ZoomData out;
		out.center = center.cwiseProduct(size);
		out.diagonal = diagonal.cwiseProduct(size);
		return out;
	}

	void ZoomInterraction::updateZoom(const sibr::Vector2f& canvasSize)
	{
		const auto & d = callBackData;
		if (d.ctrlPressed) {
			return;
		}

		sibr::Vector2f posF = zoomData.uvFromBoxPos(d.positionRatio);

		if (d.isHoovered && d.isClickedRight && !zoomData.underMofidication) {
			zoomData.underMofidication = true;
			zoomData.tmpTopLeft = posF;
			zoomData.firstClickPixel = d.mousePos;
		}
		if (d.isHoovered && zoomData.underMofidication) {
			zoomData.tmpBottonRight = posF;
			zoomData.secondClickPixel = d.mousePos;
		}

		if (zoomData.underMofidication) {
			ImGui::GetWindowDrawList()->AddRect(
				ImVec2(zoomData.firstClickPixel[0], zoomData.firstClickPixel[1]),
				ImVec2(zoomData.secondClickPixel[0], zoomData.secondClickPixel[1]),
				IM_COL32(255, 0, 0, 255), 0, 0, 2
			);
		}

		if (d.isReleasedRight && zoomData.underMofidication) {
			zoomData.underMofidication = false;
			if ((zoomData.tmpBottonRight - zoomData.tmpTopLeft).cwiseProduct(canvasSize).cwiseAbs().minCoeff() > 10) {
				zoomData.center = 0.5f*(zoomData.tmpBottonRight + zoomData.tmpTopLeft);
				zoomData.diagonal = 0.5f*(zoomData.tmpBottonRight - zoomData.tmpTopLeft).cwiseAbs();
				auto scaledBox = zoomData.scaled(canvasSize);
				float target_ratio = canvasSize[0] / canvasSize[1];
				float current_ratio = scaledBox.diagonal[0] / scaledBox.diagonal[1];
				if (current_ratio > target_ratio) {
					scaledBox.diagonal.y() = scaledBox.diagonal.x() / target_ratio;
				} else {
					scaledBox.diagonal.x() = scaledBox.diagonal.y() * target_ratio;
				}
				zoomData.diagonal = scaledBox.diagonal.cwiseQuotient(canvasSize);
			}
		}

		if (d.isHoovered && d.scroll != 0) {
			zoomData.diagonal = zoomData.diagonal.cwiseProduct(pow(1.15f, -d.scroll)*sibr::Vector2f(1, 1));
		}

		

		zoomData.diagonal = zoomData.diagonal.cwiseMin(sibr::Vector2f(0.5, 0.5));
		using Box = Eigen::AlignedBox2f;
		using Corner = Box::CornerType;

		Box target(sibr::Vector2f(0, 0), sibr::Vector2f(1, 1));
		Box current(zoomData.topLeft(), zoomData.bottomRight());
		
		if (!target.contains(current)) {
			Box inside = current;
			inside.clamp(target);
			for (int c = 0; c < 4; ++c) {
				Corner cType = (Corner)c;
				if ( (current.corner(cType)-inside.corner(cType)).isZero() ) {			
					Corner opposite = (Corner)(3 - c);
					zoomData.center += (inside.corner(opposite) - current.corner(opposite));
					break;
				}
			}
		}

	}


	void SegmentSelection::update(const CallBackData & callback, const sibr::Vector2i & size, const ZoomData & zoom)
	{
		sibr::Vector2i pos = zoom.scaled(size.cast<float>()).uvFromBoxPos(callback.positionRatio).cast<int>();

		if (callback.isHoovered && callback.isClickedRight && callback.ctrlPressed && (!first || valid)) {
			firstPosScreen = callback.mousePos.cast<int>();
			firstPosIm = pos.cast<int>();
			secondPosScreen = firstPosScreen;
			first = true;
		} else if (callback.isHoovered && first) {
			secondPosScreen = callback.mousePos.cast<int>();
			secondPosIm = pos.cast<int>();

			if (callback.isClickedRight) {
				first = false;
				valid = true;
				computeRasterizedLine();
			}
		}
	}

	void SegmentSelection::computeRasterizedLine()
	{
		if (!valid) {
			return;
		}

		sibr::Vector2i diff = secondPosIm - firstPosIm;
		int l = diff.cwiseAbs().maxCoeff();
		rasterizedLine.resize(l + 1);
		for (int i = 0; i <= l; ++i) {
			rasterizedLine[i] = (firstPosIm.cast<float>() + (i / (float)l)*diff.cast<float>()).cast<int>();
		}
	}

	void DisplayImageGui(
		GLuint texture,
		const sibr::Vector2i & displaySize,
		const sibr::Vector2f& uv0,
		const sibr::Vector2f& uv1 
	) {
		ImGui::Image((void*)(intptr_t)(texture), ImVec2(float(displaySize[0]), float(displaySize[1])), ImVec2(uv0[0], uv0[1]), ImVec2(uv1[0], uv1[1]));
	}

	void ImageWithCallback(
		GLuint texture,
		const sibr::Vector2i & displaySize,
		CallBackData & callbackDataOut,
		const sibr::Vector2f & uv0,
		const sibr::Vector2f & uv1
	) {
		CallBackData & data = callbackDataOut;

		data.itemPos = toSIBR<float>(ImGui::GetCursorScreenPos());
		DisplayImageGui(texture, displaySize, uv0, uv1);

		data.itemSize = toSIBR<float>(ImGui::GetItemRectSize());
		data.isHoovered = ImGui::IsItemHovered();
		data.isClickedLeft = ImGui::IsMouseClicked(0);
		data.isReleasedLeft = ImGui::IsMouseReleased(0);
		data.isClickedRight = ImGui::IsItemClicked(1);
		data.isReleasedRight = ImGui::IsMouseReleased(1);
		data.ctrlPressed = ImGui::GetIO().KeyCtrl;
		data.scroll = ImGui::GetIO().MouseWheel;

		if (data.isHoovered) {
			data.mousePos = toSIBR<float>(ImGui::GetIO().MousePos);
			data.positionRatio = (data.mousePos - data.itemPos).cwiseQuotient(data.itemSize);
		}
	}

	void ImageWithZoom(GLuint texture, const sibr::Vector2i & displaySize, ZoomInterraction & zoom)
	{
		ImageWithCallback(texture, displaySize, zoom.callBackData, zoom.zoomData.topLeft(), zoom.zoomData.bottomRight());
		zoom.updateZoom(displaySize.template cast<float>());
	}

} // namespace sibr


namespace ImGui {

	const float GetTitleBarHeight() { return GetTextLineHeight() + GetStyle().FramePadding.y * 2.0f; }

	void PushScaledItemWidth(float item_width)
	{
		ImGui::PushItemWidth(ImGui::GetIO().FontGlobalScale * item_width);
	}

	bool TabButton(const char * label, bool highlight, const ImVec2 & size)
	{
		if (highlight) {
			ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0, 0.8f, 0.8f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0, 0.6f, 0.6f));
		}
		bool b = ImGui::Button(label, size);
		if (highlight) {
			ImGui::PopStyleColor(2);
		}
		return b;
	}

	void PlotMultiLines(const char* label, std::vector<float*> values, int values_count, const std::vector<ImVec4>& colors, float scale_min, float scale_max, ImVec2 graph_size) {
		// Note: code extracted from ImGui and udpated to display multiple lines on the same graph.
		ImGuiWindow* window = GetCurrentWindow();
		if (window->SkipItems)
			return;

		ImGuiContext& g = *GImGui;
		const ImGuiStyle& style = g.Style;
		// Force the plot type.
		ImGuiPlotType plot_type = ImGuiPlotType_Lines;
		const ImVec2 label_size = CalcTextSize(label, NULL, true);
		if (graph_size.x == 0.0f)
			graph_size.x = CalcItemWidth();
		if (graph_size.y == 0.0f)
			graph_size.y = label_size.y + (style.FramePadding.y * 2);

		const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + ImVec2(graph_size.x, graph_size.y));
		const ImRect inner_bb(frame_bb.Min + style.FramePadding, frame_bb.Max - style.FramePadding);
		const ImRect total_bb(frame_bb.Min, frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0));
		ItemSize(total_bb, style.FramePadding.y);
		if (!ItemAdd(total_bb, 0, &frame_bb))
			return;
		const bool hovered = ItemHoverable(inner_bb, 0);

		// Determine scale from values if not specified
		if (scale_min == FLT_MAX || scale_max == FLT_MAX)
		{
			float v_min = FLT_MAX;
			float v_max = -FLT_MAX;
			for (int j = 0; j < values.size(); ++j) {
				for (int i = 0; i < values_count; i++)
				{
					const float v = values[j][i];
					v_min = ImMin(v_min, v);
					v_max = ImMax(v_max, v);
				}
			}
			if (scale_min == FLT_MAX)
				scale_min = v_min;
			if (scale_max == FLT_MAX)
				scale_max = v_max;
		}

		RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);
		int values_offset = 0;

		if (values_count > 0)
		{
			int res_w = ImMin((int)graph_size.x, values_count) + ((plot_type == ImGuiPlotType_Lines) ? -1 : 0);
			int item_count = values_count + ((plot_type == ImGuiPlotType_Lines) ? -1 : 0);

			// No tooltip for now.

			const float t_step = 1.0f / (float)res_w;
			const float inv_scale = (scale_min == scale_max) ? 0.0f : (1.0f / (scale_max - scale_min));

			for (int vid = 0; vid < values.size(); ++vid) {
				float v0 = values[vid][(0 + values_offset) % values_count];
				float t0 = 0.0f;
				ImVec2 tp0 = ImVec2(t0, 1.0f - ImSaturate((v0 - scale_min) * inv_scale));                       // Point in the normalized space of our target rectangle
				float histogram_zero_line_t = (scale_min * scale_max < 0.0f) ? (-scale_min * inv_scale) : (scale_min < 0.0f ? 0.0f : 1.0f);   // Where does the zero line stands

				const ImU32 col_base = GetColorU32(colors[vid >= colors.size() ? 0 : vid]);
				const ImU32 col_hovered = col_base;

				for (int n = 0; n < res_w; n++)
				{
					const float t1 = t0 + t_step;
					const int v1_idx = (int)(t0 * item_count + 0.5f);
					IM_ASSERT(v1_idx >= 0 && v1_idx < values_count);
					const float v1 = values[vid][(v1_idx + values_offset + 1) % values_count];
					const ImVec2 tp1 = ImVec2(t1, 1.0f - ImSaturate((v1 - scale_min) * inv_scale));

					// NB: Draw calls are merged together by the DrawList system. Still, we should render our batch are lower level to save a bit of CPU.
					ImVec2 pos0 = ImLerp(inner_bb.Min, inner_bb.Max, tp0);
					ImVec2 pos1 = ImLerp(inner_bb.Min, inner_bb.Max, (plot_type == ImGuiPlotType_Lines) ? tp1 : ImVec2(tp1.x, histogram_zero_line_t));
					if (plot_type == ImGuiPlotType_Lines)
					{
						window->DrawList->AddLine(pos0, pos1, col_base);
					}
					else if (plot_type == ImGuiPlotType_Histogram)
					{
						if (pos1.x >= pos0.x + 2.0f)
							pos1.x -= 1.0f;
						window->DrawList->AddRectFilled(pos0, pos1, col_base);
					}

					t0 = t1;
					tp0 = tp1;
				}
			}
		}
	}

}
