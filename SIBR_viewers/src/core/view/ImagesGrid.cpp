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


#include "ImagesGrid.hpp"

#include <imgui/imgui.h>

#define GUI_TEXT(txt) { std::stringstream sss; sss << txt << std::endl;  ImGui::Text(sss.str().c_str()); }

namespace sibr
{
	void ImagesGrid::onUpdate(Input & input, const Viewport & vp)
	{
		const Vector2f size = vp.finalSize();

		if (current_level_tex) {
			imSizePixels = { current_level_tex->w(), current_level_tex->h() };
			imSizePixels = imSizePixels.cwiseQuotient(pow(2.0, current_lod)*Vector2f(1, 1)).unaryExpr([](float f) { return std::floor(f); });

			num_imgs = (int)current_layer->imgs_texture_array->depth();
		}

		currentActivePix = pixFromScreenPos(input.mousePosition(), size);
		_vp = vp;

		setupGrid(vp);

		updateZoomBox(input, vp);
		updateZoomScroll(input);
		updateDrag(input, size);

		if (currentActivePix && input.key().isActivated(Key::LeftControl) && input.mouseButton().isReleased(Mouse::Code::Left) ) {
			if (selectionMode == IMAGE_SELECTION) {
				current_layer->image_selection.switchSelection(currentActivePix.im);
			}
			if (selectionMode == PIXEL_SELECTION && !current_layer->flip_texture) {
				current_layer->pixel_selection.switchSelection(currentActivePix);
			}
		}
	
		std::vector<int> all_ims;
		std::iota(all_ims.begin(), all_ims.end(), 0);
		addImagesToHighlight("imBorders", all_ims, { 0,0,0 });

		if (currentActivePix) {
			addPixelsToHighlight("activePix", { currentActivePix }, { 0, 1, 0 }, 0.25f);
		}
		
		const auto & imgs_list = current_layer->image_selection.get();
		if (!imgs_list.empty()) {
			std::vector<int> selected_ims(std::begin(imgs_list), std::end(imgs_list));
			addImagesToHighlight("imSelection", selected_ims, { 0,1,0 }, 0.1f);
		}
		

	}

	void ImagesGrid::onRender(const Viewport & viewport)
	{
		viewport.bind();

		viewport.clear(Vector3f(0.7f, 0.7f, 0.7f));

		if (!current_level_tex) {
			return;
		}

		draw_utils.image_grid(num_imgs, current_level_tex->handle(), grid_adjusted, viewRectangle.tl(), viewRectangle.br(), current_lod, current_layer->flip_texture);

		for (const auto & ims_highlight : images_to_highlight) {
			const auto & imgs = ims_highlight.second;
			for (int im : imgs.data) {
				highlightImage(im, viewport, imgs.color, imgs.alpha);
			}		
		}

		for (const auto & pixels_highlight : pixels_to_highlight) {
			const auto & pix_data = pixels_highlight.second;
			for (const auto  pix : pix_data.data) {
				highlightPixel(pix, viewport, pix_data.color);
			}
		}

		displayZoom(viewport, draw_utils);
	}

	void ImagesGrid::onRender(IRenderTarget & dst)
	{
		dst.bind();

		Viewport vp(0.0f, 0.0f, (float)dst.w(), (float)dst.h());
		onRender(vp);

		dst.unbind();
	}

	void ImagesGrid::onGUI()
	{
		if (ImGui::Begin("grid_gui")) {

			
			optionsGUI();

			listImagesLayerGUI();

			if (currentActivePix) {
				GUI_TEXT("current pix : " << currentActivePix.im << ", " << currentActivePix.pos.transpose());

				Vector4f value = current_layer->imgs_texture_array->readBackPixel(currentActivePix.im, currentActivePix.pos[0], currentActivePix.pos[1], current_lod);
				if (integer_pixel_values) {
					Vector4i value_i = (255 * value).cast<int>();
					GUI_TEXT(" \t value : " << value_i.transpose());
				} else {
					GUI_TEXT(" \t value : " << value.transpose());
				}


			}

			std::stringstream s;
			s << "active images : ";
			for (int im : current_layer->image_selection.get()) {
				s << im << ", ";
			}
			ImGui::Text(s.str().c_str());

		}
		ImGui::End();
	}

	void ImagesGrid::addImagesToHighlight(const std::string & name, const std::vector<int>& imgs, const Vector3f & col, float alpha_fill)
	{
		images_to_highlight[name] = { imgs, col, alpha_fill };
	}

	void ImagesGrid::addPixelsToHighlight(const std::string & name, const std::vector<MVpixel>& pixs, const Vector3f & col, float alpha_fill)
	{
		pixels_to_highlight[name] = { pixs, col, alpha_fill };
	}

	const MVpixel & ImagesGrid::getCurrentPixel()
	{
		return currentActivePix;
	}

	void ImagesGrid::listImagesLayerGUI()
	{
		
		if (ImGui::CollapsingHeader("images_layers")) {
			
			// 0 name | 1 infos | 2 options
			ImGui::Columns(3, "images_layers_list");

			ImGui::Separator();

			ImGui::Text("layer");
			ImGui::NextColumn();

			ImGui::Text("num x w x h");
			ImGui::NextColumn();

			ImGui::Text("options");
			ImGui::NextColumn();

			ImGui::Separator();

			for (auto imgs_it = images_layers.begin(); imgs_it != images_layers.end(); ++imgs_it) {
				if (ImGui::Selectable(imgs_it->name.c_str(), current_layer == imgs_it)) {
					current_layer = imgs_it;
					current_level_tex = current_layer->imgs_texture_array;
				}

				ImGui::NextColumn();

				auto & tex_arr = imgs_it->imgs_texture_array;
				GUI_TEXT(tex_arr->depth() << " x " << tex_arr->w() << " x " << tex_arr->h());
				ImGui::NextColumn();

				ImGui::Checkbox(("flip##" + imgs_it->name).c_str(), &imgs_it->flip_texture);

				ImGui::NextColumn();

				ImGui::Separator();
			}

			ImGui::Columns(1);
			
		}
	}

	void ImagesGrid::optionsGUI()
	{
		if (ImGui::CollapsingHeader("grid_options")) {


			if (ImGui::SliderInt("num per row", &num_per_row, 1, num_imgs)) {
				viewRectangle.center = { 0.5f, 0.5f };
				viewRectangle.diagonal = { 0.5f, 0.5f };
			}
			if (ImGui::SliderInt("pyramid level", &current_lod, 0, 10)) {
				currentActivePix.isDefined = false;
			}

			static const std::vector<const char*> selection_mode_str = { "no selection", "image" ,"pixel" };
			for (int i = 0; i < (int)selection_mode_str.size(); ++i) {
				if (i != 0) {
					ImGui::SameLine();
				}
				if (ImGui::RadioButton(selection_mode_str[i], selectionMode == (SelectionMode)i)) {
					selectionMode = (SelectionMode)i;
				}
			}

			ImGui::Checkbox("integer pixel values", &integer_pixel_values);
		}
	}

	bool ImagesGrid::name_collision(const std::string & name) const
	{
		for (const auto & layer : images_layers) {
			if (layer.name == name) {
				return true;
			}
		}
		return false;
	}

	void ImagesGrid::setupFirstLayer()
	{
		if (images_layers.size() == 1) {
			current_layer = images_layers.begin();
			current_level_tex = current_layer->imgs_texture_array;
		}
	}

	DrawUtilities::DrawUtilities()
	{
		initBaseShader();
		initGridShader();
	}

	void DrawUtilities::baseRendering(const Mesh & mesh, Mesh::RenderMode mode, const Vector3f & color, 
		const Vector2f & translation, const Vector2f & scaling, float alpha, const Viewport & vp)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);

		vp.bind();
		baseShader.begin();

		scalingGL.set(scaling);
		translationGL.set(translation);
		colorGL.set(color);
		alphaGL.set(alpha);

		mesh.render(false, false, mode);

		baseShader.end();

		glDisable(GL_BLEND);
	}

	void DrawUtilities::rectangle(const Vector3f & color, const Vector2f & tl, const Vector2f & br, bool fill, float alpha, const Viewport & vp)
	{
		auto rectangleMesh = std::make_shared<Mesh>();

		rectangleMesh->vertices({
			{ tl.x(), tl.y() , 0 },
			{ tl.x(), br.y() , 0 },
			{ br.x(), br.y() , 0 },
			{ br.x(), tl.y() , 0 }
			});

		if (fill) {
			rectangleMesh->triangles({
				{ 0,1,2 },
				{ 0,2,3 }
				});

			baseRendering(*rectangleMesh, Mesh::FillRenderMode, color, { 0,0 }, { 1,1 }, alpha, vp);
		}

		rectangleMesh->triangles({
			{ 0,0,1 },{ 1,1,2 },{ 2,2,3 },{ 3,3,0 }
			});

		baseRendering(*rectangleMesh, Mesh::LineRenderMode, color, { 0,0 }, { 1,1 }, 1.0f, vp);
	}

	void DrawUtilities::rectanglePixels(const Vector3f & color, const Vector2f & center, const Vector2f & diagonalPixs, bool fill, float alpha, const Viewport & vp)
	{
		Vector2f diagUV = diagonalPixs.cwiseQuotient(vp.finalSize());
		Vector2f tl = center - diagUV;
		Vector2f br = center + diagUV;
		rectangle(color, tl, br, fill, alpha, vp);
	}

	void DrawUtilities::circle(const Vector3f & color, const Vector2f & center, float radius, bool fill, float alpha, const Vector2f & scaling, int precision)
	{
		
		static Mesh::Vertices vertices;
		static Mesh::Triangles circleTriangles, circleFillTriangles;

		int n = precision;
		if (circleFillTriangles.size() != n) {
			n = precision;
			circleTriangles.resize(n);
			circleFillTriangles.resize(n);
			for (int i = 0; i < n; ++i) {
				int next = (i + 1) % n;
				circleTriangles[i] = Vector3u(i, i, next);
				circleFillTriangles[i] = Vector3u(i, next, n);
			}

			vertices.resize(n + 1);
		}

		double base_angle = 2.0*M_PI / (double)n;
		float rho = 0.5f*radius*(float)(1.0 + cos(0.5*base_angle));

		for (int i = 0; i < n; ++i) {
			double angle = i * base_angle;
			vertices[i] = Vector3f((float)cos(angle), (float)sin(angle), (float)0.0);
		}
		vertices[n] = Vector3f(0, 0, 0);

		auto circleMesh = std::make_shared<Mesh>();
		auto circleFilledMesh = std::make_shared<Mesh>();
		circleMesh->vertices(vertices);
		circleFilledMesh->vertices(vertices);
		circleMesh->triangles(circleTriangles);
		circleFilledMesh->triangles(circleFillTriangles);

		if (fill) {
			baseRendering(*circleFilledMesh, Mesh::FillRenderMode, color, { 0,0 }, { radius, radius }, alpha, {});
		}
		baseRendering(*circleMesh, Mesh::LineRenderMode, color, { 0,0 }, { radius, radius }, 1.0f, {});
	}

	void DrawUtilities::circlePixels(const Vector3f & color, const Vector2f & center, float radius, bool fill, float alpha, const Vector2f & winSize, int precision)
	{
		Vector2f centerUV = center.cwiseQuotient(winSize);
		Vector2f scaling = radius * Vector2f(1, 1).cwiseQuotient(winSize);

		circle(color, centerUV, 1.0f, fill, alpha, scaling, precision);
	}

	void DrawUtilities::linePixels(const Vector3f & color, const Vector2f & ptA, const Vector2f & ptB, const Vector2f & winSize)
	{
		Vector2f uvA = ptA.cwiseQuotient(winSize);
		Vector2f uvB = ptB.cwiseQuotient(winSize);

		Mesh line;
		line.vertices({
			{ uvA.x(), uvA.y(), 0.0f },
			{ uvB.x(), uvB.y(), 0.0f }
			});
		line.triangles({
			Vector3u(0,0,1)
			});

		baseRendering(line, Mesh::LineRenderMode, color, { 0,0 }, { 1.0, 1.0 }, 1.0f, {});

	}

	void DrawUtilities::image_grid(int num_imgs, uint texture, const Vector2f & grid, const Vector2f & tl, const Vector2f & br, int lod, bool flip_texture)
	{
		gridShader.begin();

		numImgsGL.set(num_imgs);
		gridGL.set(grid);
		lodGL.set((float)lod);

		gridTopLeftGL.set(tl);
		gridBottomRightGL.set(br);

		flip_textureGL.set(flip_texture);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
		RenderUtility::renderScreenQuad();

		gridShader.end();
	}

	void DrawUtilities::initBaseShader()
	{
		const std::string translationScalingVertexShader =
			"#version 420															\n"
			"layout(location = 0) in vec3 in_vertex;								\n"
			"uniform vec2 translation;												\n"
			"uniform vec2 scaling;													\n"
			"void main(void) {														\n"
			"	gl_Position = vec4(scaling*in_vertex.xy+translation,0.0, 1.0);		\n"
			"}																		\n";

		const std::string colorAlphaFragmentShader =
			"#version 420														\n"
			"uniform vec3 color;												\n"
			"uniform float alpha;												\n"
			"out vec4 out_color;												\n"
			"void main(void) {													\n"
			"		out_color = vec4(color,alpha);								\n"
			"}																	\n";

		baseShader.init("InterfaceUtilitiesBaseShader", translationScalingVertexShader, colorAlphaFragmentShader);
		colorGL.init(baseShader, "color");
		alphaGL.init(baseShader, "alpha");
		scalingGL.init(baseShader, "scaling");
		translationGL.init(baseShader, "translation");
	}

	void DrawUtilities::initGridShader()
	{
		const std::string gridVertexShader =
			"#version 420										\n"
			"layout(location = 0) in vec3 in_vertex;			\n"
			"out vec2 uv_coord;									\n"
			"uniform vec2 zoomTL;								\n"
			"uniform vec2 zoomBR;								\n"
			"void main(void) {									\n"
			"	uv_coord = 0.5*in_vertex.xy + vec2(0.5);		\n"
			"	uv_coord.y = 1.0 - uv_coord.y;					\n"
			"	uv_coord = zoomTL + (zoomBR-zoomTL)*uv_coord;	\n"	
			"	gl_Position = vec4(in_vertex.xy,0.0, 1.0);		\n"
			"}													\n";


		

		const std::string gridFragmentShader =
			"#version 420														\n"
			"layout(binding = 0) uniform sampler2DArray texArray;				\n"
			"uniform int numImgs;												\n"
			"uniform vec2 grid;													\n"
			"uniform float lod;													\n"
			"uniform bool flip_texture;											\n"
			"in vec2 uv_coord;													\n"
			"out vec4 out_color;												\n"
			"void main(void) {													\n"
			"	vec2 uvs = uv_coord;											\n"
			"	uvs =  grid*uvs;												\n"
			"  if( uvs.x < 0 || uvs.y < 0 ) { discard; } 						\n"
			"   vec2 fracs = fract(uvs); 										\n"
			"   vec2 mods = uvs - fracs; 										\n"
			"   int n = int(mods.x + grid.x*mods.y); 							\n"
			" if ( n< 0 || n > numImgs || mods.x >= grid.x || mods.y >= (float(numImgs)/grid.x) ) { discard; } else { \n"
			"	out_color = textureLod(texArray,vec3(fracs.x, flip_texture ? 1.0 -fracs.y : fracs.y,n), lod);	}		\n"
			"	//out_color = vec4(n/64.0,0.0,0.0,1.0); }						\n"
			"	//out_color = vec4(uv_coord.x,uv_coord.y,0.0,1.0);	}			\n"
			"}																	\n";


		gridShader.init("InterfaceUtilitiesMultiViewShader", gridVertexShader, gridFragmentShader);
		gridTopLeftGL.init(gridShader, "zoomTL");
		gridBottomRightGL.init(gridShader, "zoomBR");
		numImgsGL.init(gridShader, "numImgs");
		gridGL.init(gridShader, "grid");
		lodGL.init(gridShader, "lod");
		flip_textureGL.init(gridShader, "flip_texture");
	}

	MVpixel GridMapping::pixFromScreenPos(const Vector2i & pos, const Vector2f & size)
	{
		Vector2f uvScreen = (pos.cast<float>() + 0.5*Vector2f(1, 1)).cwiseQuotient(size);

		Vector2f posF = viewRectangle.tl() + 2.0*viewRectangle.diagonal.cwiseProduct(uvScreen);
		posF = posF.cwiseProduct(grid_adjusted);

		//std::cout << posF.transpose() << " " << numImgs << std::endl;

		if (posF.x() < 0 || posF.y() < 0 || posF.x() >= grid_adjusted.x() /* || posF.y() >= grid.y()  */) {
			return MVpixel();
		}

		int x = (int)std::floor(posF.x());
		int y = (int)std::floor(posF.y());

		int n = x + num_per_row * y;
		if (n >= num_imgs) {
			return MVpixel();
		}

		Vector2f frac = posF - Vector2f(x, y);
		int j = (int)std::floor(frac.x()*imSizePixels.x());
		int i = (int)std::floor(frac.y()*imSizePixels.y());
		return MVpixel(n, Vector2i(j, i));
	}

	Vector2f GridMapping::uvFromMVpixel(const MVpixel & pix, bool use_center)
	{
		Vector2f pos = ((pix.pos.cast<float>() + (use_center ? 0.5 : 0)*Vector2f(1, 1)).cwiseQuotient(imSizePixels) +
			Vector2f(pix.im % num_per_row, pix.im / num_per_row)).cwiseQuotient(grid_adjusted);
		pos = (pos - viewRectangle.tl()).cwiseQuotient(viewRectangle.diagonal) - Vector2f(1, 1);
		pos.y() = -pos.y();
		return pos;
	}

	void GridMapping::updateZoomBox(const Input & input, const sibr::Viewport & vp)
	{
		Vector2f size = vp.finalSize();

		if (input.key().isPressed(Key::Q)) {
			viewRectangle.center = Vector2f(0.5, 0.5);
			viewRectangle.diagonal = Vector2f(0.5, 0.5);
		}

		if (input.mouseButton().isPressed(Mouse::Code::Right) && !input.key().isActivated(Key::LeftControl) && !zoomSelection) {
			zoomSelection.isActive = true;
			zoomSelection.first = input.mousePosition();
		}

		if (zoomSelection) {
			zoomSelection.second = input.mousePosition();

			Viewport aligned_vp = Viewport(0, 0, vp.finalWidth(), vp.finalHeight());
			
			Vector2f currentTL = (zoomSelection.first.cwiseMin(zoomSelection.second)).cast<float>();
			Vector2f currentBR = (zoomSelection.first.cwiseMax(zoomSelection.second)).cast<float>();
			
			const auto clamp = [](Vector2f & v, float w, float h) { v = v.cwiseMax(Vector2f(1, 1)).cwiseMin(Vector2f(w - 2, h - 2)); };
			clamp(currentTL, vp.finalRight(), vp.finalBottom());
			clamp(currentBR, vp.finalRight(), vp.finalBottom());

			if (input.mouseButton().isReleased(Mouse::Code::Right)) {
				zoomSelection.isActive = false;
				if (((currentBR - currentTL).array() > Vector2f(5, 5).array()).all()) {
					Vector2f tlPix = viewRectangle.tl().cwiseProduct(size) + (viewRectangle.br() - viewRectangle.tl()).cwiseProduct(currentTL);
					Vector2f brPix = viewRectangle.tl().cwiseProduct(size) + (viewRectangle.br() - viewRectangle.tl()).cwiseProduct(currentBR);

					Vector2f center = 0.5f*(brPix + tlPix);
					Vector2f diag = 0.5f*(brPix - tlPix);

					float new_ratio = diag.x() / diag.y();
					float target_ratio = size.x() / size.y();
					if (new_ratio > target_ratio) {
						diag.y() = diag.x() / target_ratio;
					} else {
						diag.x() = diag.y() * target_ratio;
					}

					viewRectangle.center = center.cwiseQuotient(size);
					viewRectangle.diagonal = diag.cwiseQuotient(size);
				}
				
			} else if (!input.mouseButton().isActivated(Mouse::Code::Right) && input.isInsideViewport(aligned_vp)) {
				zoomSelection.isActive = false;
			}
		}

	}

	void GridMapping::updateZoomScroll(const Input & input)
	{
		double scroll = input.mouseScroll();
		if (scroll) {
			float ratio = (scroll > 0 ? 0.75f : 1.33f);
			if (input.key().isActivated(Key::LeftControl)) {
				ratio *= ratio;
			}
			viewRectangle.diagonal *= ratio;
		}
	}

	void GridMapping::updateCenter(const Input & input, const Vector2f & size)
	{
	}

	void GridMapping::updateDrag(const Input & input, const Vector2f & size)
	{
		if (input.mouseButton().isPressed(Mouse::Left)) {
			drag.isActive = true;
			drag.position = input.mousePosition();
			drag.center = viewRectangle.center.cast<float>();
		} else if (drag.isActive && input.mouseButton().isReleased(Mouse::Left)) {
			drag.isActive = false;
		}
		if (drag.isActive && input.mouseButton().isActivated(Mouse::Left)) {
			Vector2f translation = 2.0*(input.mousePosition() - drag.position).cast<float>().cwiseQuotient(size).cwiseProduct(viewRectangle.diagonal);
			viewRectangle.center = drag.center - translation;
		}
	}

	void GridMapping::displayZoom(const Viewport & viewport, DrawUtilities & utils)
	{
		if (zoomSelection) {
			Vector2f tl = 2.0*zoomSelection.first.cast<float>().cwiseQuotient(_vp.finalSize()) - Vector2f(1, 1);
			Vector2f br = 2.0*zoomSelection.second.cast<float>().cwiseQuotient(_vp.finalSize()) - Vector2f(1, 1);
			tl.y() = -tl.y();
			br.y() = -br.y();
			utils.rectangle(Vector3f(1, 0, 0), tl, br, false, 0.15f, viewport);
		}
	}

	void GridMapping::highlightPixel(const MVpixel & pix, const Viewport & viewport, const Vector3f & color, const Vector2f & pixScreenSize)
	{
		Vector2f pixTl = uvFromMVpixel(pix);
		Vector2f pixBR = uvFromMVpixel(MVpixel(pix.im, pix.pos + Vector2i(1, 1)));

		viewport.bind();

		if ((pixBR - pixTl).cwiseProduct(viewport.finalSize()).norm() < pixScreenSize.diagonal().norm()) {
			//if pixel size in screen space is too tiny
			draw_utils.rectanglePixels(color, 0.5*(pixTl + pixBR), pixScreenSize, true, 0.15f, viewport);
		} else {
			//otherwise highlight pixel intirely
			draw_utils.rectangle(color, pixTl, pixBR, true, 0.15f, viewport);
		}
	}

	void GridMapping::highlightImage(int im, const sibr::Viewport & viewport, const sibr::Vector3f & color, float alpha)
	{
		Vector2f imTl = uvFromMVpixel(MVpixel(im, { 0, 0 }));
		Vector2f imBR = uvFromMVpixel(MVpixel(im, imSizePixels.cast<int>()));

		draw_utils.rectangle(color, imTl, imBR, alpha != 0 , alpha, viewport);
	}

	void GridMapping::setupGrid(const Viewport & vp)
	{
		float ratio_img = imSizePixels.x() / imSizePixels.y();
		float ratio_vp = vp.finalWidth() / vp.finalHeight();
		grid_adjusted = num_per_row * Vector2f(1, ratio_img / ratio_vp);
	}

}
	
#undef GUI_TEXT