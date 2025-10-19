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

#include "../Config.hpp"
#include <core/graphics/Shader.hpp>
#include <core/system/SimpleTimer.hpp>
#include <core/graphics/Input.hpp>
#include <core/graphics/Window.hpp>

#include <string>

#include <imgui/imgui.h>

namespace sibr {

	/**
	* \ingroup sibr_view
	*/
	enum UVspace { ZERO_ONE, MINUS_ONE_ONE, ONE_ZERO };

	/**
	* \ingroup sibr_view
	*/
	template<UVspace space> struct UV : public sibr::Vector2f {

		static UV from(const sibr::Vector2f & v) { return UV(v.x(), v.y()); }

		UV(float u, float v) : Vector2f(u, v) {}

		//explicit UV(const sibr::Vector2f & v) : sibr::Vector2f(v) {}

		template<UVspace otherSpace> operator UV<otherSpace>() const;
	};

	typedef UV<sibr::MINUS_ONE_ONE> UV11;
	typedef UV<sibr::ZERO_ONE> UV01;
	typedef UV<sibr::ONE_ZERO> UV10;

	template<> template<> inline
	UV11::operator UV01() const {
		return UV01(0.5f*x() + 1, 0.5f*y() + 1);
	}

	template<> template<> inline
	UV01::operator UV11() const {
		return UV11(2.0f*x() - 1, 2.0f*y() - 1);
	}

	template<> template<> inline
	UV01::operator UV10() const {
		return UV10(x(), 1.0f - y());
	}

	template<> template<> inline
	UV10::operator UV01() const {
		return UV01(x(), 1.0f - y());
	}

	template<> template<> inline
	UV10::operator UV11() const {
		return UV11(UV01(*this));
	}

	template<> template<> inline
	UV11::operator UV10() const {
		return UV10(UV01(*this));
	}

	/**
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT InterfaceUtilities
	{
	public:
		InterfaceUtilities();

		static const std::string translationScalingVertexShader;
		static const std::string colorAlphaFragmentShader;

		sibr::GLShader baseShader;

		sibr::GLParameter colorGL;
		sibr::GLParameter alphaGL;
		sibr::GLParameter scalingGL;
		sibr::GLParameter translationGL;

		static const std::string multiViewVertexShader;
		static const std::string multiViewFragmentShader;

		static const std::string meshVertexShader;
		static const std::string meshAlphaViewFragmentShader;
		sibr::GLShader meshViewShader;
		sibr::GLParameter mvp;
		sibr::GLParameter colorMeshGL;
		sibr::GLParameter alphaMeshGL;

		sibr::GLShader multiViewShader;

		sibr::GLParameter numImgsGL;
		sibr::GLParameter gridGL;
		sibr::GLParameter multiViewTopLeftGL;
		sibr::GLParameter multiViewBottomRightGL;

		void rectangle(const sibr::Vector3f & color, const UV11 & tl, const UV11 & br, bool fill, float alpha); 
		void rectanglePixels(const sibr::Vector3f & color, const sibr::Vector2f & center, const sibr::Vector2f & diagonal, bool fill, float alpha, const sibr::Vector2f & winSize);
		void circle(const sibr::Vector3f & color, const UV11 & center, float radius, bool fill, float alpha, const sibr::Vector2f & scaling = sibr::Vector3f(1,1), int precision = 50);
		void circlePixels(const sibr::Vector3f & color, const sibr::Vector2f & center, float radius, bool fill, float alpha, const sibr::Vector2f & winSize, int precision = 50);
		void linePixels(const sibr::Vector3f & color, const sibr::Vector2f & ptA, const sibr::Vector2f & ptB, const sibr::Vector2f & winSize);

		void initAllShaders();
		void freeAllShaders();

	private:
		struct GLinitializer {
			GLinitializer() {
				//InterfaceUtilities::initBaseShader();
			}
		};

		void initBaseShader();
		void initMultiViewShader();
		void initMeshViewShader();

		//const static GLinitializer init;
	};

	struct RectangleData
	{
		RectangleData() : center({ 0.5, 0.5 }), diagonal({ 0.5, 0.5 }) {}
		sibr::Vector2f center;
		sibr::Vector2f diagonal;
		sibr::Vector2f br() const { return center + diagonal; }
		sibr::Vector2f tl() const { return center - diagonal; }
	};

	struct DragData
	{
		DragData() : isActive(false) {}
		sibr::Vector2f center;
		sibr::Vector2i position;
		bool isActive;
	};

	struct SelectionData
	{
		SelectionData() : isActive(false) {}
		sibr::Vector2i first;
		sibr::Vector2i second;
		bool isActive;

	};

	template<sibr::Mouse::Code mKey> struct DoubleClick
	{
		DoubleClick() : detection_timing_in_ms(500) {}

		bool detected(const sibr::Input & input, bool should_be_close = true) {
			if (input.mouseButton().isPressed(mKey)) {
				//std::cout << "timer.deltaTimeFromLastTic<sibr::Timer::s>() : " << timer.deltaTimeFromLastTic<>() << std::endl;			
				if (timer.deltaTimeFromLastTic<>() < detection_timing_in_ms && (!should_be_close || (firstPosition - input.mousePosition()).cwiseAbs().maxCoeff() < 10 ) ) {
					return true;
				}
				firstPosition = input.mousePosition();
				timer.tic();
			}
			return false;
		}

		double detection_timing_in_ms;

		sibr::Timer timer;
		sibr::Vector2i firstPosition;
	};

	template <typename T_Type, unsigned int T_NumComp>
	static void show(
		const sibr::Texture2DArray<T_Type, T_NumComp> & texArray,
		sibr::Window * inputWin = nullptr,
		int w = -1,
		int h = -1
	) {
		enum Mode { SLICE, GRID };

		sibr::Window * win = inputWin;
		const bool inChargeOfWindow = (win == nullptr);
		const bool useCustomSize = (w > 0 && h > 0);
		sibr::Vector2i previousSize;

		if (inChargeOfWindow) {
			sibr::Vector2i winSize = (useCustomSize ? sibr::Vector2i(w, h) : sibr::Vector2i (1600, 1200));
			win = new sibr::Window(winSize[0], winSize[1], "showTexArray");
		} else if (useCustomSize) {
			previousSize = win->size();
			win->size(w, h);
		}
		Mode mode = GRID;

		win->makeContextCurrent();

		sibr::InterfaceUtilities utils;
		utils.initAllShaders();

		sibr::Vector2i grid(3, 3), previousGrid;
		sibr::Vector2f TL(0, 0), BR(1, 1);
		int slice = 1;

		bool renderLoop = true;
		while (renderLoop) {
			sibr::Input::poll();
			sibr::Input & input = sibr::Input::global();
			if (input.key().isPressed(sibr::Key::Escape)) {
				renderLoop = false;
				if (inChargeOfWindow) {
					win->close();
				}
			}

			ImGui::Begin("Show setting");
			if (ImGui::RadioButton("Grid", (int*)&mode, 1)) {
				grid = previousGrid;
			}
			ImGui::SameLine();
			ImGui::RadioButton("Slice", (int*)&mode, 0);

			if (mode == GRID) {
				ImGui::SliderInt("GridX", &grid[0], 1, texArray.depth());
				ImGui::SliderInt("GridY", &grid[1], 1, texArray.depth());
				previousGrid = grid;
				TL = { 0, 0 }, BR = { 1,1 };
			} else if (mode == SLICE) {
				grid = sibr::Vector2i(1, 1); 
				ImGui::SliderInt("Slice", &slice, 1, texArray.depth());
				TL[1] = -slice + 2;
				BR[1] = -slice + 1;
				//ImGui::SliderFloat("L", &TL[0], -3, 3);
				//ImGui::SliderFloat("T", &TL[1], -3, 3);
				//ImGui::SliderFloat("R", &BR[0], -3, 3);
				//ImGui::SliderFloat("B", &BR[1], -3, 3);		
			}

			ImGui::End();

			const auto & viewport = win->viewport();
			viewport.bind();
			viewport.clear(sibr::Vector3f(0.7, 0.7, 0.7));

			utils.multiViewShader.begin();

			utils.numImgsGL.set((int)texArray.depth() - 1);
			sibr::Vector2f gridF = grid.cast<float>();
			utils.gridGL.set(gridF);

			utils.multiViewTopLeftGL.set(TL);
			utils.multiViewBottomRightGL.set(BR);		

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D_ARRAY, texArray.handle());
			sibr::RenderUtility::renderScreenQuad();

			utils.multiViewShader.end();

			win->swapBuffer();
		}

		if (inChargeOfWindow) {
			delete win;
		} else if(useCustomSize) {
			win->size(previousSize[0], previousSize[1]);
		}
	}

	//class SIBR_VIEW_EXPORT Draw {
	//public:
	//	static void rectangle(const sibr::Vector3f & color, const UV11 & tl, const UV11 & br, bool fill, float alpha);
	//	static void circle(const sibr::Vector3f & color, const UV11 & center, float radius, bool fill, float alpha, int precision = 50);
	//};

} // namespace sibr
