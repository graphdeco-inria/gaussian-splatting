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
#ifndef _DISABLE_EXTENDED_ALIGNED_STORAGE
# define _DISABLE_EXTENDED_ALIGNED_STORAGE
#endif
#include "../Config.hpp"
#include <core/system/Vector.hpp>
#include <core/graphics/Shader.hpp>
#include <core/graphics/Mesh.hpp>
#include <core/graphics/Window.hpp>
#include <functional>
#include <core/graphics/Input.hpp>
#include <core/view/FPSCounter.hpp>

namespace sibr {

	class InteractiveCameraHandler;
	class Raycaster;
	class Camera;

	/**
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT MeshRenderer {

		SIBR_CLASS_PTR(MeshRenderer);

		struct MeshData {
			MeshData() : dirty(false), depthTest(true) {}

			std::shared_ptr<sibr::Mesh> mesh;
			std::vector<sibr::Vector3f> points;
			std::vector<sibr::Vector3f> colors;
			bool dirty;
			bool depthTest;
		};

		struct MeshParams {
			MeshParams() : depthTest(true), backFaceCulling(true), color(sibr::Vector3f(0.7f,0.7f,0.7f)) {}
			std::shared_ptr<sibr::Mesh> mesh;
			sibr::Mesh::RenderMode mode;
			sibr::Vector3f color;
			bool depthTest;
			bool backFaceCulling;
		};

	public:
		MeshRenderer();
		void render(const sibr::Camera& viewproj);

		void addMesh(std::shared_ptr<sibr::Mesh> meshPtr, sibr::Mesh::RenderMode mode = sibr::Mesh::FillRenderMode );

		void addLines(const std::vector<sibr::Vector3f> & listPoints, const sibr::Vector3f & color);

		void addPoint(const sibr::Vector3f & point, const sibr::Vector3f & color);
		void addPoints(const std::vector<sibr::Vector3f> & listPoints, const sibr::Vector3f & color);
		void cleanPoints();
		void cleanLines();

		void resetLinesAndPoints();
		void resetMeshes();

		std::vector<MeshParams> & getMeshesParams() { return listMeshes; }

	public:
		std::vector<MeshParams> listMeshes;

		MeshData lines;
		MeshData points;
		sibr::Mesh::Ptr specialPoints;

		static const std::string meshVertexShader;
		static const std::string meshFragmentShader;

		sibr::GLShader				shaderLines;

	private:

		sibr::GLShader				shaderMesh;
		sibr::GLShader				shaderPoints;

		sibr::GLParameter			mvpLines;
		sibr::GLuniform<sibr::Vector3f>	lineColor = sibr::Vector3f(1,0,0);
		sibr::GLParameter			mvpPoints;
		sibr::GLParameter			mvpMesh;
		sibr::GLParameter			forcedColor;

		sibr::GLParameter			light_pos;
		sibr::GLParameter			radiusScreen;

	

		void initShaders();
		void updateMeshPoints(void);
		void updateMeshLines(void);

	};

	/**
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT MeshViewer {
		
		SIBR_CLASS_PTR(MeshViewer);

	public:
		MeshViewer();

		MeshViewer(
			const sibr::Vector2i & screenRes,
			const sibr::Mesh & mesh = sibr::Mesh(),
			bool launchRenderingLoop = false);

    virtual void setMainMesh(
      const sibr::Mesh & mesh,
      sibr::Mesh::RenderMode mode = sibr::Mesh::FillRenderMode,
      bool updateCam = true,
      bool setupRaycaster = true
    );

    virtual void setMainMesh(
      sibr::Window & win,
      const sibr::Mesh & mesh,
      sibr::Mesh::RenderMode mode = sibr::Mesh::FillRenderMode,
      bool updateCam = true,
      bool setupRaycaster = true
    );

    virtual void render(const sibr::Viewport & viewport, const sibr::Camera & eye);
    virtual void render(const sibr::Viewport & viewport);
    virtual void render(const sibr::Camera & eye);
    virtual void render();

    virtual void renderLoop(sibr::Window & window);

		void renderLoop(std::shared_ptr<sibr::Window> window);
		void renderLoop(const std::function<void(MeshViewer*)> & f = [](MeshViewer* m){} , bool customRendering = false, bool doReset = true);

    virtual void reset();

    static void demo();



  public:
    sibr::Input input;
    sibr::FPSCounter fpsCounter;
    std::shared_ptr<sibr::Window>			window;
    std::shared_ptr<MeshRenderer>			renderer;
    std::shared_ptr<sibr::InteractiveCameraHandler>		interactCam;
    std::shared_ptr<sibr::Raycaster>		raycaster;

    bool inChargeOfWindow;
  };

} //namespace sibr

