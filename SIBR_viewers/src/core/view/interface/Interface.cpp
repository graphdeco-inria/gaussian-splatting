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


#include "Interface.h"

#include "MeshViewer.h"
#include <core/view/InteractiveCameraHandler.hpp>

#include <imgui/imgui.h>

namespace sibr {

	MultiViewInterface::MultiViewInterface()
	{
		numImgs = 0;
		currentLayer = 0;
		currentScale = 0;
		grid = sibr::Vector2i(4, 4);
		hightligthChanged = false;
		reproMeshMode = sibr::Mesh::FillRenderMode;
		reproMeshBackFace = true;

		imagesViewBase = std::make_shared<MultiViewInterfaceView>(this, MultiViewInterfaceView::ViewType::IMAGES);
		meshViewBase = std::make_shared<MultiViewInterfaceView>(this, MultiViewInterfaceView::ViewType::MESH);
	}

	void MultiViewInterface::displayLoop(sibr::Window & window, std::function<void(MultiViewInterface*)> f)
	{
		
		if (layersData.size() == 0) {
			SIBR_ERR << " cant display interface without image layer added" << std::endl;
		}

		utils.initAllShaders();

		glClearColor(0.8f, 0.8f, 0.8f, 1.0f);

		window.size(window.size().x(), (int)std::ceil(window.size().x() / scalesData[0].imRatio));

		std::cout << " window size : " << window.size().transpose() << std::endl;

		imagesView.viewport = sibr::Viewport(&window.viewport(),0,0,1,1);
		imagesView.isActive = true;

		if (cpuMesh.get()) {
			meshViewer.setMainMesh(*cpuMesh, sibr::Mesh::FillRenderMode, false, true);
		} else {
			std::cout << " no mesh " << std::endl;
		}
		
		winSize = window.size().cast<float>();

		while (window.isOpened()) {

			//std::cout << "." << std::flush;

			window.makeContextCurrent();
			sibr::Input::poll();

			winSize = window.size().cast<float>();
			imagesInput = sibr::Input::subInput(sibr::Input::global(), imagesView.viewport);
			meshInput = sibr::Input::subInput(sibr::Input::global(), meshView.viewport);

			if (sibr::Input::global().key().isPressed(sibr::Key::Escape)) {
				break;
			}

			update(window, sibr::Input::global());

			onGui();

			render();

			f(this);

			window.swapBuffer();

		}
	}

	void MultiViewInterface::addCameras(const std::vector<InputCamera::Ptr>& input_cams)
	{
		cams = input_cams;
	}

	void MultiViewInterface::addMesh(const sibr::Mesh::Ptr &mesh)
	{
		cpuMesh = mesh;
		cpuMesh->generateNormals();
	}

	void MultiViewInterface::addMesh(const sibr::Mesh & mesh)
	{
		cpuMesh = std::make_shared<sibr::Mesh>();
		cpuMesh->vertices(mesh.vertices());
		cpuMesh->triangles(mesh.triangles());
		cpuMesh->generateNormals();
	}

	void MultiViewInterface::update(sibr::Window & window, const sibr::Input & input)
	{
		updateImageView(imagesView.viewport, imagesInput);
		updateMeshView(meshInput, window);	
	}

	void MultiViewInterface::updateImageView(const sibr::Viewport & viewport, const sibr::Input & input)
	{
		sibr::Vector2f winSize = viewport.finalSize();
		currentActivePos = pixFromScreenPos(input.mousePosition(), winSize);
		imagesViewBase->currentActivePos = currentActivePos;

		imgPixelScreenSize = screenPosPixelsFloat({ 0,{ 1,1 } }, winSize) - screenPosPixelsFloat({ 0,{ 0,0 } }, winSize);

		updateCurrentLayer(input);

		if (input.key().isActivated(sibr::Key::LeftShift)) {
			return;
		}

		updateZoomBox(input, winSize);
		//updateCenter(imagesInput, imagesViewSize);
		updateZoomScroll(input);
		updateDrag(input, winSize);
	}

	void MultiViewInterface::render()
	{
		renderImageView(imagesView.viewport);


		////if (hightligthChanged) {
		//renderHighlightPixels();
		////}
		//displayHighlightedPixels(sibr::Vector3f(0, 1, 0), 0.15);

		displayMesh(meshView.viewport);

		//if (sibr::Input::global().key().isActivated(sibr::Key::C)) {
		//	int r = 50;
		//	utils.rectanglePixels(sibr::Vector3f(1, 0, 1), sibr::Input::global().mousePosition().cast<float>(), sibr::Vector2f(r, r), true, 0.15f, imagesViewSize);
		//	utils.circlePixels(sibr::Vector3f(0, 1, 1), sibr::Input::global().mousePosition().cast<float>(), r, true, 0.15f, imagesViewSize);
		//	
		//}

	}

	void MultiViewInterface::renderImageView(const sibr::Viewport & viewport)
	{
		sibr::Vector2f imagesViewSize = viewport.finalSize();
		displayImages(viewport);

		displayZoom(viewport);

		if (currentActivePos.isDefined) {
			highlightPixel(currentActivePos, viewport);
		}

	}

	sibr::ViewBase::Ptr MultiViewInterface::getViewBase(MultiViewInterfaceView::ViewType type)
	{
		if (type == MultiViewInterfaceView::ViewType::IMAGES) {
			return sibr::ViewBase::Ptr(imagesViewBase.get());
		} else {
			return sibr::ViewBase::Ptr(meshViewBase.get());
		}
		
	}

	void MultiViewInterface::onGui()
	{
		ImGui::Separator();
		if (imagesLayers.size() != 1) {
			ImGui::SliderInt("Laplacian scale", &currentScale, 0, (int)imagesLayers.size() - 1);
			ImGui::Separator();
		}
		const size_t nLayers = layersData.size();
		if (nLayers > 1) {
			ImGui::Text("Image Layers : ");
			ImGui::Separator();

			for (size_t n = 0; n < nLayers; ++n) {
				if (ImGui::Selectable(layersData[n].name.c_str(), currentLayer == n)) {
					currentLayer = (int)n;
				}
			}
			ImGui::Separator();
		}
		if (currentScale == 0 && currentActivePos.isDefined) {
			std::stringstream ss;
			ss << "Image : " << currentActivePos.im << ", pixel : " << currentActivePos.pos << std::endl;
			ImGui::Text(ss.str().c_str());
			if (currentActivePos.isDefined) {
				//std::cout << imagesPtr[currentLayer][currentActivePos.im] << std::endl;
				//std::cout << imagesPtr[currentLayer][currentActivePos.im]->size() << std::endl;
/*				if (imagesPtr[currentLayer][currentActivePos.im]) {
					ImGui::Text(imagesPtr[currentLayer][currentActivePos.im]->pixelStr(currentActivePos.pos).c_str());
				}	*/		
			}
			ImGui::Separator();
		}

	}

	//MultiViewInterface::~MultiViewInterface()
	//{
	//	CHECK_GL_ERROR;

	//	CHECK_GL_ERROR;
	//}

	PixPos MultiViewInterface::pixFromScreenPos(const sibr::Vector2i & posScreen, const sibr::Vector2f & winSize)
	{
		UV01 uvScreen = UV10::from((posScreen.cast<float>()+0.5f*sibr::Vector2f(1,1)).cwiseQuotient(winSize));
		
		//std::cout << uvScreen.transpose() << std::endl;

		sibr::Vector2f posF = viewRectangle.tl() + (viewRectangle.br() - viewRectangle.tl()).cwiseProduct(uvScreen);
		posF.y() = 1.0f - posF.y();
		
		posF = posF.cwiseProduct(grid.cast<float>());

		//std::cout << posF.transpose() << " " << numImgs << std::endl;

		if (posF.x() < 0 || posF.y() < 0 || posF.x() >= grid.x() /* || posF.y() >= grid.y()  */ ) {
			return PixPos();
		}

		int x = (int)std::floor(posF.x());
		int y = (int)std::floor(posF.y());
		sibr::Vector2f frac = posF - sibr::Vector2f(x, y);

		int n = x + grid.x() * y;
		int j = (int)std::floor(frac.x()*scalesData[currentScale].imSize.x());
		int i = (int)std::floor(frac.y()*scalesData[currentScale].imSize.y());

		if (n >= numImgs) {
			return PixPos();
		}

		return PixPos(n, sibr::Vector2i(j, i));
	}

	UV01 MultiViewInterface::screenPos(const PixPos & pix)
	{
		sibr::Vector2f pos = (pix.pos.cast<float>().cwiseQuotient(scalesData[currentScale].imSize) +
			sibr::Vector2f(pix.im % grid.x(), pix.im / grid.x())).cwiseQuotient(grid.cast<float>());
		pos.y() = 1.0f - pos.y();
		return UV01::from((pos - viewRectangle.tl()).cwiseQuotient(viewRectangle.br() - viewRectangle.tl()));
	}

	UV01 MultiViewInterface::screenPosPixelCenter(const PixPos & pix)
	{
		sibr::Vector2f pos = ((pix.pos.cast<float>()+sibr::Vector2f(0.5,0.5)).cwiseQuotient(scalesData[currentScale].imSize) +
			sibr::Vector2f(pix.im % grid.x(), pix.im / grid.x())).cwiseQuotient(grid.cast<float>());
		pos.y() = 1.0f - pos.y();
		return UV01::from((pos - viewRectangle.tl()).cwiseQuotient(viewRectangle.br() - viewRectangle.tl()));
	}

	sibr::Vector2i MultiViewInterface::screenPosPixels(const PixPos & pix, const sibr::Vector2f & winSize)
	{
		return (screenPos(pix).cwiseProduct(winSize)).cast<int>();
	}

	sibr::Vector2f MultiViewInterface::screenPosPixelsFloat(const PixPos & pix, const sibr::Vector2f & winSize)
	{
		return screenPosPixelCenter(pix).cwiseProduct(winSize);
	}

	//void MultiViewInterface::setupFromImSizeAndNumIm(LayerData & layerData, const sibr::Vector2i & imSize)
	//{
	//	imRatio = imSize[0] / (float)imSize[1];
	//	imSizeF = imSize.cast<float>();
	//	numImgs = numIms;
	//}

	void MultiViewInterface::addHighlightPixel(const PixPos & pix, const sibr::Vector2f & winSize)
	{
		highlightedPixels.push_back(pix);
		//hightligthChanged = true;
	}

	void MultiViewInterface::renderHighlightPixels()
	{
		int pixsSize = (int)highlightedPixels.size();

		if (pixsSize == 0) {
			return;
		}

		if (!highlightedPixelsMesh.get()) {
			highlightedPixelsMesh = std::make_shared<sibr::Mesh>();
		}

		sibr::Mesh::Vertices vs(4 * pixsSize);
		sibr::Mesh::Triangles ts(2 * pixsSize);

		unsigned pixId = 0;
		for (const auto & pix : highlightedPixels) {
			UV11 tl = screenPos(pix);
			PixPos otherCorner(pix.im, pix.pos + sibr::Vector2i(1, 1));
			UV11 br = screenPos(otherCorner);

			vs[4 * pixId + 0] = { tl.x(), tl.y() , 0 };
			vs[4 * pixId + 1] = { tl.x(), br.y() , 0 };
			vs[4 * pixId + 2] = { br.x(), br.y() , 0 };
			vs[4 * pixId + 3] = { br.x(), tl.y() , 0 };

			ts[2 * pixId + 0] = { 4 * pixId + 0,4 * pixId + 1,4 * pixId + 2 };
			ts[2 * pixId + 1] = { 4 * pixId + 0,4 * pixId + 2,4 * pixId + 3 };

			++pixId;
		}



		highlightedPixelsMesh->vertices(vs);
		highlightedPixelsMesh->triangles(ts);

		//hightligthChanged = false;
	}

	void MultiViewInterface::highlightPixel(const PixPos & pix, const sibr::Viewport & viewport, const sibr::Vector3f & color, const sibr::Vector2f & pixScreenSize)
	{
		UV01 pixTl = screenPos(pix);
		PixPos otherCorner(pix.im, pix.pos + sibr::Vector2i(1, 1));
		UV01 pixBR = screenPos(otherCorner);

		viewport.bind();
				
		if ((pixBR-pixTl).cwiseProduct(viewport.finalSize()).cwiseAbs().minCoeff() < 2.0f) {
			//if pixel size in screen space is tinier than 2 screen pix
			utils.rectanglePixels(color, 0.5f*(pixTl+pixBR).cwiseProduct(viewport.finalSize()), pixScreenSize, true, 0.15f, viewport.finalSize());
		} else {	
			//otherwise hightligh pixel intirely
			utils.rectangle(color, pixTl, pixBR, true, 0.15f);
		}

	}

	void MultiViewInterface::displayImages(const sibr::Viewport & viewport)
	{
		
			//const sibr::Vector2f & winSize;
		//std::cout << imagesView.viewport.left() << " " << imagesView.viewport.right() << " " << imagesView.viewport.top() << " " << imagesView.viewport.bottom() << std::endl;
		viewport.bind();

		viewport.clear(sibr::Vector3f(0.7f, 0.7f, 0.7f));
		//glClear(GL_COLOR_BUFFER_BIT);

		utils.multiViewShader.begin();

		utils.numImgsGL.set((int)imagesLayers[currentScale][currentLayer]->depth() - 1);
		sibr::Vector2f gridF = grid.cast<float>();
		utils.gridGL.set(gridF);

		utils.multiViewTopLeftGL.set(viewRectangle.tl());
		utils.multiViewBottomRightGL.set(viewRectangle.br());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, imagesLayers[currentScale][currentLayer]->handle());
		sibr::RenderUtility::renderScreenQuad();

		utils.multiViewShader.end();

		for (int i = 0; i <(int)imagesLayers[currentScale][currentLayer]->depth(); ++i) {
			UV01 imTl = screenPos(PixPos(i, sibr::Vector2i(0, 0)));
			UV01 imBR = screenPos(PixPos(i, scalesData[currentScale].imSize.cast<int>()));
			utils.rectangle(sibr::Vector3f(0, 0, 0), imTl, imBR, false, 1.0);
		}

		//utils.rectangle(sibr::Vector3f(0, 1, 0), UV01(0.25,0.25), UV01(0.75, 0.75), true, 0.15);
		//utils.circle(sibr::Vector3f(0, 0, 1), UV01(0.33, 0.33), 0.1, true, 0.25);

		if (reproMesh.get()) {
			utils.meshViewShader.begin();
			utils.alphaMeshGL.set(0.25f);
			utils.colorMeshGL.set(sibr::Vector3f(1, 0, 1));

			sibr::Vector2i viewPortSize(imagesView.viewport.finalWidth(), imagesView.viewport.finalHeight());

			Eigen::AlignedBox2d winBox;
			winBox.extend(sibr::Vector2d(0, 0));

			winBox.extend(viewPortSize.cast<double>());
			//std::cout << std::endl;

			//std::cout << " winbox " << (winBox.center()-0.5*winBox.diagonal()).transpose() << " " << (winBox.center() + 0.5*winBox.diagonal()).transpose() << std::endl;
			for (int i = 0; i <(int)cams.size(); ++i) {
				utils.mvp.set(cams[i]->viewproj());

				//std::cout << i << std::endl;
				glClearDepth(1.0);
				glClear(GL_DEPTH_BUFFER_BIT);

				sibr::Vector2i tlImgPix = screenPosPixels(PixPos(i, sibr::Vector2i(0, cams[i]->h() - 1)), viewPortSize.cast<float>());
				sibr::Vector2i brImgPix = screenPosPixels(PixPos(i, sibr::Vector2i(cams[i]->w() - 1, 0)), viewPortSize.cast<float>());

				Eigen::AlignedBox2d box;
				box.extend(tlImgPix.cast<double>());
				box.extend(brImgPix.cast<double>());
				//std::cout << "\t box " << (box.center() - 0.5*box.diagonal()).transpose() << " " << (box.center() + 0.5*box.diagonal()).transpose() << std::endl;

				Eigen::AlignedBox2d renderBox = winBox.intersection(box);

				if (renderBox.isEmpty()) {
					continue;
				}

				//tlImgPix = renderBox.corner(Eigen::AlignedBox2d::CornerType::BottomLeft).cast<int>();
				//	brImgPix = renderBox.corner(Eigen::AlignedBox2d::CornerType::TopRight).cast<int>();
				//std::cout << "\t renderBox " << (renderBox.center() - 0.5*renderBox.diagonal()).transpose() << " " << (renderBox.center() + 0.5*renderBox.diagonal()).transpose() << std::endl;

				//std::cout << tlImgPix.x() << " " << tlImgPix.y() << " " << (brImgPix - tlImgPix).x() << " " << (brImgPix - tlImgPix).y() << std::endl;
				glViewport(tlImgPix.x(), tlImgPix.y(), std::abs((brImgPix - tlImgPix).x()), std::abs((brImgPix - tlImgPix).y()));

				reproMesh->render(true, reproMeshBackFace, reproMeshMode);
			}

			utils.meshViewShader.end();
		}
	}

	void MultiViewInterface::displayMesh(const sibr::Viewport & viewport)
	{
		if (meshView.isActive) {		
			meshViewer.render(viewport);
		}
	}

	void MultiViewInterface::displayZoom(const sibr::Viewport & viewport)
	{
		if (zoomSelection.isActive) {
			viewport.bind();
			UV01 tl = UV01::from(zoomSelection.first.cast<float>().cwiseQuotient(viewport.finalSize()));
			UV01 br = UV01::from(zoomSelection.second.cast<float>().cwiseQuotient(viewport.finalSize()));
			utils.rectangle(sibr::Vector3f(1, 0, 0), tl, br, true, 0.15f);
		}
	}

	void MultiViewInterface::displayHighlightedPixels(const sibr::Vector3f & color, float alpha)
	{
		if (!highlightedPixelsMesh.get()) {
			return;
		}

		utils.baseShader.begin();

		utils.scalingGL.set(1.0f);
		utils.translationGL.set(sibr::Vector2f(0, 0));
		utils.colorGL.set(color);
		utils.alphaGL.set(alpha);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);

		highlightedPixelsMesh->render(false, false);

		utils.alphaGL.set(1.0f);
		highlightedPixelsMesh->render(false, false, sibr::Mesh::LineRenderMode);

		utils.baseShader.end();

		highlightedPixelsMesh.reset();

	}

	void MultiViewInterface::updateMeshView(const sibr::Input & input, sibr::Window & window)
	{

		if (meshView.isActive) {

			//meshViewer.trackBall->updateAspectWithViewport(meshView.viewport);
			meshViewer.interactCam->update(input, 1 / 60.f, meshView.viewport);

			if (sibr::Input::global().key().isPressed(sibr::Key::Left)) {
				sibr::Vector2i oldWinSize = window.size();
				window.size(oldWinSize.x() /2, oldWinSize.y());
				imagesView.viewport = sibr::Viewport(&window.viewport(), 0, 0, 1, 1);
				
				meshView.isActive = false;
			}
		} else {
			if (sibr::Input::global().key().isPressed(sibr::Key::Right)) {
				sibr::Vector2i oldWinSize = window.size();
				window.size(oldWinSize.x() * 2, oldWinSize.y());
				meshView.viewport = sibr::Viewport(&window.viewport(), 0.5, 0, 1, 1);
				imagesView.viewport = sibr::Viewport(&window.viewport(), 0, 0, 0.5, 1);
				meshViewer.interactCam->setup(cpuMesh, meshView.viewport);
				meshView.isActive = true;
			}
		}
	}

	void MultiViewInterface::updateMeshView(const sibr::Input & input, const sibr::Viewport & viewport)
	{
		if (!meshView.isActive && cpuMesh.get() ) {
			meshViewer.setMainMesh(*cpuMesh, sibr::Mesh::FillRenderMode, false, true);
			meshViewer.interactCam->setup(cpuMesh, viewport);
			meshView.isActive = true;
		}
		if (meshView.isActive) {
			meshViewer.interactCam->update(input,1/60.0f, viewport);
		}				
	}

	void MultiViewInterface::updateZoomBox(const sibr::Input & input, const sibr::Vector2f & winSize)
	{
		if (input.key().isPressed(sibr::Key::Q)) {
			viewRectangle.center = sibr::Vector2f(0.5, 0.5);
			viewRectangle.diagonal = sibr::Vector2f(0.5, 0.5);
		}

		if (input.mouseButton().isPressed(sibr::Mouse::Code::Right) && !zoomSelection.isActive) {
			zoomSelection.isActive = true;
			zoomSelection.first = input.mousePosition();
			zoomSelection.first.y() = (int)winSize.y() - zoomSelection.first.y() - 1;
		}
		if (input.mouseButton().isActivated(sibr::Mouse::Code::Right) && zoomSelection.isActive) {
			zoomSelection.second = input.mousePosition();
			zoomSelection.second.y() = (int)winSize.y() - zoomSelection.second.y() - 1;
		}
		if (input.mouseButton().isReleased(sibr::Mouse::Code::Right) && zoomSelection.isActive) {	
			sibr::Vector2f currentTL = (zoomSelection.first.cwiseMin(zoomSelection.second)).cast<float>();
			sibr::Vector2f currentBR = (zoomSelection.first.cwiseMax(zoomSelection.second)).cast<float>();

			if (((currentBR - currentTL).array() > sibr::Vector2f(10, 10).array()).all()) {
				

				sibr::Vector2f tlPix = viewRectangle.tl().cwiseProduct(winSize) + (viewRectangle.br() - viewRectangle.tl()).cwiseProduct(currentTL);
				sibr::Vector2f brPix = viewRectangle.tl().cwiseProduct(winSize) + (viewRectangle.br() - viewRectangle.tl()).cwiseProduct(currentBR);

				sibr::Vector2f center = 0.5f*(brPix + tlPix);
				sibr::Vector2f diag = 0.5f*(brPix - tlPix);

				float new_ratio = diag.x() / diag.y();
				float target_ratio = scalesData[currentScale].imRatio;
				if (new_ratio > target_ratio) {
					diag.y() = diag.x() / target_ratio;
				} else {
					diag.x() = diag.y() * target_ratio;
				}

				viewRectangle.center = center.cwiseQuotient(winSize);
				viewRectangle.diagonal = diag.cwiseQuotient(winSize);

				zoomSelection.isActive = false;
			}
		}
	}

	void MultiViewInterface::updateCurrentLayer(const sibr::Input & input)
	{
		int i = -1;

		std::vector<sibr::Key::Code> keys = {
			sibr::Key::Num1, sibr::Key::Num2, sibr::Key::Num3, sibr::Key::Num4, sibr::Key::Num5,
			sibr::Key::Num6, sibr::Key::Num7, sibr::Key::Num8, sibr::Key::Num9
		};

		for (int k = 0; k < (int)keys.size(); ++k) {
			if (input.key().isPressed(keys[k])) {
				i = k;
				break;
			}
		}

		if (i >= 0 && i < (int)imagesLayers.size()) {
			currentLayer = i;
		}
	}

	void MultiViewInterface::updateZoomScroll(const sibr::Input & input)
	{
		double scroll = input.mouseScroll();

		if (scroll  != 0) {	
			float ratio = (scroll > 0 ? 0.75f : 1.33f);
			if (input.key().isActivated(sibr::Key::LeftControl)) {
				ratio *= ratio;
			}
			viewRectangle.diagonal *= ratio;
		}
	}

	void MultiViewInterface::updateCenter(const sibr::Input & input, const sibr::Vector2f & winSize)
	{
		if (dclick.detected(input)) {
			//std::cout << "dclick : " << std::endl;
			sibr::Vector2f translation = (dclick.firstPosition.cast<float>().cwiseQuotient(winSize)-sibr::Vector2f(0.5,0.5)).cwiseProduct(viewRectangle.br() - viewRectangle.tl());
			translation.y() = -translation.y();
			viewRectangle.center += translation;
		}
	}

	void MultiViewInterface::updateDrag(const sibr::Input & input, const sibr::Vector2f & winSize)
	{
		if (input.mouseButton().isPressed(sibr::Mouse::Left)) {
			drag.isActive = true;
			drag.position = input.mousePosition();
			drag.center = viewRectangle.center;
		} else if (drag.isActive && input.mouseButton().isReleased(sibr::Mouse::Left)) {
			drag.isActive = false;
		}
		if (drag.isActive && input.mouseButton().isActivated(sibr::Mouse::Left)) {
			sibr::Vector2f translation = (input.mousePosition() - drag.position).cast<float>().cwiseQuotient(winSize).cwiseProduct(viewRectangle.br() - viewRectangle.tl());
			translation.y() = -translation.y();
			viewRectangle.center = drag.center - translation;
		}
	}

	
	void MultiViewInterfaceView::onUpdate(Input & input, const sibr::Viewport & viewport)
	{
		if (viewType == ViewType::IMAGES) {
			//i.imagesInput = input;
			//i.imagesView.viewport = viewport;
			interfacePtr->updateImageView(viewport, input);
		} else if (viewType == ViewType::MESH) {
			interfacePtr->updateMeshView(input, viewport);
		}
	}

	void MultiViewInterfaceView::onRender(const sibr::Viewport & viewport)
	{
		if (viewType == ViewType::IMAGES) {
			interfacePtr->renderImageView(viewport);
			interfacePtr->onGui();
		} else if (viewType == ViewType::MESH) {
			interfacePtr->displayMesh(viewport);
		}
	}

} //namespace sibr
