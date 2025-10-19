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


#include "DatasetView.hpp"

namespace sibr {
	
	DatasetView::DatasetView(const BasicIBRScene & scene, const Vector2u & defaultRenderingRes, const Vector2i & defaultViewRes)
		: MultiViewBase(defaultViewRes)
	{
		const auto & input_cams = scene.cameras()->inputCameras();
		const auto & input_images = scene.images()->inputImages();

		if (input_images.size() != input_cams.size()) {
			SIBR_ERR << "cams not matching input images";

		}
		const std::string mmm_str = "mesh";
		MultiMeshManager::Ptr mmm(new MultiMeshManager(mmm_str));
		mmm->addMesh("proxy", scene.proxies()->proxyPtr());		
		mmm->getCameraHandler().fromCamera(*input_cams[0]);
		for (int i = 0; i < (int)input_cams.size(); ++i) {
			cams.push_back(*input_cams[i]);
		}

		const std::string grid_str = "grid";
		ImagesGrid::Ptr grid(new ImagesGrid());
		grid->addImageLayer("input images", input_images);

		addSubView(meshSubViewStr, mmm, defaultRenderingRes);
		addSubView(gridSubViewStr, grid, defaultRenderingRes);
	}

	void DatasetView::onGui(Window & win)
	{
	}

	void DatasetView::onUpdate(Input & input)
	{
		MultiViewBase::onUpdate(input);


		Input meshInput = Input::subInput(input, getMeshView().viewport);
		if (meshInput.key().isActivated(Key::LeftControl) && meshInput.mouseButton().isActivated(Mouse::Right)) {
			RaycastingCamera cam = RaycastingCamera(getMMM()->getCameraHandler().getCamera());
			Ray ray = cam.getRay(meshInput.mousePosition().cast<float>());
			auto hit = proxyData().raycaster->intersect(ray);

			if (hit.hitSomething()) {
				currentRepro.point3D = ray.at(hit.dist());
				currentRepro.active = true;
				currentRepro.repros.clear();
				repro(currentRepro);
				getGrid()->addPixelsToHighlight("zinputRepro", { }, { 1,0,0 }, 0.25f);
			}
		}

		Input gridInput = Input::subInput(input, getGridView().viewport);
		if (gridInput.key().isActivated(Key::LeftControl) && gridInput.mouseButton().isActivated(Mouse::Right)) {
			const auto & pix = getGrid()->getCurrentPixel();
			if (pix) {
				Ray ray = cams[pix.im].getRay(pix.pos.cast<float>());
				auto hit = proxyData().raycaster->intersect(ray);
				if (hit.hitSomething()) {
					currentRepro.point3D = ray.at(hit.dist());
					currentRepro.active = true;
					currentRepro.repros.clear();
					repro(currentRepro);
					getGrid()->addPixelsToHighlight("zinputRepro", { pix }, { 1,0,0 }, 0.25f);
				}
			}
		}


	}

	void DatasetView::onRender(Window & win)
	{
		if (currentRepro) {
			displayRepro(currentRepro);
		}

		MultiViewBase::onRender(win);
	}

	void DatasetView::repro(ReprojectionData & data)
	{
		const Vector3f & pt = data.point3D;
		for (int im = 0; im<(int)cams.size(); ++im) {
			const auto & cam = cams[im];
			if (!cam.frustumTest(pt)) {
				continue;
			}
			Vector3f pt2d = cam.projectImgSpaceInvertY(pt);

			if (data.occlusionTest) {
				float dist = (cam.position() - pt).norm();
				Ray ray = Ray(pt, (cam.position() - pt).normalized());
				auto hit = proxyData().raycaster->intersect(ray, 0.01f);
				if (hit.hitSomething() && std::abs(hit.dist() - dist) / dist > 0.01f) {
					continue;
				}
			}

			data.repros.push_back(MVpixel(im, pt2d.xy().cast<int>()));
		}
	}

	void DatasetView::displayRepro(const ReprojectionData & data)
	{
		getMMM()->addPoints("repro 3D point", { data.point3D });

		Mesh::Ptr reproLines(new Mesh());
		std::vector<MVpixel> pixs;
		std::vector<int> repro_imgs;
		for (const auto & rep : data.repros) {
			const auto & cam = cams[rep.im];
			Mesh reproLine;
			reproLine.vertices({ cam.position(), data.point3D });
			reproLine.triangles({ 0,0,1 });
			reproLines->merge(reproLine);
			repro_imgs.push_back(rep.im);
		}

		getMMM()->addMeshAsLines("repro ines", reproLines).setColor({ 1,0,1 });
		getGrid()->addPixelsToHighlight("repros", data.repros, { 0,0,1 }, 0.25f);
		//getGrid()->addImagesToHighlight("reproImgs", repro_imgs, { 0,1,0 }, 0.1f);
	}

	MultiViewBase::BasicSubView & DatasetView::getMeshView()
	{
		return _subViews[meshSubViewStr];
	}

	MultiViewBase::BasicSubView & DatasetView::getGridView()
	{
		return _subViews[gridSubViewStr];
	}

	MultiMeshManager::Ptr DatasetView::getMMM()
	{
		return std::static_pointer_cast<MultiMeshManager>(getMeshView().view);
	}

	ImagesGrid::Ptr sibr::DatasetView::getGrid()
	{
		return std::static_pointer_cast<ImagesGrid>(getGridView().view);
	}

	MeshData & DatasetView::proxyData()
	{
		return getMMM()->getMeshData("proxy");
	}

}

