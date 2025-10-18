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



# include "core/graphics/GUI.hpp"
# include "core/view/MultiViewManager.hpp"

namespace sibr
{
	MultiViewBase::MultiViewBase(const Vector2i & defaultViewRes)
	{
		/// \todo TODO: support launch arg for stereo mode.
		renderingMode(IRenderingMode::Ptr(new MonoRdrMode()));

		//Default view resolution.
		setDefaultViewResolution(defaultViewRes);

		_timeLastFrame = std::chrono::steady_clock::now();
		_deltaTime = 0.0;
		_exportPath = "./screenshots";
	}

	void MultiViewBase::onUpdate(Input& input)
	{
		if (input.key().isActivated(Key::LeftControl) && input.key().isPressed(Key::LeftAlt) && input.key().isPressed(Key::P)) {
			_onPause = !_onPause;
		}
		if (_onPause) {
			return;
		}

		// Elapsed time since last rendering.
		const auto timeNow = std::chrono::steady_clock::now();
		_deltaTime = (float)(std::chrono::duration_cast<std::chrono::microseconds>(timeNow - _timeLastFrame).count())/1000000.0f;
		_timeLastFrame = timeNow;

		for (auto & subview : _subViews) {
			if (subview.second.view->active()) {
				auto subInput = !subview.second.view->isFocused() ? Input() : Input::subInput(input, subview.second.viewport, false);

				if (subview.second.handler) {
					subview.second.handler->update(subInput, _deltaTime, subview.second.viewport);
				}

				subview.second.updateFunc(subview.second.view, subInput, subview.second.viewport, _deltaTime);

			}
		}

		for (auto & subview : _ibrSubViews) {
			MultiViewBase::IBRSubView & fView = subview.second;

			if (fView.view->active()) {
				auto subInput = !fView.view->isFocused() ? Input() : Input::subInput(input, fView.viewport, false);

				if (fView.handler) {
					fView.handler->update(subInput, _deltaTime, fView.viewport);
				}

				fView.cam = fView.updateFunc(fView.view, subInput, fView.viewport, _deltaTime);

				/// If we use the default update func and the integrated handler, 
				/// we have to use the handler's camera.
				if (fView.defaultUpdateFunc && fView.handler) {
					fView.cam = fView.handler->getCamera();
				}

			}
		}

		for (auto & subMultiView : _subMultiViews) {
			subMultiView.second->onUpdate(input);
		}
	}

	void MultiViewBase::onRender(Window& win)
	{
		// Render all views.
		for (auto & subview : _ibrSubViews) {
			if (subview.second.view->active()) {

				renderSubView(subview.second);

				if (_enableGUI && _showSubViewsGui) {
					subview.second.view->onGUI();
					if (subview.second.handler) {
						subview.second.handler->onGUI("Camera " + subview.first);
					}
				}
			}
		}
		for (auto & subview : _subViews) {
			if (subview.second.view->active()) {

				renderSubView(subview.second);
				
				if (_enableGUI && _showSubViewsGui) {
					subview.second.view->onGUI();
					if (subview.second.handler) {
						subview.second.handler->onGUI("Camera " + subview.first);
					}
				}
			}
		}
		for (auto & subMultiView : _subMultiViews) {
			subMultiView.second->onRender(win);
		}


	}

	void MultiViewBase::onGui(Window & win)
	{
	}

	void MultiViewBase::addSubView(const std::string & title, ViewBase::Ptr view, const Vector2u & res, const ImGuiWindowFlags flags)
	{
		const ViewUpdateFunc updateFunc =
			[](ViewBase::Ptr& vi, Input& in, const Viewport& vp, const float dt) {
			vi->onUpdate(in, vp);
		};
		addSubView(title, view, updateFunc, res, flags);
	}

	void MultiViewBase::addSubView(const std::string & title, ViewBase::Ptr view, const ViewUpdateFunc updateFunc, const Vector2u & res, const ImGuiWindowFlags flags)
	{
		float titleBarHeight = 0.0f;
		if(_enableGUI) titleBarHeight = ImGui::GetTitleBarHeight();
		// We have to shift vertically to avoid an overlap with the menu bar.
		const Viewport viewport(0.0f, titleBarHeight,
			res.x() > 0 ? res.x() : (float)_defaultViewResolution.x(),
			(res.y() > 0 ? res.y() : (float)_defaultViewResolution.y()) + titleBarHeight);
		RenderTargetRGB::Ptr rtPtr(new RenderTargetRGB((uint)viewport.finalWidth(), (uint)viewport.finalHeight(), SIBR_CLAMP_UVS));
		_subViews[title] = {view, rtPtr, viewport, title, flags, updateFunc };

	}

	void MultiViewBase::addIBRSubView(const std::string & title, ViewBase::Ptr view, const IBRViewUpdateFunc updateFunc, const Vector2u & res, const ImGuiWindowFlags flags, const bool defaultFuncUsed)
	{
		float titleBarHeight = 0.0f;
		if(_enableGUI) titleBarHeight = ImGui::GetTitleBarHeight();
		// We have to shift vertically to avoid an overlap with the menu bar.
		const Viewport viewport(0.0f, titleBarHeight,
			res.x() > 0 ? res.x() : (float)_defaultViewResolution.x(),
			(res.y() > 0 ? res.y() : (float)_defaultViewResolution.y()) + titleBarHeight);
		RenderTargetRGB::Ptr rtPtr(new RenderTargetRGB((uint)viewport.finalWidth(), (uint)viewport.finalHeight(), SIBR_CLAMP_UVS));
		if (_ibrSubViews.count(title) > 0){
			const auto handler = _ibrSubViews[title].handler;
			_ibrSubViews[title] = { view, rtPtr, viewport, title, flags, updateFunc, defaultFuncUsed };
			_ibrSubViews[title].handler = handler;
		}
		else {
			_ibrSubViews[title] = { view, rtPtr, viewport, title, flags, updateFunc, defaultFuncUsed };
		}
		_ibrSubViews[title].shouldUpdateLayout = true;
	}

	void MultiViewBase::addIBRSubView(const std::string & title, ViewBase::Ptr view, const Vector2u & res, const ImGuiWindowFlags flags)
	{
		const auto updateFunc = [](ViewBase::Ptr& vi, Input& in, const Viewport& vp, const float dt) {
			vi->onUpdate(in, vp);
			return InputCamera();
		};
		addIBRSubView(title, view, updateFunc, res, flags, true);
	}

	void MultiViewBase::addIBRSubView(const std::string & title, ViewBase::Ptr view, const IBRViewUpdateFunc updateFunc, const Vector2u & res, const ImGuiWindowFlags flags)
	{
		addIBRSubView(title, view, updateFunc, res, flags, false);
	}

	void MultiViewBase::addSubMultiView(const std::string & title, MultiViewBase::Ptr multiview)
	{
		_subMultiViews[title] = multiview;
	}

	ViewBase::Ptr & MultiViewBase::getIBRSubView(const std::string & title)
	{
		if (_subViews.count(title) > 0) {
			return _subViews.at(title).view;
		}
		if (_ibrSubViews.count(title) > 0) {
			return _ibrSubViews.at(title).view;
		}

		SIBR_ERR << " No subview with name <" << title << "> found." << std::endl;

		return _subViews.begin()->second.view;
	}

	Viewport & MultiViewBase::getIBRSubViewport(const std::string & title)
	{
		if (_subViews.count(title) > 0) {
			return _subViews.at(title).viewport;
		}
		else if (_ibrSubViews.count(title) > 0) {
			return _ibrSubViews.at(title).viewport;
		}

		SIBR_ERR << " No subviewport with name <" << title << "> found." << std::endl;

		return _subViews.begin()->second.viewport;
	}

	void MultiViewBase::renderSubView(SubView & subview) 
	{
		
		if (!_onPause) {

			const Viewport renderViewport(0.0, 0.0, (float)subview.rt->w(), (float)subview.rt->h());
			subview.render(_renderingMode, renderViewport);

			// Offline video dumping, continued. We ignore additional rendering as those often are GUI overlays.
			if (subview.handler != NULL && (subview.handler->getCamera().needVideoSave() || subview.handler->getCamera().needSave())) {
				
				ImageRGB frame;

				subview.rt->readBack(frame);
				
				if (subview.handler->getCamera().needSave()) {
					frame.save(subview.handler->getCamera().savePath());
				}
				_videoFrames.push_back(frame.toOpenCVBGR());
				
			}
			
			// Additional rendering.
			subview.renderFunc(subview.view, renderViewport, std::static_pointer_cast<IRenderTarget>(subview.rt));

			// Render handler if needed.
			if (subview.handler) {
				subview.rt->bind();
				renderViewport.bind();
				subview.handler->onRender(renderViewport);
				subview.rt->unbind();
			}
		}

		if(_enableGUI)
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
			subview.view->setFocus(showImGuiWindow(subview.view->name(), *subview.rt, subview.flags, subview.viewport, false, subview.shouldUpdateLayout));
			ImGui::PopStyleVar();
		}
		// If we have updated the layout, don't do it next frame.
		subview.shouldUpdateLayout = false;
	}

	ViewBase::Ptr MultiViewBase::removeSubView(const std::string & title)
	{
		ViewBase::Ptr viewPtr = nullptr;
		if (_subViews.count(title) > 0) {
			viewPtr = _subViews.at(title).view;
			_subViews.erase(title);
		}
		else if (_ibrSubViews.count(title) > 0) {
			viewPtr = _ibrSubViews.at(title).view;
			_ibrSubViews.erase(title);
		}
		else {
			SIBR_WRG << "No view named <" << title << "> found." << std::endl;
		}
		return viewPtr;
	}

	void MultiViewBase::renderingMode(const IRenderingMode::Ptr& mode)
	{
		_renderingMode = std::move(mode);
	}

	const Viewport MultiViewBase::getViewport(void) const
	{
		return Viewport(0.0f, 0.0f, (float)_defaultViewResolution.x(), (float)_defaultViewResolution.y());
	}

	void MultiViewBase::addCameraForView(const std::string & name, ICameraHandler::Ptr cameraHandler)
	{
		if (_subViews.count(name) > 0) {
			_subViews.at(name).handler = cameraHandler;
		}
		else if (_ibrSubViews.count(name) > 0) {
			_ibrSubViews.at(name).handler = cameraHandler;

			SubView & subview = _ibrSubViews.at(name);
		}
		else {
			SIBR_WRG << "No view named <" << name << "> found." << std::endl;
		}

	}

	void MultiViewBase::addAdditionalRenderingForView(const std::string & name, const AdditionalRenderFunc renderFunc)
	{
		if (_subViews.count(name) > 0) {
			_subViews.at(name).renderFunc = renderFunc;
		}
		else if (_ibrSubViews.count(name) > 0) {
			_ibrSubViews.at(name).renderFunc = renderFunc;
		}
		else {
			SIBR_WRG << "No view named <" << name << "> found." << std::endl;
		}
	}

	int MultiViewBase::numSubViews() const
	{
		return static_cast<int>(_subViews.size() + _ibrSubViews.size() + _subMultiViews.size());
	}

	void MultiViewBase::captureView(const std::string & subviewName, const std::string & path, const std::string & filename)
	{
		if (_subViews.count(subviewName)) {
			captureView(_subViews[subviewName], path, filename);
		}
		else if (_ibrSubViews.count(subviewName)) {
			captureView(_ibrSubViews[subviewName], path, filename);
		}
		else {
			SIBR_WRG << "No View in the MultiViewManager with " << subviewName << " as a name!" << std::endl;
		}
	}

	void MultiViewBase::captureView(const SubView & view, const std::string& path, const std::string & filename) {

		const uint w = view.rt->w();
		const uint h = view.rt->h();

		ImageRGB renderingImg(w, h);

		view.rt->readBack(renderingImg);

		std::string finalPath = path + (!path.empty() ? "/" : "");
		if (!filename.empty()) {
			finalPath.append(filename);
		}
		else {
			const std::string autoName = view.view->name() + "_" + sibr::timestamp();
			finalPath.append(autoName + ".png");
		}

		makeDirectory(path);
		renderingImg.save(finalPath, true);
	}

	void MultiViewBase::mosaicLayout(const Viewport & vp)
	{
		const int viewsCount = numSubViews();
		
		// Do square decomposition for now.
		// Find the next square.
		const int sideCount = int(std::ceil(std::sqrt(viewsCount)));
		int verticalShift = 0;
		if(_enableGUI) verticalShift = ImGui::GetTitleBarHeight();

		Viewport usedVP = Viewport(vp.finalLeft(), vp.finalTop() + verticalShift, vp.finalRight(), vp.finalBottom());
		Vector2f itemRatio = Vector2f(1, 1) / sideCount;

		int vid = 0;
		for (auto & view : _ibrSubViews) {
			// Compute position on grid.
			const int col = vid % sideCount;
			const int row = vid / sideCount;
			view.second.viewport = Viewport(usedVP, col*itemRatio[0], row * itemRatio[1], (col + 1)*itemRatio[0], (row + 1)*itemRatio[1]);
			view.second.shouldUpdateLayout = true;
			++vid;
		}
		for (auto & view : _subViews) {
			// Compute position on grid.
			const int col = vid % sideCount;
			const int row = vid / sideCount;
			view.second.viewport = Viewport(usedVP, col*itemRatio[0], row * itemRatio[1], (col + 1)*itemRatio[0], (row + 1)*itemRatio[1]);
			view.second.shouldUpdateLayout = true;
			++vid;
		}
		for (auto & view : _subMultiViews) {
			// Compute position on grid.
			const int col = vid % sideCount;
			const int row = vid / sideCount;
			view.second->mosaicLayout(Viewport(usedVP, col*itemRatio[0], row * itemRatio[1], (col + 1)*itemRatio[0], (row + 1)*itemRatio[1]));
			++vid;
		}

	}
	
	void MultiViewBase::toggleSubViewsGUI()
	{
		_showSubViewsGui = !_showSubViewsGui;

		for (auto & view : _subMultiViews) {
			view.second->toggleSubViewsGUI();
		}
	}

	void MultiViewBase::setExportPath(const std::string & path) {
		_exportPath = path;
		sibr::makeDirectory(path);
	}

	MultiViewBase::SubView::SubView(ViewBase::Ptr view_, RenderTargetRGB::Ptr rt_, const sibr::Viewport viewport_, const std::string& name_, const ImGuiWindowFlags flags_) :
		view(view_), rt(rt_), handler(), viewport(viewport_), flags(flags_), shouldUpdateLayout(false) {
		renderFunc = [](ViewBase::Ptr&, const Viewport&, const IRenderTarget::Ptr&) {};
		view->setName(name_);
	}

	MultiViewBase::BasicSubView::BasicSubView(ViewBase::Ptr view_, RenderTargetRGB::Ptr rt_, const sibr::Viewport viewport_, const std::string& name_, const ImGuiWindowFlags flags_, ViewUpdateFunc f_) :
		SubView(view_, rt_, viewport_, name_, flags_), updateFunc(f_) {
	}

	void MultiViewBase::BasicSubView::render(const IRenderingMode::Ptr& rm, const Viewport& renderViewport) const  {
		rt->bind();
		renderViewport.bind();
		renderViewport.clear();
		view->onRender(renderViewport);
		rt->unbind();
	}

	MultiViewBase::IBRSubView::IBRSubView(ViewBase::Ptr view_, RenderTargetRGB::Ptr rt_, const sibr::Viewport viewport_, const std::string& name_, const ImGuiWindowFlags flags_, IBRViewUpdateFunc f_, const bool defaultUpdateFunc_) :
		SubView(view_, rt_, viewport_, name_, flags_), updateFunc(f_), defaultUpdateFunc(defaultUpdateFunc_) {
		cam = sibr::InputCamera();
	}

	void MultiViewBase::IBRSubView::render(const IRenderingMode::Ptr& rm, const Viewport& renderViewport) const  {
		if (rm) {
			rm->render(*view, cam, renderViewport, rt.get());
		}
	}

	MultiViewManager::MultiViewManager(Window& window, bool resize)
		: _window(window), _fpsCounter(false)
	{
		_enableGUI = window.isGUIEnabled();
		
		if (resize) {
			window.size(
				Window::desktopSize().x() - 200,
				Window::desktopSize().y() - 200);
			window.position(100, 100);
		}

		/// \todo TODO: support launch arg for stereo mode.
		renderingMode(IRenderingMode::Ptr(new MonoRdrMode()));

		//Default view resolution.
		int w = int(window.size().x() * 0.5f);
		int h = int(window.size().y() * 0.5f);
		setDefaultViewResolution(Vector2i(w, h));

		if(_enableGUI) ImGui::GetStyle().WindowBorderSize = 0.0;
	}

	void MultiViewManager::onUpdate(Input & input)
	{
		MultiViewBase::onUpdate(input);

		if (input.key().isActivated(Key::LeftControl) && input.key().isActivated(Key::LeftAlt) && input.key().isReleased(Key::G)) {
			toggleGUI();
		}
	}

	void MultiViewManager::onRender(Window & win)
	{
		win.viewport().bind();
		glClearColor(37.f / 255.f, 37.f / 255.f, 38.f / 255.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);
		glClearColor(1.f, 1.f, 1.f, 1.f);

		onGui(win);

		MultiViewBase::onRender(win);

		_fpsCounter.update(_enableGUI && _showGUI);
	}

	void MultiViewManager::onGui(Window & win)
	{
		MultiViewBase::onGui(win);

		// Menu
		if (_showGUI && ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("Menu"))
			{
				ImGui::MenuItem("Pause", "", &_onPause);
				if (ImGui::BeginMenu("Display")) {
					const bool currentScreenState = win.isFullscreen();
					if (ImGui::MenuItem("Fullscreen", "", currentScreenState)) {
						win.setFullscreen(!currentScreenState);
					}

					const bool currentSyncState = win.isVsynced();
					if (ImGui::MenuItem("V-sync", "", currentSyncState)) {
						win.setVsynced(!currentSyncState);
					}

					const bool isHiDPI = ImGui::GetIO().FontGlobalScale > 1.0f;
					if (ImGui::MenuItem("HiDPI", "", isHiDPI)) {
						if (isHiDPI) {
							ImGui::GetStyle().ScaleAllSizes(1.0f / win.scaling());
							ImGui::GetIO().FontGlobalScale = 1.0f;
						} else {
							ImGui::GetStyle().ScaleAllSizes(win.scaling());
							ImGui::GetIO().FontGlobalScale = win.scaling();
						}
					}

					if (ImGui::MenuItem("Hide GUI (!)", "Ctrl+Alt+G")) {
						toggleGUI();
					}
					ImGui::EndMenu();
				}


				if (ImGui::MenuItem("Mosaic layout")) {
					mosaicLayout(win.viewport());
				}

				if (ImGui::MenuItem("Row layout")) {
					Vector2f itemSize = win.size().cast<float>();
					itemSize[0] = std::round(float(itemSize[0]) / float(_subViews.size() + _ibrSubViews.size()));
					const float verticalShift = ImGui::GetTitleBarHeight();
					float vid = 0.0f;
					for (auto & view : _ibrSubViews) {
						// Compute position on grid.
						view.second.viewport = Viewport(vid*itemSize[0], verticalShift, (vid + 1.0f)*itemSize[0] - 1.0f, verticalShift + itemSize[1] - 1.0f);
						view.second.shouldUpdateLayout = true;
						++vid;
					}
					for (auto & view : _subViews) {
						// Compute position on grid.
						view.second.viewport = Viewport(vid*itemSize[0], verticalShift, (vid + 1.0f)*itemSize[0] - 1.0f, verticalShift + itemSize[1] - 1.0f);
						view.second.shouldUpdateLayout = true;
						++vid;
					}
				}


				if (ImGui::MenuItem("Quit", "Escape")) { win.close(); }
				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Views"))
			{
				for (auto & subview : _subViews) {
					if (ImGui::MenuItem(subview.first.c_str(), "", subview.second.view->active())) {
						subview.second.view->active(!subview.second.view->active());
					}
				}
				for (auto & subview : _ibrSubViews) {
					if (ImGui::MenuItem(subview.first.c_str(), "", subview.second.view->active())) {
						subview.second.view->active(!subview.second.view->active());
					}
				}
				if (ImGui::MenuItem("Metrics", "", _fpsCounter.active())) {
					_fpsCounter.toggleVisibility();
				}
				if (ImGui::BeginMenu("Front when focus"))
				{
					for (auto & subview : _subViews) {
						const bool isLockedInBackground = subview.second.flags & ImGuiWindowFlags_NoBringToFrontOnFocus;
						if (ImGui::MenuItem(subview.first.c_str(), "", !isLockedInBackground)) {
							if(isLockedInBackground) {
								subview.second.flags &= ~ImGuiWindowFlags_NoBringToFrontOnFocus;
							} else {
								subview.second.flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
							}
						}
					}
					for (auto & subview : _ibrSubViews) {
						const bool isLockedInBackground = subview.second.flags & ImGuiWindowFlags_NoBringToFrontOnFocus;
						if (ImGui::MenuItem(subview.first.c_str(), "", !isLockedInBackground)) {
							if (isLockedInBackground) {
								subview.second.flags &= ~ImGuiWindowFlags_NoBringToFrontOnFocus;
							} else {
								subview.second.flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
							}
						}
					}
					ImGui::EndMenu();
				}
				if (ImGui::MenuItem("Reset Settings to Default", "")) {
					_window.resetSettingsToDefault();
				}
				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Capture"))
			{

				if (ImGui::MenuItem("Set export directory...")) {
					std::string selectedDirectory;
					if (showFilePicker(selectedDirectory, FilePickerMode::Directory)) {
						if (!selectedDirectory.empty()) {
							_exportPath = selectedDirectory;
						}
					}
				}

				for (auto & subview : _subViews) {
					if (ImGui::MenuItem(subview.first.c_str())) {
						captureView(subview.second, _exportPath);
					}
				}
				for (auto & subview : _ibrSubViews) {
					if (ImGui::MenuItem(subview.first.c_str())) {
						captureView(subview.second, _exportPath);
					}
				}

				if (ImGui::MenuItem("Export Video")) {
					std::string saveFile;
					if (showFilePicker(saveFile, FilePickerMode::Save)) {
						const std::string outputVideo = saveFile + ".mp4";
						if(!_videoFrames.empty()) {
							SIBR_LOG << "Exporting video to : " << outputVideo << " ..." << std::flush;
							FFVideoEncoder vdoEncoder(outputVideo, 30, Vector2i(_videoFrames[0].cols, _videoFrames[0].rows));
							for (int i = 0; i < _videoFrames.size(); i++) {
								vdoEncoder << _videoFrames[i];
							}
							_videoFrames.clear();
							std::cout << " Done." << std::endl;
							
						} else {
							SIBR_WRG << "No frames to export!! Check save frames in camera options for the view you want to render and play the path and re-export!" << std::endl;
						}
					}
				}

				ImGui::EndMenu();
			}

			ImGui::EndMainMenuBar();
		}
	}

	void MultiViewManager::toggleGUI()
	{
		_showGUI = !_showGUI;
		if (!_showGUI) {
			SIBR_LOG << "[MultiViewManager] GUI is now hidden, use Ctrl+Alt+G to toggle it back on." << std::endl;
		}
		toggleSubViewsGUI();
	}


} // namespace sibr
