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

# include <type_traits>
# include <chrono>

# include "core/view/Config.hpp"
# include "core/graphics/Window.hpp"
# include "core/graphics/Texture.hpp"
# include "core/view/RenderingMode.hpp"
# include "core/view/FPSCamera.hpp"

# include "core/assets/InputCamera.hpp"
# include "core/graphics/Input.hpp"
# include "core/graphics/Image.hpp"
# include "core/graphics/RenderUtility.hpp"
# include "core/assets/CameraRecorder.hpp"
# include "core/view/ViewBase.hpp"
# include "core/graphics/Shader.hpp"
# include "core/view/FPSCounter.hpp"
#include "core/video/FFmpegVideoEncoder.hpp"
#include "InteractiveCameraHandler.hpp"
#include <random>
#include <map>


namespace sibr
{

	/**
	 * MultiViewBase is designed to provide
	 * more flexibility and with a multi-windows system in mind.
	 * Once a MultiViewBase is created, you can register standard and
	 * IBR subviews, providing additional functions for update and
	 * rendering if needed, along with support for ImGui interfaces.
	 * MultiViewBase will wrap those views and manage them on screen.
	 * To support legacy rendering modes and views, we introduce a
	 * distinction between standard subviews, that will be rendered through
	 * a call to onRender(Viewport&), and IBR subviews rendered through
	 * a onRenderIBR(rt, eye) call. This also means that after updating
	 * (via onUpdate) an IBR subview, you have to return the camera
	 * that will be used for the onRenderIBR call.
	 * Note: new IBR views don't have to implement this distinction.
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT MultiViewBase
	{
		SIBR_CLASS_PTR(MultiViewBase);

	public:

		/// Update callback for a standard view. Passes the view pointer, the correct input state, and the correct viewport.
		typedef  std::function<void(sibr::ViewBase::Ptr &, sibr::Input&, const sibr::Viewport&, const float)> ViewUpdateFunc;
		/// Update callback for an IBR view, see main description for the return value.
		typedef  std::function<sibr::InputCamera(sibr::ViewBase::Ptr &, sibr::Input&, const sibr::Viewport&, const float)> IBRViewUpdateFunc;
		/// Additional render callback for a subview.
		typedef  std::function<void(sibr::ViewBase::Ptr &, const sibr::Viewport&, const IRenderTarget::Ptr& )> AdditionalRenderFunc;

		/*
		 * \brief Creates a MultiViewBase in a given OS window.
		 * \param defaultViewRes the default resolution for each subview
		 */
		MultiViewBase(const Vector2i & defaultViewRes = { 800, 600 });

		/**
		 * \brief Update subviews and the MultiViewBase.
		 * \param input The input state to use.
		 */
		virtual void	onUpdate(Input & input);

		/**
		 * \brief Render the content of the MultiViewBase
		 * \param win The OS window into which the rendering should be performed.
		 */
		virtual void	onRender(Window& win);

		/**
		 * \brief Render additional gui
		 * \param win The OS window into which the rendering should be performed.
		 */
		virtual void	onGui(Window& win);

		/**
		 * \brief Register a standard subview (for instance a SceneDebugView). It will be rendered via a call to onRender(Viewport) in an implicit rendertarget managed by the MultiViewBase.
		 * \param title the title of the view.
		 * \param view a pointer to the view.		
		 * \param res a custom resolution used for the internal rendering and display. If null, the default value is used.
		 * \param flags ImGui_WindowFlags to pass to the internal window manager.
		 */
		void	addSubView(const std::string& title, ViewBase::Ptr view,
						const Vector2u & res = Vector2u(0,0),
						const ImGuiWindowFlags flags = 0);

		/**
		* \brief Register a standard subview (for instance a SceneDebugView). It will be rendered via a call to onRender(Viewport) in an implicit rendertarget managed by the MultiViewBase.
		* \param title the title of the view.
		* \param view a pointer to the view.
		* \param updateFunc the function that will be called to update your view.
		*					It will pass you the view, the correct Input (mouse position
		*					from 0,0 in the top left corner, key presses and mouse clicks
		*					only if the cursor is over the view), and the Viewport in the
		*					OS window.
		* \param res a custom resolution used for the internal rendering and display. If null, the default value is used.
		* \param flags ImGui_WindowFlags to pass to the internal window manager.
		*/
		void	addSubView(const std::string& title, ViewBase::Ptr view,
			const ViewUpdateFunc updateFunc,
			const Vector2u & res = Vector2u(0, 0),
			const ImGuiWindowFlags flags = 0);

		/**
		* \brief Register an IBR subview (for instance an ULRView). It will be rendered via a call to onRenderIBR(rt,cam,dst).
		* \param title the title of the view.
		* \param view a pointer to the view.
		* \param res a custom resolution used for the internal rendering. If null, the default value is used.
		* \param flags ImGui_WindowFlags to pass to the internal window manager.
		*/
		void	addIBRSubView(const std::string& title, ViewBase::Ptr view, 
						const Vector2u & res = Vector2u(0, 0),
						const ImGuiWindowFlags flags = 0);

		/**
		* \brief Register an IBR subview (for instance an ULRView). It will be rendered via a call to onRenderIBR(rt,cam,dst).
		* \param title the title of the view.
		* \param view a pointer to the view.
		* \param updateFunc the function that will be called to update your view.
		*					It will pass you the view, the correct Input (mouse position
		*					from 0,0 in the top left corner, key presses and mouse clicks
		*					only if the cursor is over the view), and the Viewport in the
		*					OS window. You should return the camera to use during rendering.
		* \param res a custom resolution used for the internal rendering. If null, the default value is used.
		* \param flags ImGui_WindowFlags to pass to the internal window manager.
		*/
		void	addIBRSubView(const std::string& title, ViewBase::Ptr view,
			const IBRViewUpdateFunc updateFunc,
			const Vector2u & res = Vector2u(0, 0),
			const ImGuiWindowFlags flags = 0);

		/** Add another multi-view system as a subsystem of this one.
		 * \param title a name for the multiview
		 * \param multiview the multiview system to add as a subview
		 */
		void	addSubMultiView(const std::string & title, MultiViewBase::Ptr multiview);

		/**
		* \param title
		* \return Return viewbase associated with title, will EXIT_ERROR if no view found
		* \note This covers both basic and IBR subviews.
		* \todo Rename without the IBR prefix
		*/
		ViewBase::Ptr &	getIBRSubView(const std::string& title);

		/**
		* \param title
		* \return the Viewport associated with title, will EXIT_ERROR if no viewport found
		* \note This covers both basic and IBR subviews.
		* \todo Rename without the IBR prefix
		*/
		Viewport & getIBRSubViewport(const std::string &title);

		/**
		* \brief Unregister a subview.
		* \param title the title of the view to remove.
		* \return the view removed from the MultiViewManager.
		*/
		ViewBase::Ptr removeSubView(const std::string& title);
	
		/**
		 * \brief Change the rendering mode.
		 * \param mode The rendering mode to use.
		 */
		void renderingMode(const IRenderingMode::Ptr& mode);


		/**
		 * \brief Define the default rendering and display size for new subviews.
		 * \param size the default size to use.
		 */
		void setDefaultViewResolution(const Vector2i& size);

		/**
		 * \brief Returns the default viewport used for subviews rendering.
		 * \return the current default subview viewport
		 */
		const Viewport getViewport(void) const;

		/**
		 * \brief Returns the last frame time.
		 * \return the last frame time.
		 */
		const float & deltaTime() const { return _deltaTime; }

		/**
		 * \brief Add a camera handler that will automatically be updated and used by the MultiViewManager for the given subview.
		 * \param name the name of the subview to which the camera should be associated.
		 * \param cameraHandler a pointer to the camera handler to register.
		 */
		void addCameraForView(const std::string & name, ICameraHandler::Ptr cameraHandler);

		/**
		* \brief Register a function performing additional rendering for a given subview, 
		* called by the MultiViewManager after calling onRender() on the subview.
		* \param name the name of the subview to which the function should be associated.
		* \param renderFunc the function performing additional rendering..
		*/
		void addAdditionalRenderingForView(const std::string & name, const AdditionalRenderFunc renderFunc);

		/**
		* \brief Count NOT recursively the number of subviews.
		*/
		int numSubViews() const;

		/** Place all subviews on a regular grid in the given viewport.
		 * \param vp the region in which the views should be layed out.
		 */
		void mosaicLayout(const Viewport & vp);

		/** Toggle the display of sub-managers GUIs. */
		void toggleSubViewsGUI();

		/**
		* \brief Set the export path.
		* \param path path to the directory to use.
		*/
		void setExportPath(const std::string & path);
		/**
		* \brief captures a View content into an image file.
		* \param subviewName a string with the name of the subview.
		* \param path the path to save the output.
		* \param filename the name of the output file, needs to have an OpenCV compatible file type.
		*/
		void captureView(const std::string& subviewName, const std::string& path = "./screenshots", const std::string& filename = "");
	protected:

		/** Internal representation of a subview.
		 * Note: this representation should remain *internal* to the multi view system, avoid any abstraction leak.
		 */
		struct SubView {
			ViewBase::Ptr view; ///< Pointer to the view.
			RenderTargetRGB::Ptr rt; ///< Destination RT.
			ICameraHandler::Ptr handler; ///< Potential camera handler.
			AdditionalRenderFunc renderFunc; ///< Optional additonal rendering function.
			sibr::Viewport viewport; ///< Viewport in the global window.
			ImGuiWindowFlags flags = 0; ///< ImGui flags.
			bool shouldUpdateLayout = false; ///< Should the layout be updated at the next frame.

			/// Default constructor.
			SubView() = default;

			/// Destructor.
			virtual ~SubView() = default;
			
			/** Constructor.
			 *\param view_ the view
			 *\param rt_ the destination RT
			 *\param viewport_ the viewport
			 *\param name_ the view name
			 *\param flags_ the ImGui flags
			 */
			SubView(ViewBase::Ptr view_, RenderTargetRGB::Ptr rt_, const sibr::Viewport viewport_, 
				const std::string & name_, const ImGuiWindowFlags flags_);

			/** Render the subview.
			 *\param rm the rendering mode to use
			 *\param renderViewport the viewport to use in the destination RT
			 */
			virtual void render(const IRenderingMode::Ptr & rm, const Viewport & renderViewport) const = 0;
		};

		/** Specialization of Subview for basic views. */
		struct BasicSubView final : SubView {
			ViewUpdateFunc updateFunc; ///< The update function.

			/// Default constructor.
			BasicSubView() : SubView() {};

			/// Destructor.
			virtual ~BasicSubView() = default;
			
			/** Constructor.
			 *\param view_ the view
			 *\param rt_ the destination RT
			 *\param viewport_ the viewport
			 *\param name_ the view name
			 *\param flags_ the ImGui flags
			 *\param f_ the update function
			 */
			BasicSubView(ViewBase::Ptr view_, RenderTargetRGB::Ptr rt_, const sibr::Viewport viewport_, 
				const std::string & name_, const ImGuiWindowFlags flags_, ViewUpdateFunc f_);

			/** Render the subview.
			 *\param rm the rendering mode to use (unused)
			 *\param renderViewport the viewport to use in the destination RT
			 */
			void render(const IRenderingMode::Ptr & rm, const Viewport & renderViewport) const override;
		};

		/** Specialization of Subview for views using a render mode (IBR views mainly). */
		struct IBRSubView final : SubView {
			IBRViewUpdateFunc updateFunc; ///< The update function.
			sibr::InputCamera cam; ///< The current camera.
			bool defaultUpdateFunc = true; ///< Was the default update function used.

			/// Default constructor.
			IBRSubView() : SubView() {};

			/// Destructor.
			virtual ~IBRSubView() = default;
			
			/** Constructor.
			 *\param view_ the view
			 *\param rt_ the destination RT
			 *\param viewport_ the viewport
			 *\param name_ the view name
			 *\param flags_ the ImGui flags
			 *\param f_ the update function
			 *\param defaultUpdateFunc_ was the default update function use (to avoid some collisions)
			 */
			IBRSubView(ViewBase::Ptr view_, RenderTargetRGB::Ptr rt_, const sibr::Viewport viewport_, 
				const std::string & name_, const ImGuiWindowFlags flags_, IBRViewUpdateFunc f_, const bool defaultUpdateFunc_);

			/** Render the subview.
			 *\param rm the rendering mode to use
			 *\param renderViewport the viewport to use in the destination RT
			 */
			 void render(const IRenderingMode::Ptr & rm, const Viewport & renderViewport) const override;
		};

	protected:

		/** Helper to add an IBR subview.
		* \param title the title of the view.
		* \param view a pointer to the view.
		* \param updateFunc the function that will be called to update your view.
		*					It will pass you the view, the correct Input (mouse position
		*					from 0,0 in the top left corner, key presses and mouse clicks
		*					only if the cursor is over the view), and the Viewport in the
		*					OS window. You should return the camera to use during rendering.
		* \param res a custom resolution used for the internal rendering. If null, the default value is used.
		* \param flags ImGui_WindowFlags to pass to the internal window manager.
		* \param defaultFuncUsed a flag denoting if the default function had to be used
		* */
		void addIBRSubView(const std::string & title, ViewBase::Ptr view, 
											const IBRViewUpdateFunc updateFunc, const Vector2u & res, 
											const ImGuiWindowFlags flags, const bool defaultFuncUsed);

		/** Perform rendering for a given subview.
		 *\param subview the subview to render
		 **/
		void renderSubView(SubView & subview);

		/** Capture a view as an image on disk.
		 *\param view the view to capture
		 *\param path the destination direcotry path
		 *\param filename an optional filename, a timestamp will be appended
		 *\note if the filename is empty, the name of the view is used, with a timestamp appended.
		 **/
		static void captureView(const SubView & view, const std::string & path = "./screenshots/", const std::string & filename = "");
		
		IRenderingMode::Ptr _renderingMode = nullptr; ///< Rendering mode.
		std::map<std::string, BasicSubView> _subViews; ///< Regular subviews.
		std::map<std::string, IBRSubView> _ibrSubViews; ///< IBR subviews.
		std::map<std::string, std::shared_ptr<MultiViewBase> > _subMultiViews; ///< Nested multi-views.

		Vector2i _defaultViewResolution; ///< Default view resolution.

		std::string _exportPath; ///< Capture output path.
		std::vector<cv::Mat> _videoFrames; ///< Video frames.

		std::chrono::time_point<std::chrono::steady_clock> _timeLastFrame; ///< Last frame time point.
		float _deltaTime; ///< Elapsed time.
		bool _showSubViewsGui = true; ///< Show the GUI of the subviews.
		bool _onPause = false; ///< Paused interaction and update.
		bool _enableGUI = true; ///< Should the GUI be enabled.
	};

	/** A multiview manager is a multi-view system that displays its subviews in an OS window.
	use it as a based for applications with multiple subviews.
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT MultiViewManager : public MultiViewBase
	{
	public:
		/*
		 * \brief Creates a MultiViewManager in a given OS window.
		 * \param window The OS window to use.
		 * \param resize Should the window be resized by the manager to maximize usable space.
		 */
		MultiViewManager(Window& window, bool resize = true);

		/**
		 * \brief Update subviews and the MultiViewManager.
		 * \param input The Input state to use.
		 */
		void	onUpdate(Input & input) override;

		/**
		 * \brief Render the content of the MultiViewManager and its interface
		 * \param win The OS window into which the rendering should be performed.
		 */
		void	onRender(Window& win) override;

		/**
		 * \brief Render menus and additional gui
		 * \param win The OS window into which the rendering should be performed.
		 */
		void	onGui(Window& win) override;

	private:

		/** Show/hide the GUI. */
		void toggleGUI();

		Window& _window; ///< The OS window.
		FPSCounter _fpsCounter; ///< A FPS counter.
		bool _showGUI = true; ///< Should the GUI be displayed.

	};

	///// INLINE /////

	inline void MultiViewBase::setDefaultViewResolution(const Vector2i& size) {
		_defaultViewResolution = size;
	}

	

} // namespace sibr
