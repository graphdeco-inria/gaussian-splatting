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

# include "Config.hpp"
# include <core/system/Config.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/view/ViewBase.hpp>
# include <core/renderer/CopyRenderer.hpp>
# include <projects/ulr/renderer/ULRV3Renderer.hpp>
# include <core/renderer/PoissonRenderer.hpp>

namespace sibr { 

	/**
	 * \class ULRV3View
	 * \brief Wrap a ULR renderer with additional parameters and information.
	 */
	class SIBR_EXP_ULR_EXPORT ULRV3View : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(ULRV3View);

		/// Rendering mode: default, use only one camera, use all cameras but one.
		enum RenderMode { ALL_CAMS, ONE_CAM, LEAVE_ONE_OUT, EVERY_N_CAM };

		/// Blending mode: keep the four best values per pixel, or aggregate them all.
		enum WeightsMode { ULR_W , VARIANCE_BASED_W, ULR_FAST};

	public:

		/**
		 * Constructor
		 * \param ibrScene The scene to use for rendering.
		 * \param render_w rendering width
		 * \param render_h rendering height
		 */
		ULRV3View(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h);

		/** Replace the current scene.
		 *\param newScene the new scene to render */
		void setScene(const sibr::BasicIBRScene::Ptr & newScene);

		/**
		 * Perform rendering. Called by the view manager or rendering mode.
		 * \param dst The destination rendertarget.
		 * \param eye The novel viewpoint.
		 */
		void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;

		/**
		 * Update inputs (do nothing).
		 * \param input The inputs state.
		 */
		void onUpdate(Input& input) override;

		/**
		 * Update the GUI.
		 */
		void onGUI() override;

		/** \return a reference to the renderer. */
		const ULRV3Renderer::Ptr & getULRrenderer() const { return _ulrRenderer; }

		/** Set the renderer blending weights mode.
		 *\param mode the new mode to use
		 *\sa WeightsMode
		 **/
		void setMode(const WeightsMode mode);

		/** \return a reference to the scene */
		const std::shared_ptr<sibr::BasicIBRScene> & getScene() const { return _scene; }

	protected:

		/**
		 * Update the camera informations in the ULR renderer based on the current rendering mode and selected index.
		 * \param allowResetToDefault If true, when the rendering mode is ALL_CAMS, the cameras information will be updated.
		 */
		void updateCameras(bool allowResetToDefault);

		std::shared_ptr<sibr::BasicIBRScene> _scene; ///< The current scene.
		ULRV3Renderer::Ptr		_ulrRenderer; ///< The ULR renderer.
		PoissonRenderer::Ptr	_poissonRenderer; ///< The poisson filling renderer.

		RenderTargetRGBA::Ptr	_blendRT; ///< ULR destination RT.
		RenderTargetRGBA::Ptr	_poissonRT; ///< Poisson filling destination RT.

		bool					_poissonBlend = false; ///< Should Poisson filling be applied.

		RenderMode				_renderMode = ALL_CAMS; ///< Current rendering mode.
		WeightsMode				_weightsMode = ULR_W; ///< Current blend weights mode.
		int						_singleCamId = 0; ///< Selected camera for the single view mode.
		int						_everyNCamStep = 1; ///< Camera step size for the every other N mode.
	};

} /*namespace sibr*/ 
