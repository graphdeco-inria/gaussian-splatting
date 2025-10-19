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
# include "core/scene/BasicIBRScene.hpp"
# include <core/renderer/CopyRenderer.hpp>
# include <projects/ulr/renderer/ULRV2Renderer.hpp>
# include <core/renderer/PoissonRenderer.hpp>

namespace sibr { 

	/** View associated to ULRRenderer v2, providing interface and options. */
	class SIBR_EXP_ULR_EXPORT ULRV2View : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(ULRV2View);

		/** Camera selection mode. */
		enum class RenderMode { NORMAL = 0, ONLY_ONE_CAM = 1, LEAVE_ONE_OUT = 2 };

	public:

		/** Constructor.
		 *\param ibrScene the scene
		 *\param render_w rendering width
		 *\param render_h rendering height
		 **/
		ULRV2View( const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h );

		/** Destructor. */
		~ULRV2View();

		/** Render using the ULR algorithm.
		 *\param dst destination target
		 *\param eye novel viewpoint
		 **/
		virtual void onRenderIBR( sibr::IRenderTarget& dst, const sibr::Camera& eye );

		/** Update state absed on user inputs.
		 *\param input the view input
		 **/
		virtual void onUpdate(Input& input);

		/** Display GUI. */
		virtual void onGUI() override;

		/** Select input cameras to use for rendering.
		 *\param eye the current viewpoint
		 *\return a list of camera indices.
		 **/
		virtual std::vector<uint> chosen_cameras(const sibr::Camera& eye) ;

		/** Select input cameras to use for rendering, based only on distance.
		 *\param eye the current viewpoint
		 *\return a list of camera indices.
		 **/
		virtual std::vector<uint> chosen_camerasNew(const sibr::Camera& eye);

		/** Select input cameras to use for rendering.
		 *\param eye the current viewpoint
		 *\return a list of camera indices.
		 **/
		virtual std::vector<uint> chosen_cameras_angdist(const sibr::Camera& eye);

		/** Set the altMesh and use instead of scene proxy.
		 *\param m mesh to use
		 **/
		void	altMesh(std::shared_ptr<sibr::Mesh> m)	{ _altMesh = m; }

		/** Toggle occlusion testing.
		 *\param val should occlusion testing be performed
		 */
		void	doOccl(bool val) { _ulr->doOccl(val); }

		/** \return a pointer to the alt mesh if it exists */
		std::shared_ptr<sibr::Mesh> 	altMesh()	{ return _altMesh; }

		/** Set the number of cmaeras to select for blending.
		 *\param dist number of cameras for the distance criterion
		 *\param angle number of cameras for the angle criterion
		 **/
		void	setNumBlend(short int dist, short int angle);

		/** Set the input RGBD textures.
		 *\param iRTs the new textures to use.
		 */
		void	inputRTs(const std::vector<std::shared_ptr<RenderTargetRGBA32F> >& iRTs) { _inputRTs = iRTs;}

		/** Set the masks for ignoring some regions of the input images.
		 *\param masks the new masks
		 **/
		void	setMasks( const std::vector<RenderTargetLum::Ptr>& masks );

		/** Load masks from disk.
		 *\param ibrScene the scene
		 *\param w resolution width
		 *\param h resolution height
		 *\param maskDir masks directory path
		 *\param preFileName mask files prefix
		 *\param postFileName mask files suffix and extension
		 */
		void	loadMasks(
			const sibr::BasicIBRScene::Ptr& ibrScene, int w, int h,
			const std::string& maskDir = "",
			const std::string& preFileName = "",
			const std::string& postFileName = ""
		);

		/** Set the camera selection mode.
		 *\param mode the new mode. 
		 */
		void		setRenderMode(RenderMode mode) { _renderMode = mode; }
		/** \return the camera selection mode. */
		RenderMode	getRenderMode() const { return _renderMode; }

		/** Set the view ID when in single view mode.
		 *\param id the camera id to use
		 */
		void		setSingleViewId(int id) { _singleCamId = id; }
		/** \return the current selected camera ID in single view mode. */
		int			getSingleViewId(void)  const { return _singleCamId; }

		/** Toggle poisson blending.
		 *\param val if true, Poisson blending is disabled.
		 */
		void noPoissonBlend(bool val) { _noPoissonBlend = val; }
		/** \return true if pOisson blending is disabled. */
		bool noPoissonBlend() const { return _noPoissonBlend; }

		/** Compute soft visibility map.
		 *\param depthMap view depth map
		 *\param out will contain the soft visibility map
		 */
		void computeVisibilityMap(const sibr::ImageL32F & depthMap, sibr::ImageRGBA & out);

		/** \return a pointer to the scene */
		const std::shared_ptr<sibr::BasicIBRScene> & getScene() const { return _scene; }

	public:
		ULRV2Renderer::Ptr		_ulr; ///< ULRV2 renderer.
		PoissonRenderer::Ptr	_poisson; ///< Poisson filling renderer.

	protected:
		
		std::shared_ptr<sibr::BasicIBRScene> _scene; ///< the current scene.
		std::shared_ptr<sibr::Mesh>	_altMesh; ///< For the cases when using a different mesh than the scene
		int _numDistUlr, _numAnglUlr; ///< Number of cameras to select for each criterion.

		std::vector<std::shared_ptr<RenderTargetRGBA32F> > _inputRTs; ///< input RTs -- usually RGB but can be alpha or other

		bool _noPoissonBlend = false; ///< Runtime status of the poisson blend.

		RenderTargetRGBA::Ptr _blendRT; ///< ULR destination RT.
		RenderTargetRGBA::Ptr _poissonRT; ///< Poisson filling destination RT.

		RenderMode _renderMode; ///< Current camera selection mode.
		int _singleCamId; ///< Selected camera in single view mode.

		bool testAltlULRShader; ///< TT: to switch with alternate shader with tab
	};

} /*namespace sibr*/ 
