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
# include <core/scene/BasicIBRScene.hpp>
# include <core/renderer/CopyRenderer.hpp>
# include <projects/ulr/renderer/ULRRenderer.hpp>
# include <core/view/ViewBase.hpp>

namespace sibr { 

	/** View associated to ULRRenderer v1, providing interface and options. */
	class SIBR_EXP_ULR_EXPORT ULRView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR( ULRView );

	public:

		/** Constructor.
		 *\param ibrScene the scene
		 *\param render_w rendering width
		 *\param render_h rendering height
		 **/
		ULRView( const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h );

		/** Destructor. */
		~ULRView();

		/** Render using the ULR algorithm. 
		 *\param dst destination target
		 *\param eye novel viewpoint
		 **/
		virtual void onRenderIBR( sibr::IRenderTarget& dst, const sibr::Camera& eye );

		/** Select input cameras to use for rendering.
		 *\param eye the current viewpoint
		 *\return a list of camera indices.
		 **/
		virtual std::vector<uint> chosen_cameras(const sibr::Camera& eye) ;

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
		void	setNumBlend(short int dist, short int angle) { _numDistUlr = dist, _numAnglUlr = angle; }

		/** Set the input RGBD textures.
		 *\param iRTs the new textures to use. 
		 */
		void	inputRTs(const std::vector<std::shared_ptr<RenderTargetRGBA32F> >& iRTs) { _inputRTs = iRTs;}

		/** Set the masks for ignoring some regions of the input images.
		 *\param masks the new masks
		 **/
		void	setMasks( const std::vector<RenderTargetLum::Ptr>& masks);

	protected:

		ULRRenderer::Ptr		_ulr; ///< Renderer.
		std::shared_ptr<sibr::BasicIBRScene> _scene; ///< Scene.
		std::shared_ptr<sibr::Mesh>	_altMesh; ///< For the cases when using a different mesh than the scene
		short int _numDistUlr, _numAnglUlr; ///< max number of selected cameras for each criterion.
		std::vector<std::shared_ptr<RenderTargetRGBA32F> > _inputRTs; ///< input RTs -- usually RGB but can be alpha or other

	};

} /*namespace sibr*/ 
