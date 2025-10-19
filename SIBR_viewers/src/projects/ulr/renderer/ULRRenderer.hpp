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
# include <core/graphics/Texture.hpp>
# include <core/graphics/Shader.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/renderer/RenderMaskHolder.hpp>
# include <core/scene/BasicIBRScene.hpp>

namespace sibr { 

	/** Legacy ULR renderer. Process each input image separately and accumulate them.
	 **/
	class SIBR_EXP_ULR_EXPORT ULRRenderer : public RenderMaskHolder
	{
		SIBR_CLASS_PTR(ULRRenderer);

		/** Constructor.
		 *\param w rendering width
		 *\param h rendering height
		 */
		ULRRenderer(const uint w, const uint h);

		/** Render.
		 *\param imgs_ulr vector of selected image IDs
		 *\param eye novel viewpoint
		 *\param scene the scene to render
		 *\param altMesh optional alternative mesh
		 *\param inputRTs the RGBD input images
		 *\param output destination target
		 */
		void process(std::vector<uint>& imgs_ulr, const sibr::Camera& eye,
			const sibr::BasicIBRScene::Ptr scene,
			std::shared_ptr<sibr::Mesh>& altMesh,
			const std::vector<std::shared_ptr<RenderTargetRGBA32F> >& inputRTs,
			IRenderTarget& output);

		/** Toggle occlusion testing.
		 *\param val should occlusion testing be performed
		 */
		void doOccl(bool val) { _doOccl = val; }

	private:
		sibr::RenderTargetRGBA32F::Ptr _ulr0_RT;
		sibr::RenderTargetRGBA32F::Ptr _ulr1_RT;
		sibr::RenderTargetRGBA32F::Ptr _depth_RT;

		sibr::GLShader _ulrShaderPass1;
		sibr::GLShader _ulrShaderPass2;
		sibr::GLShader _depthShader;

		sibr::GLParameter _ulrShaderPass1_nCamPos;
		sibr::GLParameter _ulrShaderPass1_iCamPos;
		sibr::GLParameter _ulrShaderPass1_iCamDir;
		sibr::GLParameter _ulrShaderPass1_iCamProj;
		sibr::GLParameter _ulrShaderPass1_occlTest;
		sibr::GLParameter _ulrShaderPass1_masking;
		sibr::GLParameter _depthShader_proj;

		bool	_doOccl;

   };

} /*namespace sibr*/
