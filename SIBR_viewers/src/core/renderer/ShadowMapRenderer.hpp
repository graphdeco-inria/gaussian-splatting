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

# include <core/renderer/Config.hpp>

# include "core/assets/InputCamera.hpp"
# include "core/graphics/Texture.hpp"
# include "core/graphics/Camera.hpp"
# include "core/graphics/RenderUtility.hpp"
# include "core/assets/Resources.hpp"
# include "core/graphics/Shader.hpp"
# include "core/graphics/Mesh.hpp"


namespace sibr
{

	/** Render high quality soft shadows, designed to mimick the sun shadowing.
	\note Soft shadowing require a lot of texture fetches that can impact performances.
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT ShadowMapRenderer
	{

	public:

		/** Constructor.
		\param depthMapCam the light viewpoint
		\param depthMap_RT depth map rendered from the light viewpoint
		*/
		ShadowMapRenderer(const sibr::InputCamera& depthMapCam, std::shared_ptr<sibr::RenderTargetLum32F> depthMap_RT)  ;
		
		/// Destructor.
		~ShadowMapRenderer();

		/**
		Render soft sun shadows on the mesh, using the precomputed depth map.
		\param w the target width
		\param h the target height
		\param cam the viewpoint to use
		\param mesh the mesh to render
		\param bias shadow acne bias
		*/
		void render(int w, int h,const sibr::InputCamera &cam, const Mesh& mesh, float bias= 0.0005f);

		std::shared_ptr<sibr::RenderTargetLum> _shadowMap_RT; ///< Result containing the soft shadows.

		std::shared_ptr<sibr::RenderTargetLum32F> _depthMap_RT; ///< Depth map rendered from the light viewpoint.

		
	private:

		sibr::GLShader				_shadowMapShader; ///< Shadow rendering.
		sibr::GLParameter			_shadowMapShader_MVP; ///< Final MVP uniform.
		sibr::GLParameter			_depthMap_MVP; ///< Light MVP uniform.
		sibr::GLParameter			_depthMap_MVPinv; ///< Light inverse MVP uniform.
		sibr::GLParameter			_depthMap_radius; ///< Depth map radius.
		sibr::GLParameter			_lightDir; ///< Light direction uniform.
		sibr::GLParameter			_bias_control; ///< Bias uniform.
		sibr::GLParameter			_sun_app_radius; ///< Sun radius (for soft shadows).
		std::shared_ptr<sibr::Texture2DLum32F> _textureDepthMap; ///< Shadow map target (unused).

	};

} // namespace

