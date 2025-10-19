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


# include "ShadowMapRenderer.hpp"
# include "core/graphics/RenderUtility.hpp"

float SUN_APP_DIAM = 0.5358f;

namespace sibr
{

	ShadowMapRenderer::~ShadowMapRenderer() {};

	ShadowMapRenderer::ShadowMapRenderer(const sibr::InputCamera& depthMapCam, std::shared_ptr<sibr::RenderTargetLum32F> depthMap_RT):_depthMap_RT(depthMap_RT)
	{
		_shadowMapShader.init("ShadowMapShader",
			sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("shadowMapRenderer.vp")), 
			sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("shadowMapRenderer.fp")));

		_shadowMapShader_MVP.init(_shadowMapShader,"MVP");
		_depthMap_MVP.init(_shadowMapShader, "depthMapMVP");
		_depthMap_MVPinv.init(_shadowMapShader, "depthMapMVPinv");
		_depthMap_radius.init(_shadowMapShader, "depthMapRadius");
		_lightDir.init(_shadowMapShader, "lightDir");
		_bias_control.init(_shadowMapShader, "biasControl");
		_sun_app_radius.init(_shadowMapShader, "sun_app_radius");

		sibr::Vector3f toLight = -depthMapCam.dir();
		_shadowMapShader.begin();
		_depthMap_MVP.set(depthMapCam.viewproj());
		_depthMap_MVPinv.set(depthMapCam.invViewproj());
		_depthMap_radius.set(depthMapCam.orthoRight());
		_lightDir.set(toLight);
		_sun_app_radius.set(SUN_APP_DIAM/2.0f);
		_shadowMapShader.end();
	}

	void ShadowMapRenderer::render(int w, int h, const sibr::InputCamera& cam, const Mesh& mesh, float bias ) 
	{

		//sibr::Vector1f cc(1.0);
		//_depth_RT->clear(cc);

		_shadowMap_RT.reset(new sibr::RenderTargetLum(w, h));
		
		glViewport(0, 0, _shadowMap_RT->w(), _shadowMap_RT->h());
		_shadowMap_RT->bind();
		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		_shadowMapShader.begin();
		_shadowMapShader_MVP.set(cam.viewproj());
		_bias_control.set(bias);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, _depthMap_RT->texture());

		mesh.render(true, false, sibr::Mesh::FillRenderMode);

		_shadowMapShader.end();

	}

} // namespace