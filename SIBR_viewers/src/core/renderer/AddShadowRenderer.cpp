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



#include <core/renderer/AddShadowRenderer.hpp>

namespace sibr { 
	AddShadowRenderer::AddShadowRenderer( void )
	{
		_shader.init("AddShadowShader",
			sibr::loadFile(sibr::getShadersDirectory("core") + "/texture.vert"),
			sibr::loadFile(sibr::getShadersDirectory("core") + "/addshadow.frag"));
		_paramInvProj.init(_shader, "in_inv_proj");
		_paramImgSize.init(_shader, "in_image_size");
	}

	void	AddShadowRenderer::process(
		uint backgroundTextureID,
		uint foregroundTextureID,
		const Vector2f& textureSize,
		const Camera& camera,

		IRenderTarget& dst )
	{
		dst.bind();

		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, foregroundTextureID );
		glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, backgroundTextureID );
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
		glDepthMask(GL_TRUE);		// but write the current values
		_shader.begin();
		_paramInvProj.set(camera.invViewproj());
		_paramImgSize.set(textureSize);
		RenderUtility::renderScreenQuad();
		_shader.end();

		dst.unbind();
	}

} /*namespace sibr*/ 
