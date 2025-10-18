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



#include <core/renderer/BlurRenderer.hpp>

namespace sibr { 
	BlurRenderer::BlurRenderer( void )
	{
		_shader.init("BlurShader",
			sibr::loadFile(sibr::getShadersDirectory("core") + "/texture.vert"),
			sibr::loadFile(sibr::getShadersDirectory("core") + "/blur.frag"));
		_paramImgSize.init(_shader, "in_image_size");
	}

	void	BlurRenderer::process( uint textureID, const Vector2f& textureSize, IRenderTarget& dst )
	{
		dst.bind();

		glDisable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);

		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, textureID );
		glDisable(GL_DEPTH_TEST);
		_shader.begin();
		_paramImgSize.set(textureSize);
		RenderUtility::renderScreenQuad();
		_shader.end();

		dst.unbind();
	}

} /*namespace sibr*/ 
