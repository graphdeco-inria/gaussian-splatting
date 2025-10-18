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



#include <core/renderer/TexturedMeshRenderer.hpp>

namespace sibr { 
	TexturedMeshRenderer::TexturedMeshRenderer( bool flipY )
	{	
		if(flipY)
		_shader.init("TexturedMesh",
			sibr::loadFile(sibr::getShadersDirectory("core") + "/textured_mesh_flipY.vert"),
			sibr::loadFile(sibr::getShadersDirectory("core") + "/textured_mesh.frag"));
		else
		_shader.init("TexturedMesh",
			sibr::loadFile(sibr::getShadersDirectory("core") + "/textured_mesh.vert"),
			sibr::loadFile(sibr::getShadersDirectory("core") + "/textured_mesh.frag"));
		_paramMVP.init(_shader,"MVP");
	}

	void	TexturedMeshRenderer::process(const Mesh& mesh, const Camera& eye, uint textureID, IRenderTarget& dst, bool backfaceCull)
	{
		dst.bind();
		_shader.begin();
		_paramMVP.set(eye.viewproj());
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, textureID);
		mesh.render(true, backfaceCull);
		_shader.end();
		dst.unbind();

	}

	void	TexturedMeshRenderer::process(const Mesh& mesh, const Camera& eye, const sibr::Matrix4f& model, uint textureID, IRenderTarget& dst, bool backfaceCull)
	{
		dst.bind();
		_shader.begin();
		_paramMVP.set(sibr::Matrix4f(eye.viewproj() * model));
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, textureID);
		mesh.render(true, backfaceCull);
		_shader.end();
		dst.unbind();

	}

} /*namespace sibr*/
