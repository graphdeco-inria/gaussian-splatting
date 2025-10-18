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



#include <core/renderer/PointBasedRenderer.hpp>

namespace sibr { 
	PointBasedRenderer::PointBasedRenderer()
	{	
		_shader.init("PointBased",
			sibr::loadFile(sibr::getShadersDirectory("core") + "/alpha_points.vert"),
			sibr::loadFile(sibr::getShadersDirectory("core") + "/alpha_points.frag"));
		_paramMVP.init(_shader,"mvp");
		_paramAlpha.init(_shader,"alpha");
		_paramRadius.init(_shader,"radius");
		_paramUserColor.init(_shader,"user_color");
	}

	void PointBasedRenderer::meshToDevice(const Mesh& mesh)
	{
		mesh.forceBufferGLUpdate();
	}

	void	PointBasedRenderer::process(const Mesh& mesh, const Camera& eye, IRenderTarget& dst, bool backfaceCull)
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_PROGRAM_POINT_SIZE);
		dst.bind();
		_shader.begin();
		_paramMVP.set(eye.viewproj());
		_paramAlpha.set(float(1.0));
		_paramRadius.set(3);
		_paramUserColor.set(Vector3f(.1, .1, 1.0));

		mesh.render_points();
		_shader.end();
		dst.unbind();
		glDisable(GL_PROGRAM_POINT_SIZE);
		glDisable(GL_DEPTH_TEST);
	}

	void	PointBasedRenderer::process(const Mesh& mesh, const Camera& eye, const sibr::Matrix4f& model, IRenderTarget& dst, bool backfaceCull)
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_PROGRAM_POINT_SIZE);
		dst.bind();
		_shader.begin();
		_paramMVP.set(sibr::Matrix4f(eye.viewproj() * model));
		_paramAlpha.set(float(1.0));
		_paramRadius.set(2);
		_paramUserColor.set(Vector3f(.1, .1, 1.0));
		mesh.render_points();
		_shader.end();
		dst.unbind();
		glDisable(GL_PROGRAM_POINT_SIZE);
		glDisable(GL_DEPTH_TEST);
	}

} /*namespace sibr*/
