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


#include "BinaryMeshRenderer.hpp"

namespace sibr {

	BinaryMeshRenderer::BinaryMeshRenderer()
	{
		std::string vertex_shader =
			SIBR_SHADER(420,
				uniform mat4 MVP;
		layout(location = 0) in vec3 in_vertex;
		void main(void) {
			gl_Position = MVP * vec4(in_vertex, 1.0);
		}
		);

		std::string fragment_shader = SIBR_SHADER(420,
			out vec4 out_color;
			uniform float epsilon;
		void main(void) {
			out_color = vec4(1, 1, 1, 1);
			gl_FragDepth = gl_FragCoord.z * (1.0 - epsilon);
		}
		);

		_shader.init("binaryMeshShader", vertex_shader, fragment_shader);
		_paramMVP.init(_shader, "MVP");
		epsilon.init(_shader, "epsilon");
	}

	void BinaryMeshRenderer::process(const Mesh & mesh, const Camera & eye, IRenderTarget & dst)
	{
		dst.bind();
		_shader.begin();
		_paramMVP.set(eye.viewproj());
		epsilon.send();

		mesh.render(true, false);
		
		_shader.end();
		dst.unbind();
	}
}