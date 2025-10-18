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


# include <core/renderer/NormalRenderer.hpp>
# include "core/graphics/RenderUtility.hpp"

# define USE_PIXELART_MODEN 0 // just for fun (and e-art!)

namespace sibr
{

	NormalRenderer::~NormalRenderer() {};

	NormalRenderer::NormalRenderer(int w, int h, bool generate, bool useFloats, bool imSpace)
	{
		_generate = generate;
		_useFloats = useFloats;

		if (_generate) {

			_normalShader.init("NormalShader",
				sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("normalRendererGen.vp")),
				sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("normalRenderer.fp")),
				sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("normalRendererGen.gp")));

			_normalShader_projInv.init(_normalShader, "MVPinv");
		}
		else {

			_normalShader.init("NormalShader",
				sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("normalRenderer.vp")),
				sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("normalRenderer.fp")));
		}

		_normalShader_proj.init(_normalShader, "MVP");
		_normalShader_view.init(_normalShader, "V");
		_normalShader_model.init(_normalShader, "M");
		_normalShader_imSpace.init(_normalShader, "imSpaceNormals");
		_normalShader.begin();
		_normalShader_imSpace.set(imSpace);
		_normalShader.end();

		if (_useFloats) {
			_normal_RT_32F.reset(new sibr::RenderTargetRGBA32F(w, h));
		} else {
			_normal_RT.reset(new sibr::RenderTargetRGB(w, h));
		}
		

	}

	void NormalRenderer::setWH(int w, int h) {
		if (_useFloats) {
			_normal_RT_32F.reset(new sibr::RenderTargetRGBA32F(w, h));
		}
		else {
			_normal_RT.reset(new sibr::RenderTargetRGB(w, h));
		}
	}

	void NormalRenderer::render(const sibr::InputCamera& cam, const Mesh& mesh, const Matrix4f &modelMat, bool clear)
	{
#if USE_PIXELART_MODEN
		glPointSize(10.f);
#else
		glPointSize(2.f);
#endif

		if (_useFloats) {
			if(clear)
				_normal_RT_32F->clear(sibr::Vector4f(0.5f,0.5f,0.5f,1.0f));
			glViewport(0, 0, _normal_RT_32F->w(), _normal_RT_32F->h());
			_normal_RT_32F->bind();
		} else {
			_normal_RT->bind();
			if (clear) {
				glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			}
			glViewport(0, 0, _normal_RT->w(), _normal_RT->h());
			glScissor(0, 0, _normal_RT->w(), _normal_RT->h());
		}

		_normalShader.begin();
		const Matrix4f MVP = cam.viewproj() * modelMat;
		_normalShader_proj.set(MVP);
		_normalShader_view.set(cam.view());
		_normalShader_model.set(modelMat);

		if (_generate) {
			const Matrix4f MVPinv = (cam.viewproj()*modelMat).inverse();
			_normalShader_projInv.set(MVPinv);
		}

	//	std::cout << cam.znear() << " " << cam.zfar() << " " << cam.viewproj() << std::endl;
		mesh.render(true, true, sibr::Mesh::FillRenderMode);

		_normalShader.end();

	}

} // namespace