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


# include "Config.hpp"
# include <core/assets/Resources.hpp>
# include <projects/ulr/renderer/ULRRenderer.hpp>

namespace sibr { 
ULRRenderer::ULRRenderer(const uint w, const uint h)
{
	
    std::cerr << "\n[ULRenderer] initializing" << std::endl;
    std::cerr << "\n[ULRenderer] loading shaders" << std::endl;
    _ulrShaderPass1 .init("ULR1",
			sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr.vert"),
			sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr1.frag"));
    _ulrShaderPass2 .init("ULR2",
			sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr.vert"),
			sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr2.frag"));
    _depthShader.init("Depth",
			sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr_intersect.vert"),
			sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr_intersect.frag"));

    _ulrShaderPass1_nCamPos .init(_ulrShaderPass1, "nCamPos");
    _ulrShaderPass1_iCamPos .init(_ulrShaderPass1, "iCamPos");
    _ulrShaderPass1_iCamDir .init(_ulrShaderPass1, "iCamDir");
    _ulrShaderPass1_iCamProj.init(_ulrShaderPass1, "iCamProj");
    _ulrShaderPass1_occlTest .init(_ulrShaderPass1, "occlTest");
	_ulrShaderPass1_masking .init(_ulrShaderPass1, "doMasking");
    _depthShader_proj.init(_depthShader,"proj");

    std::cerr << "\n[ULRenderer] creating render targets" << std::endl;

    _ulr0_RT .reset(new sibr::RenderTargetRGBA32F(w,h,0,4));
    _ulr1_RT .reset(new sibr::RenderTargetRGBA32F(w,h,0,4));
    _depth_RT.reset(new sibr::RenderTargetRGBA32F(w,h));

	_doOccl = true;
}

void
ULRRenderer::process(std::vector<uint>& imgs_ulr, const sibr::Camera& eye,
		const sibr::BasicIBRScene::Ptr scene,
		std::shared_ptr<sibr::Mesh>& altMesh,
		const std::vector<std::shared_ptr<RenderTargetRGBA32F> >& inputRTs,
		IRenderTarget& dst)
{
	// Get a new camera with z_near ~ 0
	sibr::Camera new_cam = eye;
	new_cam.znear( 0.001f );

    // render geometry to depth map

	glViewport(0,0, _depth_RT->w(), _depth_RT->h());
    _depth_RT->clear();
    _depth_RT->bind();

    _depthShader.begin();
    _depthShader_proj.set(new_cam.viewproj());

	glClear(GL_DEPTH_BUFFER_BIT);

	if( altMesh != nullptr )
		altMesh->render( true, true); // enable depth test - disable back culling
	else
		scene->proxies()->proxy().render( true, true); // enable depth test - disable back culling

    _depthShader.end();
    _depth_RT->unbind();

    // ULR pass 1
    _ulr0_RT->clear(sibr::Vector4f(0,0,0,1e5));
    _ulr1_RT->clear(sibr::Vector4f(0,0,0,1e5));
    for (uint i=0; i<imgs_ulr.size(); i++) {
        if (scene->cameras()->inputCameras()[imgs_ulr[i]]->isActive()) {
			const sibr::InputCamera& cam = *scene->cameras()->inputCameras()[imgs_ulr[i]];
            std::swap(_ulr0_RT, _ulr1_RT);
            _ulrShaderPass1.begin();
            _ulr0_RT->bind();
            glViewport(0,0, _ulr0_RT->w(), _ulr0_RT->h());
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, inputRTs[imgs_ulr[i]]->texture());
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, _depth_RT->texture());
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, _ulr1_RT->texture(0));
            glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, _ulr1_RT->texture(1));
            glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, _ulr1_RT->texture(2));
            glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, _ulr1_RT->texture(3));
			if (useMasks()){
					glActiveTexture(GL_TEXTURE6);
					glBindTexture(GL_TEXTURE_2D, getMasks()[imgs_ulr[i]]->texture());
			}
			_ulrShaderPass1_masking.set(useMasks());
			_ulrShaderPass1_nCamPos.set(eye.position());
            _ulrShaderPass1_iCamPos.set(cam.position());
            _ulrShaderPass1_iCamDir.set(cam.dir());
            _ulrShaderPass1_iCamProj.set(cam.viewproj());
            _ulrShaderPass1_occlTest.set(_doOccl);
			sibr::RenderUtility::renderScreenQuad();
            _ulr0_RT->unbind();
            _ulrShaderPass1.end();

#if 0
			{
				sibr::ImageRGBA32F img2;
				_ulr0_RT->readBack(img2);
				show(img2); // DEBUG
			}
#endif
        }
    }

    // ULR pass 2
    // enable depth test to ensure depth of proxy is written to
    // depth buffer by the shader
//    glEnable(GL_DEPTH_TEST); /// \todo TODO -- breaks with fences -- check
    _ulrShaderPass2.begin();
    dst.clear();
    dst.bind();
    glViewport(0,0, dst.w(), dst.h());
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _depth_RT->texture());
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, _ulr0_RT->texture(0));
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, _ulr0_RT->texture(1));
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, _ulr0_RT->texture(2));
    glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, _ulr0_RT->texture(3));
	sibr::RenderUtility::renderScreenQuad();
    dst.unbind();
    _ulrShaderPass2.end();

#if 0
	sibr::ImageRGB img;
	dst.readBack(img);
	show(img); // DEBUG
#endif

}

} /*namespace sibr*/ 
