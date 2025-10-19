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



#include "core/graphics/RenderUtility.hpp"
#include "core/view/RenderingMode.hpp"
#include "core/assets/Resources.hpp"
#include "core/graphics/Image.hpp"

namespace sibr
{
	MonoRdrMode::MonoRdrMode( void )
	{
		_clear = true;
		_quadShader.init("Texture",
			sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("texture.vp")),
			sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("texture.fp")));
	}

	void	MonoRdrMode::render( ViewBase& view, const sibr::Camera& eye, const sibr::Viewport& viewport, IRenderTarget* optDest )
	{
		/// TODO: clean everything. Resolution handling.

		//int w = (int)viewport.finalWidth();
		//int h = (int)viewport.finalHeight();

		//if (!_destRT || _destRT->w() != w || _destRT->h() != h)
		//	_destRT.reset( new RenderTarget(w, h) );
		//
		//view.onRenderIBR(*_destRT, eye);
		//_destRT->unbind();

		//_quadShader.begin();
		////if(_ibr->isPortraitAcquisition() && !_ibr->args().fullscreen)
		////	glViewport(0,0, _h, _w);
		////else
		////	glViewport(0,0, _w * _ibr->args().rt_factor, (_ibr->args().fullscreen ? screenHeight : _h) * _ibr->args().rt_factor);
		//viewport.use();
		////glViewport(0,0, size().x(), size().y());

		//glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _destRT->texture());
		//RenderUtility::renderScreenQuad(false /*_ibr->isPortraitAcquisition()*/);
		//_quadShader.end();

		int w = (int)viewport.finalWidth();
		int h = (int)viewport.finalHeight();

		if (!_destRT)// || _destRT->w() != w || _destRT->h() != h)
			_destRT.reset( new RenderTarget(w, h, SIBR_GPU_LINEAR_SAMPLING) );
		glViewport(0, 0, w, h);
		_destRT->bind();

		if( _clear ) {
			viewport.clear();
			// blend with previous
			view.preRender(*_destRT);
		}
		else {
			// can come from somewhere else
			view.preRender(*_prevR);
		}

		view.onRenderIBR(*_destRT, eye);
		_destRT->unbind();

		//show(*_destRT, "before");

		//glEnable (GL_BLEND);
		//glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glDisable (GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		//glDepthMask(GL_FALSE);

		//glEnable (GL_BLEND);
		//glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		_quadShader.begin();
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _destRT->texture());

		if (optDest) // Optionally you can render to another RenderTarget
		{
			glViewport(0, 0, optDest->w(), optDest->h());
			optDest->bind();
		}
		else
		{
			viewport.bind();
		}

		RenderUtility::renderScreenQuad(/*_ibr->isPortraitAcquisition()*/);

		if (optDest) // Optionally you can render to another RenderTarget
			optDest->unbind();

		_quadShader.end();

#if 0
std::cerr <<"End of render pass 1" << std::endl;
		show(*(_destRT));
#endif

	}

	StereoAnaglyphRdrMode::StereoAnaglyphRdrMode( void )
	{
		_clear = true;
		_stereoShader.init("StereoAnaglyph",
			sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("anaglyph.vp")),
			sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("anaglyph.fp")));
		_leftRT.reset(), _rightRT.reset();
/* default values good for Street-10 scene */

		_focalDist = 100;
		_eyeDist = (float)0.065; /* meters */
	}

	void	StereoAnaglyphRdrMode::render( ViewBase& view, const sibr::Camera& eye, const sibr::Viewport& viewport, IRenderTarget* optDest)
	{
		int w = (int)viewport.finalWidth();
		int h = (int)viewport.finalHeight();

		if (!_leftRT || _leftRT->w() != w || _leftRT->h() != h)
			_leftRT.reset( new RenderTarget(w, h) );
		if (!_rightRT || _rightRT->w() != w || _rightRT->h() != h)
			_rightRT.reset( new RenderTarget(w, h) );

		InputCamera leye(eye, w, h);
		InputCamera reye(eye, w, h);
		leye.size(w, h); reye.size(w, h);
		leye.position(eye.position()-_eyeDist*eye.right());

		// setup left eye
		leye.setStereoCam(true, _focalDist, _eyeDist);
		_leftRT->bind();
		if( _clear )  {
			viewport.clear();
			view.preRender(*_leftRT);
		}
		else {
			// can come from somewhere else
			view.preRender(*_prevL);
		}

		view.onRenderIBR(*_leftRT, leye);
		_leftRT->unbind();

		// setup right eye
		reye.position(eye.position()+_eyeDist*eye.right());
		reye.setStereoCam(false, _focalDist, _eyeDist);

		// render right eye
		_rightRT->bind();
		if( _clear ) {
			viewport.clear();
			view.preRender(*_rightRT);
		}
		else {
			// can come from somewhere else
			view.preRender(*_prevR);
		}
		view.onRenderIBR(*_rightRT, reye);
		_rightRT->unbind();

		glDisable (GL_BLEND);
		glDisable(GL_DEPTH_TEST);

		_stereoShader.begin();
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _leftRT->texture());
		glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, _rightRT->texture());

		if (optDest) // Optionally you can render to another RenderTarget
		{
			glViewport(0, 0, optDest->w(), optDest->h());
			optDest->bind();
		}

		RenderUtility::renderScreenQuad();

		if (optDest) // Optionally you can render to another RenderTarget
			optDest->unbind();

		_stereoShader.end();

	}

} // namespace sibr
