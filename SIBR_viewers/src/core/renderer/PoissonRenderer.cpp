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


/**
*
* Poisson synthesis.
*/

#include <utility>
#include <algorithm>

#include <core/assets/Resources.hpp>
#include <core/graphics/RenderUtility.hpp>

#include <core/renderer/PoissonRenderer.hpp>

using namespace sibr;

/** Number of levels poisson multi-grid */
#define POISSON_LEVELS 5

/** Number of relaxation/jacobi iterations at each level */
#define POISSON_ITERATIONS  2

/** Ratio of successive levels of poisson multi-grid */
#define MULTIGRID_SCALE 2

namespace sibr { 
	// -----------------------------------------------------------------------

	PoissonRenderer ::PoissonRenderer ( uint w, uint h ) :
		_size(w, h)
	{
		std::string vp = sibr::loadFile(sibr::getShadersDirectory("core") + "/texture.vert");
		_jacobiShader  .init("Jacobi",  vp, sibr::loadFile(sibr::getShadersDirectory("core") + "/poisson_jacobi.frag"));
		_restrictShader.init("Restrict",vp, sibr::loadFile(sibr::getShadersDirectory("core") + "/poisson_restrict.frag"));
		_interpShader  .init("Interp",  vp, sibr::loadFile(sibr::getShadersDirectory("core") + "/poisson_interp.frag"));
		_divergShader  .init("Diverg",  vp, sibr::loadFile(sibr::getShadersDirectory("core") + "/poisson_diverg.frag"));

		// GLParameters
		_jacobi_weights.init(_jacobiShader,   "weights");
		_jacobi_scale.init(_jacobiShader, "scale");
		_restrict_scale.init(_restrictShader, "scale");
		_interp_scale  .init(_interpShader,   "scale");

		_poisson_div_RT.resize(POISSON_LEVELS);
		for (uint i=0; i<_poisson_div_RT.size(); i++) {
			uint ww = std::max(1u, uint(w/pow( (float)MULTIGRID_SCALE, (int)i)));
			uint hh = std::max(1u, uint(h/pow( (float)MULTIGRID_SCALE, (int)i)));
			_poisson_div_RT[i].reset(new sibr::RenderTargetRGBA(ww,hh, SIBR_CLAMP_UVS));
		}
		_poisson_RT.reset(new sibr::RenderTargetRGBA(w,h, SIBR_CLAMP_UVS | SIBR_GPU_LINEAR_SAMPLING));
		_poisson_tmp_RT.reset(new sibr::RenderTargetRGBA(w,h, SIBR_CLAMP_UVS | SIBR_GPU_LINEAR_SAMPLING));
		_enableFix = true;

	}

	// -----------------------------------------------------------------------

	uint PoissonRenderer::render( uint texture )
	{
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Poisson filling");
		// divergence of gradient map and dirichlet constraints
		_divergShader.begin();
		_poisson_div_RT[0]->clear();
		_poisson_div_RT[0]->bind();
		glViewport(0, 0, _poisson_div_RT[0]->w(), _poisson_div_RT[0]->h());
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texture);
		RenderUtility::renderScreenQuad();
		_poisson_div_RT[0]->unbind();
		_divergShader.end();

		//  restrict the divergence
		for (int k=0; k<int(_poisson_div_RT.size())-1; k++) {
			_restrictShader.begin();
			_poisson_div_RT[k+1]->clear();
			_poisson_div_RT[k+1]->bind();
			glViewport(0,0, _poisson_div_RT[k+1]->w(), _poisson_div_RT[k+1]->h());
			glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _poisson_div_RT[k]->texture());
			_restrict_scale.set(float(_poisson_div_RT[k]->w())/_poisson_div_RT[k+1]->w());
			RenderUtility::renderScreenQuad();
			_poisson_div_RT[k+1]->unbind();
			_restrictShader.end();
		}

		// perform jacobi iterations and upsample the result to higher level
		bool isFirst = _enableFix;
		for (int k=(int)_poisson_div_RT.size()-1; k>=0; k--) {
			for (uint i=0; i<POISSON_ITERATIONS; i++) {
				double h   =  pow( (float)MULTIGRID_SCALE, (int)k);            // Jacobi relaxation filter kernel taken from
				double hsq =  h*h;                               // Real-Time Gradient-Domain Painting, SIGGRAPH '08
				double xh0 = -2.1532 + 1.5070/h + 0.5882/hsq;    // http://graphics.cs.cmu.edu/projects/gradient-paint/
				double xh1 =  0.1138 + 0.9529/h + 1.5065/hsq;
				double xh  =  ((i%2 == 0) ? xh0 : xh1);
				double m   =  (-8*hsq - 4)/(3.0*hsq);
				double e   =  (hsq + 2)/(3.0*hsq);
				double c   =  (hsq - 1)/(3.0*hsq);

				std::swap(_poisson_tmp_RT, _poisson_RT);

				_jacobiShader.begin();
				_poisson_RT->clear();
				_poisson_RT->bind();
				glViewport(0,0, _poisson_div_RT[k]->w(), _poisson_div_RT[k]->h());
				glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _poisson_tmp_RT->texture());
				_jacobi_weights.set((float)xh, (float)e, (float)c, (float)(1.0/(m-xh)));
				_jacobi_scale.set( isFirst ? (float(_poisson_tmp_RT->w()) / _poisson_div_RT[k]->w()) : 1.0f);
				RenderUtility::renderScreenQuad();
				_poisson_RT->unbind();
				_jacobiShader.end();

				isFirst = false;
			}

			if (k > 0) {
				std::swap(_poisson_tmp_RT, _poisson_RT);
				_interpShader.begin();
				_poisson_RT->clear();
				_poisson_RT->bind();
				glViewport(0,0, _poisson_div_RT[k-1]->w(), _poisson_div_RT[k-1]->h());
				glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _poisson_tmp_RT->texture());
				glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, _poisson_div_RT[k-1]->texture());
				_interp_scale.set(float(_poisson_div_RT[k-1]->w()) / _poisson_div_RT[k]->w());
				RenderUtility::renderScreenQuad();
				_poisson_RT->unbind();
				_interpShader.end();
			} else {
				std::swap(_poisson_tmp_RT, _poisson_RT);
				_interpShader.begin();
				_poisson_RT->clear();
				_poisson_RT->bind();
				glViewport(0,0, _poisson_RT->w(), _poisson_RT->h());
				glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _poisson_tmp_RT->texture());
				glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texture);
				_interp_scale.set(1.0f);
				RenderUtility::renderScreenQuad();
				_poisson_RT->unbind();
				_interpShader.end();
			}
		}
		glPopDebugGroup();
		return _poisson_RT->texture();
	}

	void	PoissonRenderer::process( const RenderTargetRGBA::Ptr& src, RenderTargetRGBA::Ptr& dst )
	{
		SIBR_ASSERT(src != nullptr);
		/// \todo TODO SR: support IRenderTarget instead of just RGBA
		render(src->texture());
		std::swap(dst, _poisson_RT);
	}

	void	PoissonRenderer::process( uint texID, RenderTargetRGBA::Ptr& dst )
	{
		render(texID);
		std::swap(dst, _poisson_RT);
	}

} /*namespace sibr*/ 
