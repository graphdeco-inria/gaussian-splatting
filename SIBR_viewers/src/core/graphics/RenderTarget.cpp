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



#include "core/graphics/RenderTarget.hpp"
//#define HEADLESS

namespace sibr
{
	void			blit(const IRenderTarget& src, const IRenderTarget& dst, GLbitfield mask, GLenum filter)
	{
#ifdef HEADLESS
		SIBR_ERR << "No named blit frame buffer in headless " << std::endl;
#else
		glBlitNamedFramebuffer(
			src.fbo(), dst.fbo(),
			0, 0, src.w(), src.h(),
			0, 0, dst.w(), dst.h(),
			mask, filter);
#endif
	}

	void			blit_and_flip(const IRenderTarget& src, const IRenderTarget& dst, GLbitfield mask, GLenum filter)
	{
#ifdef HEADLESS
		SIBR_ERR << "No named blit frame buffer in headless " << std::endl;
#else
		glBlitNamedFramebuffer(
			src.fbo(), dst.fbo(),
			0, 0, src.w(), src.h(),
			0, dst.h(), dst.w(), 0,
			mask, filter);
#endif
	}

	
} // namespace sibr
