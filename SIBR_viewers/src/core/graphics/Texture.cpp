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



#include "core/graphics/Texture.hpp"
//#define HEADLESS

namespace sibr
{
	void			blit(const ITexture2D& src, const IRenderTarget& dst, GLbitfield mask, GLenum filter, bool flip)
	{
		GLuint sourceFrameBuffer = 0;
		glGenFramebuffers(1, &sourceFrameBuffer);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, sourceFrameBuffer);
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src.handle(), 0);

		SIBR_ASSERT(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

#ifdef HEADLESS

		SIBR_ERR << "No named frame buffers in headless " << std::endl;
#else
		glBlitNamedFramebuffer(
			sourceFrameBuffer, dst.fbo(),
			0, 0, src.w(), src.h(),
			0, (flip ? dst.h() : 0), dst.w(), (flip ? 0 : dst.h()),
			mask, filter);

		glDeleteFramebuffers(1, &sourceFrameBuffer);
#endif
	}

	void			blit_and_flip(const ITexture2D& src, const IRenderTarget& dst, GLbitfield mask, GLenum filter)
	{
		blit(src, dst, mask, filter, true);
	}

	void			blitToColorAttachment(const ITexture2D& src, IRenderTarget& dst, int location, GLenum filter, bool flip)
	{
		// To blit only to a specific color attachment, it should be the only draw buffer registered.
		// So we override the drawbuffer from dst temporarily.
		glBindFramebuffer(GL_FRAMEBUFFER, dst.fbo());
		glDrawBuffer(GL_COLOR_ATTACHMENT0 + location);
		
		GLuint sourceFrameBuffer = 0;
		glGenFramebuffers(1, &sourceFrameBuffer);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, sourceFrameBuffer);
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src.handle(), 0);

		SIBR_ASSERT(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

#ifdef HEADLESS
		SIBR_ERR << "No named frame buffers in headless " << std::endl;
#else
		glBlitNamedFramebuffer(
			sourceFrameBuffer, dst.fbo(),
			0, 0, src.w(), src.h(),
			0, (flip ? dst.h() : 0), dst.w(), (flip ? 0 : dst.h()),
			GL_COLOR_BUFFER_BIT, filter);

		glDeleteFramebuffers(1, &sourceFrameBuffer);
#endif

		// Restore the drawbuffers.
		// We use bind() as it guarantees that all color buffers will be bound.
		dst.bind();
		dst.unbind();
	}

	void			blit(const IRenderTarget& src, const ITexture2D& dst, GLbitfield mask, GLenum filter)
	{
		GLuint dstFrameBuffer = 0;
		glGenFramebuffers(1, &dstFrameBuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dstFrameBuffer);
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst.handle(), 0);

		SIBR_ASSERT(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

#ifdef HEADLESS
		SIBR_ERR << "No named frame buffers in headless " << std::endl;
#else
		glBlitNamedFramebuffer(
			src.fbo(), dstFrameBuffer,
			0, 0, src.w(), src.h(),
			0, 0, dst.w(), dst.h(),
			mask, filter);
		glDeleteFramebuffers(1, &dstFrameBuffer);
#endif
	}

	void			blit(const ITexture2D& src, const ITexture2D& dst, GLbitfield mask, GLenum filter)
	{
		GLuint fbo[2];
		glGenFramebuffers(2, fbo);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo[0]);
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src.handle(), 0);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo[0]);
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst.handle(), 0);

		SIBR_ASSERT(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
		SIBR_ASSERT(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

#ifdef HEADLESS
		SIBR_ERR << "No named frame buffers in headless " << std::endl;
#else
		glBlitNamedFramebuffer(
			fbo[0], fbo[1],
			0, 0, src.w(), src.h(),
			0, 0, dst.w(), dst.h(),
			mask, filter);
		glDeleteFramebuffers(2, fbo);
#endif
	}

} // namespace sibr
