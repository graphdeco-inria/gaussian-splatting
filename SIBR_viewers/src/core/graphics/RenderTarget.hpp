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


#pragma once

# include "core/graphics/Config.hpp"
# include "core/system/Vector.hpp"
# include "core/graphics/Image.hpp"
# include "core/graphics/Types.hpp"
# include "core/system/Vector.hpp"
# include "core/graphics/RenderUtility.hpp"


# define SIBR_MAX_SHADER_ATTACHMENTS (1<<3)

namespace sibr
{



	/** Rendertarget interface. A render target wraps an OpenGL framebuffer, 
	* that can have one depth buffer, one stencil buffer, and one or more color attachments.
	* This generic interface is typeless, \sa RenderTarget.
	* \ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT IRenderTarget
	{
	public:
		typedef std::shared_ptr<IRenderTarget>	Ptr;
		typedef std::unique_ptr<IRenderTarget>	UPtr;
	public:
		/// Destructor.
		virtual ~IRenderTarget(void) { }

		/** Get the texture handle of the t-th color attachment. 
		\param t the color attachment slot
		\return the texture handle
		\deprecated Use handle instead.
		*/
		virtual GLuint texture(uint t = 0) const = 0;

		/** Get the texture handle of the t-th color attachment.
		\param t the color attachment slot
		\return the texture handle
		*/
		virtual GLuint handle(uint t = 0) const = 0;

		/** Bind the rendertarget for drawing. All color buffers are bound, along 
			with the depth and optional stencil buffers.*/
		virtual void bind(void) = 0;

		/** Unbind the rendertarget.
		\note This will bind the window rendertarget. */
		virtual void unbind(void) = 0;

		/** Clear the content of the rendertarget. */
		virtual void clear(void) = 0;

		/** \return the rendertarget width. */
		virtual uint   w(void) const = 0;

		/** \return the rendertarget height. */
		virtual uint   h(void) const = 0;

		/** \return the framebuffer handle. */
		virtual GLuint fbo(void) const = 0;
	};

	/**
	* A render target wraps an OpenGL framebuffer, that can have one depth buffer, 
	* one stencil buffer, and one or more color attachments.
	* \sa IRenderTarget.
	* \ingroup sibr_graphics
	*/
	template<typename T_Type, unsigned int T_NumComp>
	class RenderTarget : public IRenderTarget {
		SIBR_DISALLOW_COPY(RenderTarget);
	public:
		typedef		Image<T_Type, T_NumComp>		PixelImage;
		typedef		typename PixelImage::Pixel		PixelFormat;
		typedef		std::shared_ptr<RenderTarget<T_Type, T_NumComp>>	Ptr;
		typedef		std::unique_ptr<RenderTarget<T_Type, T_NumComp>>	UPtr;

	private:

		GLuint m_fbo = 0; ///< Framebuffer handle.
		GLuint m_depth_rb = 0; ///< Depth renderbuffer handle.
		GLuint m_stencil_rb = 0; ///< Stencil renderbuffer handle.
		GLuint m_textures[SIBR_MAX_SHADER_ATTACHMENTS]; ///< Color texture handles.
		uint   m_numtargets = 0; ///< Number of active color attachments.
		bool   m_autoMIPMAP = false; ///< Generate mipmaps on the fly.
		bool   m_msaa = false; ///< Use multisampled targets.
		bool   m_stencil = false; ///< Has a stencil buffer.
		uint   m_W = 0; ///< Width.
		uint   m_H = 0; ///< Height.

	public:

		/// Constructor.
		RenderTarget(void);
		
		/** Constructor and allocation.
		\param w the target width
		\param h the target height
		\param flags options
		\param num the number of color attachments.
		*/
		RenderTarget(uint w, uint h, uint flags = 0, uint num = 1);

		/// Destructor.
		~RenderTarget(void);

		/** Get the texture handle of the t-th color attachment.
		\param t the color attachment slot
		\return the texture handle
		\deprecated Use handle instead.
		*/
		GLuint texture(uint t = 0) const;
		
		/** Get the texture handle of the t-th color attachment. 
		\param t the color attachment slot
		\return the texture handle
		*/
		GLuint handle(uint t = 0) const;

		/** \return the depth buffer handle. */
		GLuint depthRB() const;

		/** Bind the rendertarget for drawing. All color buffers are bound, along
			with the depth and optional stencil buffers.*/
		void bind(void);

		/** Unbind the rendertarget.
		\note This will bind the window rendertarget. */
		void unbind(void);

		/** Clear the rendertarget buffers with default values.
		 * \warning This function will unbind the render target after clearing.
		 */
		void clear(void);

		/** Clear the rendertarget buffers, using a custom clear color.
		 * \param v the clear color
		 * \warning This function will unbind the render target after clearing.
		 * \bug This function does not rescale values for uchar (so background is either 0 or 1)
		 */
		void clear(const typename RenderTarget<T_Type, T_NumComp>::PixelFormat& v);

		/** Clear the stencil buffer only. */
		void clearStencil(void);

		/** Clear the depth buffer only. */
		void clearDepth(void);

		/** Readback the content of a color attachment into an sibr::Image on the CPU.
		\param image will contain the texture content
		\param target the color attachment index to read
		\warning Might cause a GPU flush/sync.
		*/
		template <typename TType, uint NNumComp>
		void readBack(sibr::Image<TType, NNumComp>& image, uint target = 0) const;

		/** Readback the content of a color attachment into a cv::Mat on the CPU.
		\param image will contain the texture content
		\param target the color attachment index to read
		\warning Might cause a GPU flush/sync.
		*/
		template <typename TType, uint NNumComp>
		void readBackToCVmat(cv::Mat& image, uint target = 0) const;

		/** Readback the content of the depth attachment into an sibr::Image on the CPU.
		\param image will contain the depth content
		\warning Might cause a GPU flush/sync.
		\warning Image orientation might be inconsistent with readBack (flip around horizontal axis).
		*/
		template <typename TType, uint NNumComp>
		void readBackDepth(sibr::Image<TType, NNumComp>& image) const;

		/** \return the number of active color targets. */
		uint   numTargets(void)  const;

		/** \return the target width. */
		uint   w(void)  const;

		/** \return the target height. */
		uint   h(void)  const;

		/** \return the framebuffer handle. */
		GLuint fbo(void)  const;
	};

	/**
	Copy the content of a render target to another render target, resizing if needed.
	\param src source rendertarget
	\param dst destination rendertarget
	\param mask which part of the buffer to copy (color, depth, stencil).
	\param filter filtering mode if the two rendertargets have different dimensions (linear or nearest)
	\note The blit can only happen for color attachment 0 in both src and dst.
	\warning If the mask contains the depth or stencil, filter must be GL_NEAREST
	 \ingroup sibr_graphics
	*/
	SIBR_GRAPHICS_EXPORT void			blit(const IRenderTarget& src, const IRenderTarget& dst, GLbitfield mask = GL_COLOR_BUFFER_BIT, GLenum filter = GL_LINEAR);

	/**
	Copy the content of a render target to another render target, resizing if needed and flipping the result.
	\param src source rendertarget
	\param dst destination rendertarget
	\param mask which part of the buffer to copy (color, depth, stencil).
	\param filter filtering mode if the two rendertargets have different dimensions (linear or nearest)
	\note The blit can only happen for color attachment 0 in both src and dst.
	\warning If the mask contains the depth or stencil, filter must be GL_NEAREST
	 \ingroup sibr_graphics
	*/
	SIBR_GRAPHICS_EXPORT void			blit_and_flip(const IRenderTarget& src, const IRenderTarget& dst, GLbitfield mask = GL_COLOR_BUFFER_BIT, GLenum filter = GL_LINEAR);

	/** Display a rendertarget color content in a popup window (backed by OpenCV).
	\param rt the rendertarget to display
	\param layer the color attachment to display
	\param windowTitle name of the window
	\param closeWindow should the window be closed when pressing a key
	\ingroup sibr_graphics
	*/
	template <typename T_Type, unsigned T_NumComp>
	static void		show( const RenderTarget<T_Type, T_NumComp> & rt, uint layer=0, const std::string& windowTitle="sibr::show()" , bool closeWindow = true ) {
		sibr::Image<T_Type, T_NumComp>	img(rt.w(), rt.h());
		rt.readBack(img, layer);
		show(img, windowTitle, closeWindow);
	}
	
	/** Display a rendertarget depth content in a popup window (backed by OpenCV).
	\param rt the rendertarget to display
	\param windowTitle name of the window
	\param closeWindow should the window be closed when pressing a key
	\ingroup sibr_graphics
	*/
	template <typename T_Type, unsigned T_NumComp>
	static void		showDepth( const RenderTarget<T_Type, T_NumComp> & rt, const std::string& windowTitle="sibr::show()" , bool closeWindow = true ) {
		sibr::Image<float, 3>	img(rt.w(), rt.h());
		rt.readBackDepth(img);
		show(img, windowTitle, closeWindow);
	}
	
	/** Display a rendertarget alpha content as a grey map in a popup window (backed by OpenCV).
	\param rt the rendertarget to display
	\param windowTitle name of the window
	\param closeWindow should the window be closed when pressing a key
	\ingroup sibr_graphics
	*/
	template <typename T_Type, unsigned T_NumComp>
	static void		showDepthFromAlpha( const RenderTarget<T_Type, T_NumComp> & rt, const std::string& windowTitle="sibr::show()" , bool closeWindow = true ) {
		sibr::Image<float, 4>	img(rt.w(), rt.h());
		rt.readBack(img);

		for (uint y = 0; y < img.h(); ++y)
		{
			for (uint x = 0; x < img.w(); ++x)
			{
				sibr::ColorRGBA c = img.color(x, y);
				c = sibr::ColorRGBA(1.f, 1.f, 1.f, 0.f) * c[3];
				c[3] = 1.f;
				img.color(x, y, c);
			}
		}

		show(img, windowTitle, closeWindow);
	}


	// --- Typedef RenderTarget --------------------------------------------------

	typedef RenderTarget<unsigned char, 3>  RenderTargetRGB;
	typedef RenderTarget<unsigned char, 4>  RenderTargetRGBA;
	typedef RenderTarget<unsigned char, 1>  RenderTargetLum;

	typedef RenderTarget<unsigned short, 1>    RenderTargetLum16;
	typedef RenderTarget<unsigned short, 2>    RenderTargetUV16;
	typedef RenderTarget<unsigned short, 3>    RenderTargetRGB16;
	typedef RenderTarget<unsigned short, 4>    RenderTargetRGBA16;

	typedef RenderTarget<int, 1>			   RenderTargetInt1;

	typedef RenderTarget<float, 3>          RenderTargetRGB32F;
	typedef RenderTarget<float, 4>          RenderTargetRGBA32F;
	typedef RenderTarget<float, 1>          RenderTargetLum32F;
	typedef RenderTarget<float, 2>          RenderTargetUV32F;


	// --- DEFINITIONS RenderTarget --------------------------------------------------

	template<typename T_Type, unsigned int T_NumComp>
	RenderTarget<T_Type, T_NumComp>::RenderTarget(void) {
		m_fbo = 0;
		m_depth_rb = 0;
		m_numtargets = 0;
		m_W = 0;
		m_H = 0;
	}

	template<typename T_Type, unsigned int T_NumComp>
	RenderTarget<T_Type, T_NumComp>::RenderTarget(uint w, uint h, uint flags, uint num) {
		RenderUtility::useDefaultVAO();

		m_W = w;
		m_H = h;

		bool is_depth = (GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::isdepth != 0);

		int maxRenterTargets = 0;
		glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxRenterTargets);

		SIBR_ASSERT(num <= uint(maxRenterTargets) && num > 0);
		SIBR_ASSERT(!is_depth || num == 1);

		if (flags & SIBR_GPU_INTEGER) {
			if (GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_internal_format < 0) {
				throw std::runtime_error("Integer render  - format does not support integer mapping");
			}
		}

		glGenFramebuffers(1, &m_fbo);

		if (!is_depth) {
			glGenRenderbuffers(1, &m_depth_rb); // depth buffer for color rt
			//glGenRenderbuffers(1, &m_stencil_rb); // stencil buffer for color rt
		} else
			m_depth_rb = 0;

		m_numtargets = num;
		m_autoMIPMAP = ((flags & SIBR_GPU_AUTOGEN_MIPMAP) != 0);

		m_msaa = ((flags & SIBR_GPU_MULSTISAMPLE) != 0);
		m_stencil = ((flags & SIBR_STENCIL_BUFFER) != 0);

		if (m_msaa && (m_numtargets != 1))
			throw std::runtime_error("Only one MSAA render target can be attached.");
		for (uint n = 0; n < m_numtargets; n++) {
			if (m_msaa)
				break;

			glGenTextures(1, &m_textures[n]);


			glBindTexture(GL_TEXTURE_2D, m_textures[n]);

			if (flags & SIBR_CLAMP_UVS) {
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}

			/// \todo: following causes enum compare warning -Wenum-compare
			glTexImage2D(GL_TEXTURE_2D,
				0,
				(flags & SIBR_GPU_INTEGER)
				? GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_internal_format
				: GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::internal_format,
				w, h,
				0,
				(flags & SIBR_GPU_INTEGER)
				? GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_format
				: GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::format,
				GLType<typename PixelFormat::Type>::type,
				NULL);


			if (!m_autoMIPMAP) {
#if SIBR_COMPILE_FORCE_SAMPLING_LINEAR
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#else
				if (flags & SIBR_GPU_LINEAR_SAMPLING) {
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				} else {
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				}
#endif
			} else { /// \todo TODO: this crashes with 16F RT
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			}
		}


		if (!m_msaa) {
			if (!is_depth) {
				glBindRenderbuffer(GL_RENDERBUFFER, m_depth_rb);
				if (!m_stencil)
					glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, w, h);
				else
					glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);

				//CHECK_GL_ERROR;
				//glBindRenderbuffer(GL_RENDERBUFFER, m_stencil_rb);
				//glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8, w, h);
				CHECK_GL_ERROR;
				glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
				for (uint n = 0; n < m_numtargets; n++) {
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + n, GL_TEXTURE_2D, m_textures[n], 0);
				}
				CHECK_GL_ERROR;
				if (!m_stencil)
					glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_rb);
				else
					glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_depth_rb);
				//CHECK_GL_ERROR;
				//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_stencil_rb);
			} else {
				glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_textures[0], 0);
				glDrawBuffer(GL_NONE);
				glReadBuffer(GL_NONE);
			}
		}

		if (m_msaa) {
			uint msaa_samples = ((flags >> 7) & 0xF) << 2;

			if (msaa_samples == 0)
				throw std::runtime_error("Number of MSAA Samples not set. Please use SIBR_MSAA4X, SIBR_MSAA8X, SIBR_MSAA16X or SIBR_MSAA32X as an additional flag.");

			glGenTextures(1, &m_textures[0]);
			glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m_textures[0]);
			CHECK_GL_ERROR;
			/// TODO: following causes enum compare warning -Wenum-compare
			glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE,
				msaa_samples,
				(flags & SIBR_GPU_INTEGER)
				? GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_internal_format
				: GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::internal_format,
				w, h,
				GL_TRUE
			);
			glBindRenderbuffer(GL_RENDERBUFFER, m_depth_rb);
			glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa_samples, GL_DEPTH_COMPONENT32, w, h);
			glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
			glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_textures[0], 0);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_rb);
		}

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			switch (status) {
			case GL_FRAMEBUFFER_UNSUPPORTED:
				throw std::runtime_error("Cannot create FBO - GL_FRAMEBUFFER_UNSUPPORTED error");
				break;
			default:
				SIBR_DEBUG(status);
				throw std::runtime_error("Cannot create FBO (unknow reason)");
				break;
			}
		}

		if (m_autoMIPMAP) {
			for (uint i = 0; i < m_numtargets; i++) {
				glBindTexture(GL_TEXTURE_2D, m_textures[i]);
				glGenerateMipmap(GL_TEXTURE_2D);
			}
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp>
	RenderTarget<T_Type, T_NumComp>::~RenderTarget(void) {
		for (uint i = 0; i < m_numtargets; i++)
			glDeleteTextures(1, &m_textures[i]);
		glDeleteFramebuffers(1, &m_fbo);
		glDeleteRenderbuffers(1, &m_depth_rb);
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp>
	GLuint RenderTarget<T_Type, T_NumComp>::depthRB() const {
		return m_depth_rb;
	}

	template<typename T_Type, unsigned int T_NumComp>
	GLuint RenderTarget<T_Type, T_NumComp>::texture(uint t) const {
		SIBR_ASSERT(t < m_numtargets);
		return m_textures[t];
	}
	template<typename T_Type, unsigned int T_NumComp>
	GLuint RenderTarget<T_Type, T_NumComp>::handle(uint t) const {
		SIBR_ASSERT(t < m_numtargets);
		return m_textures[t];
	}

	template<typename T_Type, unsigned int T_NumComp>
	void RenderTarget<T_Type, T_NumComp>::bind(void) {
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
		bool is_depth = (GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::isdepth != 0);
		if (!is_depth) {
			if (m_numtargets > 0) {
				GLenum drawbuffers[SIBR_MAX_SHADER_ATTACHMENTS];
				for (uint i = 0; i < SIBR_MAX_SHADER_ATTACHMENTS; i++)
					drawbuffers[i] = GL_COLOR_ATTACHMENT0 + i;
				glDrawBuffers(m_numtargets, drawbuffers);
			}
		} else {
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
		}
	}

	template<typename T_Type, unsigned int T_NumComp>
	void RenderTarget<T_Type, T_NumComp>::unbind(void) {
		if (m_autoMIPMAP) {
			for (uint i = 0; i < m_numtargets; i++) {
				glBindTexture(GL_TEXTURE_2D, m_textures[i]);
				glGenerateMipmap(GL_TEXTURE_2D);
			}
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	template<typename T_Type, unsigned int T_NumComp>
	void RenderTarget<T_Type, T_NumComp>::clear(void) {
		clear(PixelFormat());
	}

	template<typename T_Type, unsigned int T_NumComp>
	void RenderTarget<T_Type, T_NumComp>::clear(const typename RenderTarget<T_Type, T_NumComp>::PixelFormat& v) {
		bind();
		if (PixelFormat::NumComp == 1) {
			glClearColor(GLclampf(v[0]), 0, 0, 0);
		} else if (PixelFormat::NumComp == 2) {
			glClearColor(GLclampf(v[0]), GLclampf(v[1]), 0, 0);
		} else if (PixelFormat::NumComp == 3) {
			glClearColor(GLclampf(v[0]), GLclampf(v[1]), GLclampf(v[2]), 0);
		} else if (PixelFormat::NumComp == 4) {
			glClearColor(GLclampf(v[0]), GLclampf(v[1]), GLclampf(v[2]), GLclampf(v[3]));
		}
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		unbind();
	}

	template<typename T_Type, unsigned int T_NumComp>
	void RenderTarget<T_Type, T_NumComp>::clearStencil() {
		bind();
		glClearStencil(0);
		glClear(GL_STENCIL_BUFFER_BIT);
		unbind();
	}

	template<typename T_Type, unsigned int T_NumComp>
	void RenderTarget<T_Type, T_NumComp>::clearDepth() {
		bind();
		glClear(GL_DEPTH_BUFFER_BIT);
		unbind();
	}

	template<typename T_Type, unsigned int T_NumComp>
	template <typename T_IType, uint N_INumComp>
	void RenderTarget<T_Type, T_NumComp>::readBack(sibr::Image<T_IType, N_INumComp>& img, uint target) const {
		//void RenderTarget<T_Type, T_NumComp>::readBack(PixelImage& img, uint target) const {
		glFinish();
		if (target >= m_numtargets)
			SIBR_ERR << "Reading back texture out of bounds" << std::endl;

		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
		bool is_depth = (GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::isdepth != 0);
		if (!is_depth) {
			if (m_numtargets > 0) {
				sibr::Image<T_Type, T_NumComp> buffer(m_W, m_H);

				GLenum drawbuffers = GL_COLOR_ATTACHMENT0 + target;
				glDrawBuffers(1, &drawbuffers);
				glReadBuffer(drawbuffers);

				glReadPixels(0, 0, m_W, m_H,
					GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::format,
					GLType<typename PixelFormat::Type>::type,
					buffer.data()
				);

				sibr::Image<T_IType, N_INumComp>	out;
				img.fromOpenCV(buffer.toOpenCV());
			}
		} else
			SIBR_ERR << "RenderTarget::readBack: This function should be specialized "
			"for handling depth buffer." << std::endl;
		img.flipH();
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

	}


	template<typename T_Type, unsigned int T_NumComp>
	template <typename T_IType, uint N_INumComp>
	void RenderTarget<T_Type, T_NumComp>::readBackToCVmat(cv::Mat& img, uint target) const {

		using Infos = GLTexFormat<cv::Mat, T_IType, N_INumComp>;

		if (target >= m_numtargets)
			SIBR_ERR << "Reading back texture out of bounds" << std::endl;

		cv::Mat tmp(m_H, m_W, Infos::cv_type());

		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
		bool is_depth = (Infos::isdepth != 0);
		if (!is_depth) {
			if (m_numtargets > 0) {
				GLenum drawbuffers = GL_COLOR_ATTACHMENT0 + target;
				glDrawBuffers(1, &drawbuffers);
				glReadBuffer(drawbuffers);

				glReadPixels(0, 0, m_W, m_H,
					Infos::format,
					Infos::type,
					Infos::data(tmp)
				);
			}
		} else {
			SIBR_ERR << "RenderTarget::readBack: This function should be specialized "
				"for handling depth buffer." << std::endl; \
		}
		img = Infos::flip(tmp);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	template <typename TType, uint NNumComp>
	template <typename T_IType, uint N_INumComp>
	void RenderTarget<TType, NNumComp>::readBackDepth(sibr::Image<T_IType, N_INumComp>& image) const {
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

		glReadBuffer(GL_COLOR_ATTACHMENT0);

		sibr::Image<float, 1> buffer(m_W, m_H);
		glReadPixels(0, 0, m_W, m_H,
			GL_DEPTH_COMPONENT,
			GL_FLOAT,
			buffer.data()
		);

		sibr::Image<T_IType, N_INumComp>	out(buffer.w(), buffer.h());
		for (uint y = 0; y < buffer.h(); ++y)
			for (uint x = 0; x < buffer.w(); ++x)
				out.color(x, y, sibr::ColorRGBA(1, 1, 1, 1.f) * buffer(x, y)[0]);
		image = std::move(out);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	template<typename T_Type, unsigned int T_NumComp>
	uint   RenderTarget<T_Type, T_NumComp>::numTargets(void)  const { return m_numtargets; }
	template<typename T_Type, unsigned int T_NumComp>
	uint   RenderTarget<T_Type, T_NumComp>::w(void)  const { return m_W; }
	template<typename T_Type, unsigned int T_NumComp>
	uint   RenderTarget<T_Type, T_NumComp>::h(void)  const { return m_H; }
	template<typename T_Type, unsigned int T_NumComp>
	uint   RenderTarget<T_Type, T_NumComp>::fbo(void)  const { return m_fbo; }


} // namespace sibr
