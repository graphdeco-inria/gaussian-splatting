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

#include <type_traits>

# include "core/graphics/Config.hpp"
# include "core/system/Vector.hpp"
# include "core/graphics/Image.hpp"
# include "core/graphics/Types.hpp"
# include "core/graphics/RenderTarget.hpp"

namespace sibr
{

	/** Interface for a generic GPU 2D texture.
	* \sa Texture2D
	* \ingroup sibr_graphics
	*/
	class ITexture2D
	{
	public:
		typedef std::shared_ptr<ITexture2D>	Ptr;
		typedef std::unique_ptr<ITexture2D>	UPtr;
	public:

		/// Destructor.
		virtual ~ITexture2D(void) { }

		/** \return the texture handle. */
		virtual GLuint handle(void) const = 0;

		/** \return the texture width. */
		virtual uint   w(void) const = 0;

		/** \return the texture height. */
		virtual uint   h(void) const = 0;
	};

	/** Represent a 2D texture on the GPU, with custom format and type.
	* \sa ITexture2D
	* \ingroup sibr_graphics
	*/
	template<typename T_Type, unsigned int T_NumComp>
	class Texture2D : public ITexture2D {
		SIBR_DISALLOW_COPY(Texture2D);
	public:
		typedef		Image<T_Type, T_NumComp>			PixelImage;
		typedef		typename PixelImage::Pixel			PixelFormat;
		typedef		std::shared_ptr<Texture2D<T_Type, T_NumComp>>	Ptr;
		typedef		std::unique_ptr<Texture2D<T_Type, T_NumComp>>	UPtr;

	public:

		/// Constructor.
		Texture2D(void);

		/** Constructor from an image.
		\param img the image to upload to the GPU
		\param flags options
		*/
		template<typename ImageType> Texture2D(const ImageType& img, uint flags = 0);

		/** Constructor from a list of images, one for each mip level.
		\param miparray the images to upload to the GPU
		\param flags options
		*/
		Texture2D(const std::vector<PixelImage>& miparray, uint flags = 0);

		/// Destructor.
		~Texture2D(void);

		/** \return the texture handle. */
		GLuint handle(void) const;

		/** \return the texture width. */
		uint   w(void) const;

		/** \return the texture height. */
		uint   h(void) const;

		/** \return a CPU image containing the texture content.
			\warning Can cause a GPU flush/sync.
		*/
		sibr::Image<T_Type, T_NumComp>		readBack(void) const;

		/** Update the content of the txeture with a new image.
		\param img the new content.
		*/
		template<typename ImageType> void update(const ImageType& img);

		/** Trigger an update of the mipmaps for level 0 to maxLOD.
		\param maxLOD the maximum level of mipmap to generate. If -1, as many as possible based on the texture size.
		*/
		void mipmap(int maxLOD = -1);

	private:
		GLuint  m_Handle = 0; ///< Texture handle.
		uint    m_W = 0; ///< Texture width.
		uint    m_H = 0; ///< Texture height.
		uint    m_Flags = 0; ///< Options.
		bool	m_autoMIPMAP = false; ///< Should the mipmaps be generated automatically.

		/** Create 2D texture from a generic image (sibr::image or cv::Mat).
		\param array the image
		\param flags options
		\return the handle of the texture
		*/
		template<typename ImageType> static GLuint create2D(const ImageType& array, uint flags);

		/** Create 2D texture with custom mipmaps from a list of generic images (sibr::image or cv::Mat).
		\param miparray the images
		\param flags options
		\return the handle of the texture
		*/
		static GLuint create2D(const std::vector<PixelImage>& miparray, uint flags);

		/** Send the CPU image data to the GPU.
		\param id the created texture
		\param array the image data
		\param flags options
		*/
		template<typename ImageType> static void send2D(GLuint id, const ImageType& array, uint flags);

		/** Send the CPU images data for each mipmap to the GPU.
		\param id the created texture
		\param miparray the image data
		\param flags options
		*/
		static void send2Dmipmap(GLuint id, const std::vector<PixelImage>& miparray, uint flags);

	};


	/** Interface for a generic GPU 2D array texture.
	* \sa Texture2DArray
	* \ingroup sibr_graphics
	*/
	class ITexture2DArray
	{
	public:
		typedef std::shared_ptr<ITexture2DArray>	Ptr;
		typedef std::unique_ptr<ITexture2DArray>	UPtr;
	public:
		/// Destructor.
		virtual ~ITexture2DArray(void) { }

		/** \return the texture handle. */
		virtual GLuint	handle(void) const = 0;

		/** \return the texture width. */
		virtual uint	w(void) const = 0;

		/** \return the texture height. */
		virtual uint	h(void) const = 0;

		/** \return the texture layer count. */
		virtual uint	depth(void) const = 0;

		/** \return the number of mipmap levels. */
		virtual uint	numLODs(void) const = 0;

		/** Read back the value of a given pixel to the CPU.
		\param i layer
		\param x x coordinate in [0,w-1]
		\param y y coordinate in [0,h-1]
		\param lod the mip level
		\return a converted RGBA float color
		\warning Use only for debugging, can cause a GPU flush/sync.
		*/
		virtual Vector4f	readBackPixel(int i, int x, int y, uint lod = 0) const = 0;
	};

	/**
	* Represent an array of 2D textures on the GPU, with custom format, type and slice count.
	* \sa ITexture2DArray
	* \ingroup sibr_graphics
	*/
	template<typename T_Type, unsigned int T_NumComp>
	class Texture2DArray : public ITexture2DArray {
		SIBR_DISALLOW_COPY(Texture2DArray);
	public:
		typedef		Image<T_Type, T_NumComp>			PixelImage;
		typedef		typename PixelImage::Pixel			PixelFormat;
		typedef		RenderTarget<T_Type, T_NumComp>			PixelRT;
		typedef		std::shared_ptr<Texture2DArray<T_Type, T_NumComp>>	Ptr;
		typedef		std::unique_ptr<Texture2DArray<T_Type, T_NumComp>>	UPtr;

	public:

		/** Constructor.
		\param d number of layers
		\param flags options
		*/
		Texture2DArray(const uint d = 0, uint flags = 0);

		/** Constructor.
		\param w width
		\param h height
		\param d number of layers
		\param flags options
		*/
		Texture2DArray(const uint w, const uint h, const uint d, uint flags = 0);

		/** Constructor from a set of rendertargets.
		\param images list of rendertargets, one for each layer
		\param flags options
		\warning RTs should be of the same size.
		*/
		Texture2DArray(const std::vector<typename PixelRT::Ptr>& images, uint flags = 0);

		/** Constructor from a set of CPU images.
		\param images list of images, one for each layer
		\param flags options
		\note All images will be resized to the dimensions of the largest one.
		*/
		template<typename ImageType>
		Texture2DArray(const std::vector<ImageType>& images, uint flags = 0);

		/** Constructor from a set of CPU images that will be resized to a fix size.
		\param images list of images, one for each layer
		\param w the target width
		\param h the target height
		\param flags options
		*/
		template<typename ImageType>
		Texture2DArray(const std::vector<ImageType>& images, uint w, uint h, uint flags = 0);

		/** Constructor from a set of CPU images, with custom mipmaps.
		\param images list of lists of images, one for each mip level, each containing an image for each layer
		\param flags options
		\note All images will be resized to the dimensions of the largest one.
		*/
		template<typename ImageType>
		Texture2DArray(const std::vector<std::vector<ImageType>>& images, uint flags = 0);

		/** Constructor from a set of CPU images, with custom mipmaps.
		\param images list of lists of images, one for each mip level, each containing an image for each layer
		\param w the target width
		\param h the target height
		\param flags options
		*/
		template<typename ImageType>
		Texture2DArray(const std::vector<std::vector<ImageType>>& images, uint w, uint h, uint flags = 0);

		/** Create the texture from a set of images and send it to GPU.
		\param images list of images, one for each layer
		\param flags options
		\note All images will be resized to the dimensions of the largest one.
		*/
		template<typename ImageType>
		void createFromImages(const std::vector<ImageType>& images, uint flags = 0);

		/** Create the texture from a set of images and send it to GPU. images will be resized to the target size.
		\param images list of images, one for each layer
		\param w the target width
		\param h the target height
		\param flags options
		*/
		template<typename ImageType>
		void createFromImages(const std::vector<ImageType>& images, uint w, uint h, uint flags = 0);

		/** Create the texture from a set of images and send it to GPU while compressing them.
		\param images list of images, one for each layer
		\param compression the GL_COMPRESSED format. It must be choosen accordingly to the texture internal format.
		\param flags options
		\note All images will be resized to the dimensions of the largest one.
		*/
		template<typename ImageType>
		void createCompressedFromImages(const std::vector<ImageType>& images, uint compression, uint flags = 0);

		/** Create the texture from a set of images and send it to GPU while compressing them. images will be resized to the target size.
		\param images list of images, one for each layer
		\param w the target width
		\param h the target height
		\param compression the GL_COMPRESSED format. It must be choosen accordingly to the texture internal format.
		\param flags options
		*/
		template<typename ImageType>
		void createCompressedFromImages(const std::vector<ImageType>& images, uint w, uint h, uint compression, uint flags = 0);

		/** Create the texture from a set of images with custom mipmaps and send it to GPU.
		\param images list of lists of images, one for each mip level, each containing an image for each layer
		\param flags options
		\note All images will be resized to the dimensions of the largest one.
		*/
		template<typename ImageType>
		void createFromImages(const std::vector<std::vector<ImageType>>& images, uint flags = 0);

		/** Create the texture from a set of images with custom mipmaps and send it to GPU.
		\param images list of lists of images, one for each mip level, each containing an image for each layer
		\param w the target width
		\param h the target height
		\param flags options
		*/
		template<typename ImageType>
		void createFromImages(const std::vector<std::vector<ImageType>>& images, uint w, uint h, uint flags = 0);

		/** Update the content of all layers of the texture.
		\param images the new content to use
		\note All images will be resized to the size of the largest one.
		*/
		template<typename ImageType>
		void updateFromImages(const std::vector<ImageType>& images);

		/** Create the texture from a set of rendertargets and send it to GPU.
		\param RTs list of rendertargets, one for each layer
		\param flags options
		\warning RTs should be of the same size.
		*/
		void createFromRTs(const std::vector<typename PixelRT::Ptr>& RTs, uint flags = 0);

		/** Update the content of specific layers of the texture.
		\param images the new content to use
		\param slices the indices of the slices to update
		\note All images will be resized to the size of the largest one.
		*/
		template<typename ImageType>
		void updateSlices(const std::vector<ImageType>& images, const std::vector<int>& slices);

		/// Destructor.
		~Texture2DArray(void);

		/** \return the texture handle. */
		GLuint	handle(void) const;

		/** \return the texture width. */
		uint	w(void) const;

		/** \return the texture height. */
		uint	h(void) const;

		/** \return the texture layer count. */
		uint	depth(void) const;

		/** \return the number of mipmap levels. */
		uint	numLODs(void) const;

		/** Read back the value of a given pixel to the CPU.
		\param i layer
		\param x x coordinate in [0,w-1]
		\param y y coordinate in [0,h-1]
		\param lod the mip level
		\return a converted RGBA float color
		\warning Use only for debugging, can cause a GPU flush/sync.
		*/
		Vector4f	readBackPixel(int i, int x, int y, uint lod = 0) const;

	private:

		/** Create the texture array. */
		void createArray(uint compression = 0);

		/** Upload the images data to the GPU.
		\param images the data to upload
		*/
		template<typename ImageType>
		void sendArray(const std::vector<ImageType>& images);

		/** Copy the rendertargets data to the texture.
		\param RTs the rendertargets to copy
		*/
		void sendRTarray(const std::vector<typename PixelRT::Ptr>& RTs);

		/** Upload the images data to the GPU.
		\param images the data to upload
		*/
		template<typename ImageType>
		void sendMipArray(const std::vector<std::vector<ImageType>>& images);

		/** Flip and rescale a subset of images from a list.
		\param images the images to resize
		\param tmp a temporary buffer
		\param tw the target width
		\param th the target height
		\param slices the indices of the images to process in the list
		\return a list of pointers to the transformed images
		*/
		template<typename ImageType>
		std::vector<const ImageType*> applyFlipAndResize(
			const std::vector<ImageType>& images,
			std::vector<ImageType>& tmp, uint tw, uint th,
			const std::vector<int>& slices
		);

		/** Flip and rescale a set of images.
		\param images the images to resize
		\param tmp a temporary buffer
		\param tw the target width
		\param th the target height
		\return a list of pointers to the transformed images
		*/
		template<typename ImageType>
		std::vector<const ImageType*> applyFlipAndResize(
			const std::vector<ImageType>& images,
			std::vector<ImageType>& tmp, uint tw, uint th
		);

		GLuint  m_Handle = 0; ///< Texture handle.
		uint    m_W = 0; ///< Texture width.
		uint    m_H = 0; ///< Texture height.
		uint    m_Flags = 0; ///< Options.
		uint	m_Depth = 0; ///< Layers count.
		uint	m_numLODs = 1; ///< Mipmap level count.
	};


	/** Interface for a generic GPU cubemap texture.
	* \sa TextureCubeMap
	* \ingroup sibr_graphics
	*/
	class ITextureCubeMap
	{
	public:
		typedef std::shared_ptr<ITextureCubeMap>	Ptr;
		typedef std::unique_ptr<ITextureCubeMap>	UPtr;
	public:
		/// Destructor.
		virtual ~ITextureCubeMap(void) { }

		/** \return the texture handle. */
		virtual GLuint	handle(void) const = 0;

		/** \return the texture width. */
		virtual uint	w(void) const = 0;

		/** \return the texture height. */
		virtual uint	h(void) const = 0;
	};

	/**
	* Represent a cubemap composed of 6 2D faces on the GPU, with custom format and type.
	* \sa ITextureCubeMap
	* \ingroup sibr_graphics
	*/
	template<typename T_Type, unsigned int T_NumComp>
	class TextureCubeMap : public ITextureCubeMap {
		SIBR_DISALLOW_COPY(TextureCubeMap);

	public:
		typedef		Image<T_Type, T_NumComp>			PixelImage;
		typedef		typename PixelImage::Pixel			PixelFormat;
		typedef		RenderTarget<T_Type, T_NumComp>			PixelRT;
		typedef		std::shared_ptr<TextureCubeMap<T_Type, T_NumComp>>	Ptr;
		typedef		std::unique_ptr<TextureCubeMap<T_Type, T_NumComp>>	UPtr;

	public:

		/// Constructor.
		TextureCubeMap(void);

		/** Constructor.
		\param w width
		\param h height
		\param flags options
		*/
		TextureCubeMap(const uint w, const uint h, uint flags = 0);

		/** Create a cubemap from 6 images.
		\param xpos positive X face
		\param xneg negative X face
		\param ypos positive Y face
		\param yneg negative Y face
		\param zpos positive Z face
		\param zneg negative Z face
		\param flags options
		*/
		TextureCubeMap(const PixelImage& xpos, const PixelImage& xneg,
			const PixelImage& ypos, const PixelImage& yneg,
			const PixelImage& zpos, const PixelImage& zneg, uint flags = 0);

		/** Create the texture from 6 images.
		\param xpos positive X face
		\param xneg negative X face
		\param ypos positive Y face
		\param yneg negative Y face
		\param zpos positive Z face
		\param zneg negative Z face
		\param flags options
		*/
		void createFromImages(const PixelImage& xpos, const PixelImage& xneg,
			const PixelImage& ypos, const PixelImage& yneg,
			const PixelImage& zpos, const PixelImage& zneg, uint flags = 0);

		/// Destructor.
		~TextureCubeMap(void);

		/** \return the texture handle. */
		GLuint	handle(void) const;

		/** \return the texture width. */
		uint	w(void) const;

		/** \return the texture height. */
		uint	h(void) const;

	private:

		/** Create the cubemap texture object. */
		void createCubeMap();

		/** Upload cubemap data.
		\param xpos positive X face
		\param xneg negative X face
		\param ypos positive Y face
		\param yneg negative Y face
		\param zpos positive Z face
		\param zneg negative Z face
		*/
		void sendCubeMap(const PixelImage& xpos, const PixelImage& xneg,
			const PixelImage& ypos, const PixelImage& yneg,
			const PixelImage& zpos, const PixelImage& zneg);

		GLuint  m_Handle = 0; ///< Texture handle.
		uint    m_W = 0; ///< Texture width.
		uint    m_H = 0; ///< Texture height.
		uint    m_Flags = 0; ///< Options.

	};


	/**
	Copy the content of a texture to another texture, resizing if needed.
	\param src source texture
	\param dst destination texture
	\param mask which part of the buffer to copy (color, depth, stencil).
	\param filter filtering mode if the two buffers have different dimensions (linear or nearest)
	\warning If the mask contains the depth or stencil, filter must be GL_NEAREST
	\ingroup sibr_graphics
	*/
	SIBR_GRAPHICS_EXPORT void			blit(const ITexture2D& src, const ITexture2D& dst, GLbitfield mask = GL_COLOR_BUFFER_BIT, GLenum filter = GL_LINEAR);


	/**
	Copy the content of a texture to a render target, resizing if needed.
	\param src source texture
	\param dst destination rendertarget
	\param mask which part of the buffer to copy (color, depth, stencil).
	\param filter filtering mode if the two buffers have different dimensions (linear or nearest)
	\param flip flip the texture vertically when copying it
	\note The blit can only happen for color attachment 0 in dst.
	\warning If the mask contains the depth or stencil, filter must be GL_NEAREST
	 \ingroup sibr_graphics
	*/
	SIBR_GRAPHICS_EXPORT void			blit(const ITexture2D& src, const IRenderTarget& dst, GLbitfield mask = GL_COLOR_BUFFER_BIT, GLenum filter = GL_LINEAR, bool flip = false);

	/**
	Copy the content of a texture to a render target, resizing if needed and flipping the result.
	\param src source texture
	\param dst destination rendertarget
	\param mask which part of the buffer to copy (color, depth, stencil).
	\param filter filtering mode if the two buffers have different dimensions (linear or nearest)
	\note The blit can only happen for color attachment 0 in dst.
	\warning If the mask contains the depth or stencil, filter must be GL_NEAREST
	 \ingroup sibr_graphics
	*/
	SIBR_GRAPHICS_EXPORT void			blit_and_flip(const ITexture2D& src, const IRenderTarget& dst, GLbitfield mask = GL_COLOR_BUFFER_BIT, GLenum filter = GL_LINEAR);

	/**
	Copy the content of a texture to a specific color attachment of the destination render target, resizing if needed.
	\param src source texture
	\param dst destination rendertarget
	\param location the color attachment to blit to
	\param filter filtering mode if the two buffers have different dimensions (linear or nearest)
	\param flip flip the texture vertically when copying it
	\note No mask to specify, as this is assumed to be COLOR.
	\ingroup sibr_graphics
	*/
	SIBR_GRAPHICS_EXPORT void			blitToColorAttachment(const ITexture2D& src, IRenderTarget& dst, int location, GLenum filter = GL_LINEAR, bool flip = false);
	
	/**
	Copy the content of a rendertarget first color attachment to a texture, resizing if needed.
	\param src source rendertarget
	\param dst destination texture
	\param mask which part of the buffer to copy (color, depth, stencil).
	\param filter filtering mode if the two buffers have different dimensions (linear or nearest)
	\note The blit can only happen for color attachment 0 in dst.
	\warning If the mask contains the depth or stencil, filter must be GL_NEAREST
	 \ingroup sibr_graphics
	*/
	SIBR_GRAPHICS_EXPORT void			blit(const IRenderTarget& src, const ITexture2D& dst, GLbitfield mask = GL_COLOR_BUFFER_BIT, GLenum filter = GL_LINEAR);

	/** Display a RenderTarget into a popup OpenCV window.
	\param rt the rendertarget to display
	\param winTitle the window title
	\ingroup sibr_graphics
	*/
	template <typename T_Type, unsigned T_NumComp>
	static void		show(const RenderTarget<T_Type, T_NumComp>& rt, const std::string& winTitle = "sibr::show()") {
		Image<T_Type, T_NumComp> img;
		rt.readBack(img);
		show(img, winTitle);
	}

	/** Display a texture into a popup OpenCV window.
	\param texture the texture to display
	\param winTitle the window title
	\ingroup sibr_graphics
	*/
	template <typename T_Type, unsigned T_NumComp>
	static void		show(const Texture2D<T_Type, T_NumComp>& texture, const std::string& winTitle = "sibr::show()") {
		Image<T_Type, T_NumComp> img(texture.w(), texture.h());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture.handle());

		glGetTexImage(GL_TEXTURE_2D, 0, sibr::GLFormat<T_Type, T_NumComp>::format, sibr::GLType<T_Type>::type, img.data());
		show(img, winTitle);
	}

	// --- TYPEDEFS --------------------------------------------------

	typedef Texture2D<unsigned char, 3>     Texture2DRGB;
	typedef Texture2D<unsigned char, 4>     Texture2DRGBA;
	typedef Texture2D<unsigned char, 1>     Texture2DLum;

	typedef Texture2D<unsigned short, 4>    Texture2DRGBA16;
	typedef Texture2D<unsigned short, 1>    Texture2DLum16;
	typedef Texture2D<unsigned short, 2>    Texture2DUV16;

	typedef Texture2D<short, 2>             Texture2DUV16s;

	typedef Texture2D<float, 3>             Texture2DRGB32F;
	typedef Texture2D<float, 4>             Texture2DRGBA32F;
	typedef Texture2D<float, 2>             Texture2DUV32F;
	typedef Texture2D<float, 1>             Texture2DLum32F;


	typedef Texture2DArray<unsigned char, 1>     Texture2DArrayLum;
	typedef Texture2DArray<unsigned char, 2>     Texture2DArrayUV;
	typedef Texture2DArray<unsigned char, 3>     Texture2DArrayRGB;
	typedef Texture2DArray<unsigned char, 4>     Texture2DArrayRGBA;

	typedef Texture2DArray<unsigned short, 1>    Texture2DArrayLum16;
	typedef Texture2DArray<unsigned short, 2>    Texture2DArrayUV16;
	typedef Texture2DArray<unsigned short, 3>    Texture2DArrayRGB16;
	typedef Texture2DArray<unsigned short, 4>    Texture2DArrayRGBA16;

	typedef Texture2DArray<short, 1>             Texture2DArrayLum16s;
	typedef Texture2DArray<short, 2>             Texture2DArrayUV16s;
	typedef Texture2DArray<short, 3>             Texture2DArrayRGB16s;
	typedef Texture2DArray<short, 4>             Texture2DArrayRGBA16s;

	typedef Texture2DArray<int, 1>				 Texture2DArrayInt1;
	typedef Texture2DArray<int, 2>				 Texture2DArrayInt2;
	typedef Texture2DArray<int, 3>				 Texture2DArrayInt3;
	typedef Texture2DArray<int, 4>				 Texture2DArrayInt4;

	typedef Texture2DArray<float, 1>             Texture2DArrayLum32F;
	typedef Texture2DArray<float, 2>             Texture2DArrayUV32F;
	typedef Texture2DArray<float, 3>             Texture2DArrayRGB32F;
	typedef Texture2DArray<float, 4>             Texture2DArrayRGBA32F;


	typedef TextureCubeMap<unsigned char, 1>    TextureCubeMapLum;
	typedef TextureCubeMap<unsigned char, 3>    TextureCubeMapRGB;
	typedef TextureCubeMap<unsigned char, 4>    TextureCubeMapRGBA;

	typedef TextureCubeMap<unsigned short, 1>   TextureCubeMapLum16;
	typedef TextureCubeMap<unsigned short, 2>   TextureCubeMapUV16;
	typedef TextureCubeMap<unsigned short, 4>   TextureCubeMapRGBA16;

	typedef TextureCubeMap<short, 2>            TextureCubeMapUV16s;

	typedef TextureCubeMap<float, 1>            TextureCubeMapLum32F;
	typedef TextureCubeMap<float, 3>            TextureCubeMapRGB32F;
	typedef TextureCubeMap<float, 4>            TextureCubeMapRGBA32F;

	/* Note concerning depth buffers :
	* We don't support depth only rendertargets.
	* Other kinds of RenderTarget (e.g. RenderTargetRGB) creates
	* also a new depth buffer that is bound with the color buffer, so no need to explicitely create one.
	* typedef RenderTarget<depth24,1>        RenderTargetDepth24;
	*/


	// ----DEFINITIONS Texture2D --------------------------------------------------

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	GLuint Texture2D<T_Type, T_NumComp>::create2D(const ImageType& img, uint flags) {
		GLuint id = 0;
		CHECK_GL_ERROR;
		glGenTextures(1, &id);
		glBindTexture(GL_TEXTURE_2D, id);
		if (flags & SIBR_CLAMP_UVS) {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}
		else if (flags & SIBR_CLAMP_TO_BORDER) {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		}
		if (flags & SIBR_GPU_AUTOGEN_MIPMAP) {
			if (flags & SIBR_GPU_INTEGER) {
				throw std::runtime_error("Mipmapping on integer texture not supported, probably not even by OpenGL");
			}
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
		else {
#if SIBR_COMPILE_FORCE_SAMPLING_LINEAR
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#else
			if (flags & SIBR_GPU_LINEAR_SAMPLING) {
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			}
			else {
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			}
#endif
		}
		send2D(id, img, flags);
		CHECK_GL_ERROR;
		return id;
	}

	template<typename T_Type, unsigned int T_NumComp>
	/*static*/ GLuint Texture2D<T_Type, T_NumComp>::create2D(const std::vector<PixelImage>& miparray, uint flags) {
		GLuint id = 0;
		CHECK_GL_ERROR;
		glGenTextures(1, &id);
		glBindTexture(GL_TEXTURE_2D, id);
		if (flags & SIBR_CLAMP_UVS) {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}
		else if (flags & SIBR_CLAMP_TO_BORDER) {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		}
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, int(miparray.size()) - 1);
		send2Dmipmap(id, miparray, flags);
		CHECK_GL_ERROR;
		return id;
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2D<T_Type, T_NumComp>::send2D(GLuint id, const ImageType& img, uint flags) {
		using FormatInfos = GLTexFormat<ImageType, T_Type, T_NumComp>;

		if (flags & SIBR_GPU_INTEGER) {
			if (FormatInfos::int_internal_format < 0) {
				throw std::runtime_error("Texture format does not support integer mapping");
			}
		}

		bool flip = flags & SIBR_FLIP_TEXTURE;
		ImageType flippedImg;
		if (flip) {
			flippedImg = FormatInfos::flip(img);
		}
		const ImageType& sendedImg = flip ? flippedImg : img;

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, id);
		glTexImage2D(GL_TEXTURE_2D,
			0,
			(flags & SIBR_GPU_INTEGER) ? FormatInfos::int_internal_format : FormatInfos::internal_format,
			FormatInfos::width(sendedImg), FormatInfos::height(sendedImg),
			0,
			(flags & SIBR_GPU_INTEGER) ? FormatInfos::int_format : FormatInfos::format,
			FormatInfos::type,
			FormatInfos::data(sendedImg)
		);

		bool autoMIPMAP = ((flags & SIBR_GPU_AUTOGEN_MIPMAP) != 0);
		if (autoMIPMAP)
			glGenerateMipmap(GL_TEXTURE_2D);
		CHECK_GL_ERROR;
	}

	// Send 2D texture to GPU memory, each mipmap is specified
	template<typename T_Type, unsigned int T_NumComp>
	/*static*/ void Texture2D<T_Type, T_NumComp>::send2Dmipmap(GLuint id, const std::vector<PixelImage>& miparray, uint flags) {
		CHECK_GL_ERROR;
		if (flags & SIBR_GPU_INTEGER) {
			throw std::runtime_error("Mipmapping on integer texture not supported, probably not even by OpenGL");
		}
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, id);

		std::vector<PixelImage> flippedMipArray;
		bool flip = flags & SIBR_FLIP_TEXTURE;
		if (flip) {
			flippedMipArray.resize(miparray.size());
#pragma omp parallel for
			for (uint l = 0; l < miparray.size(); l++) {
				flippedMipArray[l] = miparray[l].clone();
				flippedMipArray[l].flipH();
			}
		}
		const std::vector<PixelImage>& sendedMipArray = flip ? flippedMipArray : miparray;

		for (uint l = 0; l < miparray.size(); l++) {
			glTexImage2D(GL_TEXTURE_2D,
				l,
				GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::internal_format,
				miparray[l].w(), miparray[l].h(),
				0,
				GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::format,
				GLType<typename PixelFormat::Type>::type,
				sendedMipArray[l].data()
			);
		}
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp>
	Texture2D<T_Type, T_NumComp>::Texture2D(void) {
		m_Flags = 0;
		m_W = 0;
		m_H = 0;
		m_Handle = 0;
		m_autoMIPMAP = false;
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	Texture2D<T_Type, T_NumComp>::Texture2D(const ImageType& img, uint flags) {
		using TexFormat = GLTexFormat<ImageType, T_Type, T_NumComp>;
		m_Flags = flags;
		m_W = TexFormat::width(img);
		m_H = TexFormat::height(img);
		m_Handle = create2D(img, m_Flags);
		m_autoMIPMAP = ((flags & SIBR_GPU_AUTOGEN_MIPMAP) != 0);
	}

	template<typename T_Type, unsigned int T_NumComp>
	Texture2D<T_Type, T_NumComp>::Texture2D(const std::vector<PixelImage>& miparray, uint flags) {
		m_Flags = flags;
		m_W = miparray[0].w();
		m_H = miparray[0].h();
		m_Handle = create2D(miparray, m_Flags);
		m_autoMIPMAP = false;
	}

	template<typename T_Type, unsigned int T_NumComp>
	Texture2D<T_Type, T_NumComp>::~Texture2D(void) {
		CHECK_GL_ERROR;
		glDeleteTextures(1, &m_Handle);
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp>
	GLuint Texture2D<T_Type, T_NumComp>::handle(void) const { return m_Handle; }
	template<typename T_Type, unsigned int T_NumComp>
	uint   Texture2D<T_Type, T_NumComp>::w(void) const { return m_W; }
	template<typename T_Type, unsigned int T_NumComp>
	uint   Texture2D<T_Type, T_NumComp>::h(void) const { return m_H; }


	template<typename T_Type, unsigned int T_NumComp>
	sibr::Image<T_Type, T_NumComp>		Texture2D<T_Type, T_NumComp>::readBack(void) const {

		// makes sure Vertex have the correct size (read back relies on pointers)
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, handle());

		int w, h;
		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);

		sibr::Image<T_Type, T_NumComp> img(w, h);

		glGetTexImage(GL_TEXTURE_2D,
			0,
			GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::format,
			GLType<typename PixelFormat::Type>::type,
			img.data()
		);

		// flip data vertically to get origin on lower left corner
		img.flipH();

		CHECK_GL_ERROR;

		return img;
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2D<T_Type, T_NumComp>::update(const ImageType& img) {
		using FormatInfos = GLTexFormat<ImageType, T_Type, T_NumComp>;
		if (FormatInfos::width(img) == w() && FormatInfos::height(img) == h())
		{
			bool flip = m_Flags & SIBR_FLIP_TEXTURE;
			ImageType flippedImg;
			if (flip) {
				flippedImg = FormatInfos::flip(img);
			}
			const ImageType& sendedImg = flip ? flippedImg : img;

			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glBindTexture(GL_TEXTURE_2D, handle());
			glTexSubImage2D(GL_TEXTURE_2D, 0,
				0, 0, FormatInfos::width(sendedImg), FormatInfos::height(sendedImg),
				FormatInfos::format,
				FormatInfos::type,
				FormatInfos::data(sendedImg)
			);
			if (m_autoMIPMAP)
				glGenerateMipmap(GL_TEXTURE_2D);
		}
		else {
			m_W = FormatInfos::width(img);
			m_H = FormatInfos::height(img);
			send2D(m_Handle, img, m_Flags);
		}
	}

	template<typename T_Type, unsigned int T_NumComp>
	void Texture2D<T_Type, T_NumComp>::mipmap(int maxLOD) {
		glBindTexture(GL_TEXTURE_2D, handle());
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, maxLOD >= 0 ? maxLOD : 1000);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		m_autoMIPMAP = true;
		glGenerateMipmap(GL_TEXTURE_2D);
	}



	// ----DEFINITIONS Texture2DArray --------------------------------------------------

	template<typename T_Type, unsigned int T_NumComp>
	Texture2DArray<T_Type, T_NumComp>::Texture2DArray(const uint d, uint flags) {
		m_Depth = d;
		m_Flags = flags;
	}

	template<typename T_Type, unsigned int T_NumComp>
	Texture2DArray<T_Type, T_NumComp>::Texture2DArray(const uint w, const uint h, const uint d, uint flags) {
		m_W = w;
		m_H = h;
		m_Depth = d;
		m_Flags = flags;
		createArray();
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	Texture2DArray<T_Type, T_NumComp>::Texture2DArray(const std::vector<ImageType>& images, uint flags) {
		m_Flags = flags;
		createFromImages(images, flags);
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	Texture2DArray<T_Type, T_NumComp>::Texture2DArray(const std::vector<ImageType>& images, uint w, uint h, uint flags) {
		m_Flags = flags;
		createFromImages(images, w, h, flags);
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	Texture2DArray<T_Type, T_NumComp>::Texture2DArray(const std::vector<std::vector<ImageType>>& images, uint flags) {
		m_Flags = flags;
		createFromImages(images, flags);
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	Texture2DArray<T_Type, T_NumComp>::Texture2DArray(const std::vector<std::vector<ImageType>>& images, uint w, uint h, uint flags) {
		m_Flags = flags;
		createFromImages(images, w, h, flags);
	}

	template<typename T_Type, unsigned int T_NumComp>
	Texture2DArray<T_Type, T_NumComp>::Texture2DArray(const std::vector<typename PixelRT::Ptr>& RTs, uint flags) {
		m_Flags = flags;
		createFromRTs(RTs, flags);
	}

	template<typename T_Type, unsigned int T_NumComp>
	void Texture2DArray<T_Type, T_NumComp>::createArray(uint compression) {
		CHECK_GL_ERROR;
		glGenTextures(1, &m_Handle);
		glBindTexture(GL_TEXTURE_2D_ARRAY, m_Handle);

		const bool autoMIPMAP = ((m_Flags & SIBR_GPU_AUTOGEN_MIPMAP) != 0);
		const int numMipMap = autoMIPMAP ? (int)std::floor(std::log2(std::max(m_W, m_H))) : m_numLODs;

		m_numLODs = numMipMap;

		if (m_numLODs == 1) {
			if (m_Flags & SIBR_GPU_LINEAR_SAMPLING) {
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			}
			else {
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			}

		}
		else {
			if (m_Flags & SIBR_GPU_LINEAR_SAMPLING) {
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			}
			else {
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
				glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			}
		}

		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		uint internal_format = GLFormat<T_Type, T_NumComp>::internal_format;
		if (compression)
			internal_format = compression;

		glTexStorage3D(GL_TEXTURE_2D_ARRAY, numMipMap,
			internal_format,
			m_W,
			m_H,
			m_Depth
		);

		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::sendArray(const std::vector<ImageType>& images) {
		using ImgTypeInfo = GLTexFormat<ImageType, T_Type, T_NumComp>;
		glBindTexture(GL_TEXTURE_2D_ARRAY, m_Handle);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		// Make sure all images have the same size.
		std::vector<ImageType> tmp;
		std::vector<const ImageType*> imagesPtrToSend = applyFlipAndResize(images, tmp, m_W, m_H);

		for (int im = 0; im < (int)m_Depth; ++im) {
			glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
				0,
				0, 0, im,
				m_W,
				m_H,
				1, // one slice at a time
				ImgTypeInfo::format,
				ImgTypeInfo::type,
				ImgTypeInfo::data(*imagesPtrToSend[im])
			);
			//CHECK_GL_ERROR;
		}
		bool autoMIPMAP = ((m_Flags & SIBR_GPU_AUTOGEN_MIPMAP) != 0);
		if (autoMIPMAP) {
			glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
		}
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::sendMipArray(const std::vector<std::vector<ImageType>>& images) {
		using ImgTypeInfo = GLTexFormat<ImageType, T_Type, T_NumComp>;
		glBindTexture(GL_TEXTURE_2D_ARRAY, m_Handle);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		assert(m_numLODs == images.size());
		for (int lid = 0; lid < int(images.size()); ++lid) {

			assert(m_Depth == images[lid].size());

			// Make sure all images have the same size.
			const uint dW = m_W / (1 << lid);
			const uint dH = m_H / (1 << lid);
			std::vector<ImageType> tmp;
			std::vector<const ImageType*> imagesPtrToSend = applyFlipAndResize(images[lid], tmp, dW, dH);

			for (int im = 0; im < (int)m_Depth; ++im) {
				glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
					lid,
					0, 0, im,
					dW,
					dH,
					1, // one slice at a time
					ImgTypeInfo::format,
					ImgTypeInfo::type,
					ImgTypeInfo::data(*imagesPtrToSend[im])
				);
			}
		}
		// No auto mipmap when specifying the mips.
		m_Flags &= ~SIBR_GPU_AUTOGEN_MIPMAP;
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	std::vector<const ImageType*> Texture2DArray<T_Type, T_NumComp>::applyFlipAndResize(
		const std::vector<ImageType>& images,
		std::vector<ImageType>& tmp, uint tw, uint th,
		const std::vector<int>& slices)
	{
		using ImgTypeInfo = GLTexFormat<ImageType, T_Type, T_NumComp>;

		std::vector<const ImageType*> imagesPtrToSend(images.size());
		tmp.resize(images.size());

		bool flip = m_Flags & SIBR_FLIP_TEXTURE;
		//#pragma omp parallel for // Disabled due to performance reasons when live-updating slices.
		for (int slice_id = 0; slice_id < (int)slices.size(); ++slice_id) {
			int im = slices[slice_id];

			bool resize = !(tw == ImgTypeInfo::width(images[im]) && th == ImgTypeInfo::height(images[im]));
			if (!flip && !resize) {
				imagesPtrToSend[im] = &images[im];
			}
			else {
				if (resize) {
					tmp[im] = ImgTypeInfo::resize(images[im], tw, th);
				}
				if (flip) {
					tmp[im] = ImgTypeInfo::flip(resize ? tmp[im] : images[im]);
				}
				imagesPtrToSend[im] = &tmp[im];
			}
		}

		return imagesPtrToSend;
	}

	template<typename T_Type, unsigned int T_NumComp>
	template<typename ImageType>
	std::vector<const ImageType*> Texture2DArray<T_Type, T_NumComp>::applyFlipAndResize(
		const std::vector<ImageType>& images,
		std::vector<ImageType>& tmp, uint tw, uint th
	) {
		std::vector<int> slices(m_Depth);
		for (int i = 0; i < (int)m_Depth; ++i) {
			slices[i] = i;
		}
		return applyFlipAndResize(images, tmp, tw, th, slices);
	}

	template<typename T_Type, unsigned int T_NumComp>
	void Texture2DArray<T_Type, T_NumComp>::sendRTarray(const std::vector<typename PixelRT::Ptr>& RTs) {
		CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D_ARRAY, m_Handle);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		for (int im = 0; im < (int)m_Depth; ++im) {
			// Set correct RT as read-framebuffer.

			RTs[im]->bind();
			glCopyTexSubImage3D(GL_TEXTURE_2D_ARRAY,
				0,
				0, 0, im,
				0, 0,
				m_W,
				m_H
			);
			RTs[im]->unbind();
		}
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::createFromImages(const std::vector<ImageType>& images, uint flags) {
		using ImgTypeInfo = GLTexFormat<ImageType, T_Type, T_NumComp>;

		sibr::Vector2u maxSize(0, 0);
		for (const auto& img : images) {
			maxSize = maxSize.cwiseMax(sibr::Vector2u(ImgTypeInfo::width(img), ImgTypeInfo::height(img)));
		}
		createFromImages(images, maxSize[0], maxSize[1], flags);
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::createFromImages(const std::vector<ImageType>& images, uint w, uint h, uint flags) {
		m_W = w;
		m_H = h;
		m_Depth = (uint)images.size();
		m_Flags = flags;
		createArray();
		sendArray(images);
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::createCompressedFromImages(const std::vector<ImageType>& images, uint compression, uint flags) {
		using ImgTypeInfo = GLTexFormat<ImageType, T_Type, T_NumComp>;

		sibr::Vector2u maxSize(0, 0);
		for (const auto& img : images) {
			maxSize = maxSize.cwiseMax(sibr::Vector2u(ImgTypeInfo::width(img), ImgTypeInfo::height(img)));
		}
		createCompressedFromImages(images, maxSize[0], maxSize[1], compression, flags);
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::createCompressedFromImages(const std::vector<ImageType>& images, uint w, uint h, uint compression, uint flags) {
		m_W = w;
		m_H = h;
		m_Depth = (uint)images.size();
		m_Flags = flags;
		createArray(compression);
		sendArray(images);
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::createFromImages(const std::vector<std::vector<ImageType>>& images, uint flags) {
		using ImgTypeInfo = GLTexFormat<ImageType, T_Type, T_NumComp>;

		sibr::Vector2u maxSize(0, 0);
		for (const auto& img : images[0]) {
			maxSize = maxSize.cwiseMax(sibr::Vector2u(ImgTypeInfo::width(img), ImgTypeInfo::height(img)));
		}
		createFromImages(images, maxSize[0], maxSize[1], flags);
	}

	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::createFromImages(const std::vector<std::vector<ImageType>>& images, uint w, uint h, uint flags) {
		m_W = w;
		m_H = h;
		m_Depth = uint(images[0].size());
		m_Flags = flags & ~SIBR_GPU_AUTOGEN_MIPMAP;
		m_numLODs = uint(images.size());
		createArray();

		sendMipArray(images);
	}


	template<typename T_Type, unsigned int T_NumComp> template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::updateFromImages(const std::vector<ImageType>& images) {
		using ImgTypeInfo = GLTexFormat<ImageType, T_Type, T_NumComp>;

		sibr::Vector2u maxSize(0, 0);
		for (const auto& img : images) {
			maxSize = maxSize.cwiseMax(sibr::Vector2u(ImgTypeInfo::width(img), ImgTypeInfo::height(img)));
		}
		if (images.size() == m_Depth && m_W == maxSize[0] && m_H == maxSize[1]) {
			sendArray(images);
		}
		else {
			createFromImages(images, m_Flags);
		}
	}

	template<typename T_Type, unsigned int T_NumComp>  template<typename ImageType>
	void Texture2DArray<T_Type, T_NumComp>::updateSlices(const std::vector<ImageType>& images, const std::vector<int>& slices) {
		using ImgTypeInfo = GLTexFormat<ImageType, T_Type, T_NumComp>;

		int numSlices = (int)slices.size();
		if (numSlices == 0) {
			return;
		}

		sibr::Vector2u maxSize(0, 0);
		for (int i = 0; i < numSlices; ++i) {
			maxSize = maxSize.cwiseMax(sibr::Vector2u(ImgTypeInfo::width(images[slices[i]]), ImgTypeInfo::height(images[slices[i]])));
		}
		if (m_W != maxSize[0] || m_H != maxSize[1]) {
			m_W = maxSize[0];
			m_H = maxSize[1];
		}

		glBindTexture(GL_TEXTURE_2D_ARRAY, m_Handle);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		std::vector<ImageType> tmp;
		std::vector<const ImageType*> imagesPtrToSend = applyFlipAndResize(images, tmp, m_W, m_H, slices);

		for (int i = 0; i < numSlices; ++i) {
			glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
				0,
				0, 0, slices[i],
				m_W,
				m_H,
				1, // one slice at a time
				ImgTypeInfo::format,
				ImgTypeInfo::type,
				ImgTypeInfo::data(*imagesPtrToSend[slices[i]])
			);
		}
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp>
	void Texture2DArray<T_Type, T_NumComp>::createFromRTs(const std::vector<typename PixelRT::Ptr>& RTs, uint flags) {
		m_W = 0;
		m_H = 0;
		for (const auto& RT : RTs) {
			m_W = (std::max)(m_W, RT->w());
			m_H = (std::max)(m_H, RT->h());
		}
		m_Depth = (uint)RTs.size();
		m_Flags = flags;
		createArray();
		sendRTarray(RTs);
	}

	template<typename T_Type, unsigned int T_NumComp>
	Texture2DArray<T_Type, T_NumComp>::~Texture2DArray(void) {
		CHECK_GL_ERROR;
		glDeleteTextures(1, &m_Handle);
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp>
	GLuint Texture2DArray<T_Type, T_NumComp>::handle(void) const { return m_Handle; }

	template<typename T_Type, unsigned int T_NumComp>
	uint Texture2DArray<T_Type, T_NumComp>::w(void) const { return m_W; }

	template<typename T_Type, unsigned int T_NumComp>
	uint Texture2DArray<T_Type, T_NumComp>::h(void) const { return m_H; }

	template<typename T_Type, unsigned int T_NumComp>
	uint Texture2DArray<T_Type, T_NumComp>::depth(void) const { return m_Depth; }

	template<typename T_Type, unsigned int T_NumComp>
	uint Texture2DArray<T_Type, T_NumComp>::numLODs(void) const { return m_numLODs; }

	template<typename T_Type, unsigned int T_NumComp>
	Vector4f	Texture2DArray<T_Type, T_NumComp>::readBackPixel(int i, int x, int y, uint lod) const {
		Vector4f out;
//#define HEADLESS
#ifdef HEADLESS
		SIBR_ERR << "HEADLESS -- No support for readBackPixel" << std::endl;
#else
		glGetTextureSubImage(handle(),
			lod, x, y, i, 1, 1, 1,
			GL_RGBA, GL_FLOAT, 4 * sizeof(float), out.data()
		);
#endif
		CHECK_GL_ERROR;
		for (uint c = T_NumComp; c < 4; ++c) {
			out[c] = 0;
		}
		return out;
	}


	// ----DEFINITIONS TextureCubeMap --------------------------------------------------

	template<typename T_Type, unsigned int T_NumComp>
	TextureCubeMap<T_Type, T_NumComp>::TextureCubeMap(void) {}

	template<typename T_Type, unsigned int T_NumComp>
	TextureCubeMap<T_Type, T_NumComp>::TextureCubeMap(const uint w, const uint h, uint flags) {
		m_W = w;
		m_H = h;
		m_Flags = flags;
		createCubeMap();
	}

	template<typename T_Type, unsigned int T_NumComp>
	TextureCubeMap<T_Type, T_NumComp>::TextureCubeMap(const PixelImage& xpos, const PixelImage& xneg,
		const PixelImage& ypos, const PixelImage& yneg,
		const PixelImage& zpos, const PixelImage& zneg, uint flags) {
		m_Flags = flags;
		createFromImages(xpos, xneg, ypos, yneg, zpos, zneg, flags);
	}


	template<typename T_Type, unsigned int T_NumComp>
	void TextureCubeMap<T_Type, T_NumComp>::createCubeMap() {

		// We enable seamless junctions between cubemap faces.
		static bool enableStates = false;
		if (enableStates == false)
		{
			glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
			enableStates = true;
		}
		CHECK_GL_ERROR;

		glGenTextures(1, &m_Handle);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_Handle);

		if (m_Flags & SIBR_GPU_LINEAR_SAMPLING) {
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
		else {
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		}

		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		CHECK_GL_ERROR;
	}



	template<typename T_Type, unsigned int T_NumComp>
	void TextureCubeMap<T_Type, T_NumComp>::sendCubeMap(const PixelImage& xpos, const PixelImage& xneg,
		const PixelImage& ypos, const PixelImage& yneg,
		const PixelImage& zpos, const PixelImage& zneg) {
		CHECK_GL_ERROR;

		if (m_Flags & SIBR_GPU_INTEGER) {
			if (GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_internal_format < 0) {
				throw std::runtime_error("Texture format does not support integer mapping");
			}
		}

		// Handle flipping.
		const PixelImage* sendedXpos = &xpos;
		const PixelImage* sendedYpos = &ypos;
		const PixelImage* sendedZpos = &zpos;
		const PixelImage* sendedXneg = &xneg;
		const PixelImage* sendedYneg = &yneg;
		const PixelImage* sendedZneg = &zneg;

		PixelImage flippedXpos, flippedYpos, flippedZpos;
		PixelImage flippedXneg, flippedYneg, flippedZneg;

		// ...
		if (m_Flags & SIBR_FLIP_TEXTURE) {
			flippedXpos = xpos.clone();
			flippedXpos.flipH();
			sendedXpos = &flippedXpos;

			flippedYpos = ypos.clone();
			flippedYpos.flipH();
			sendedYpos = &flippedYpos;

			flippedZpos = zpos.clone();
			flippedZpos.flipH();
			sendedZpos = &flippedZpos;

			flippedXneg = xneg.clone();
			flippedXneg.flipH();
			sendedXneg = &flippedXneg;

			flippedYneg = yneg.clone();
			flippedYneg.flipH();
			sendedYneg = &flippedYneg;

			flippedZneg = zneg.clone();
			flippedZneg.flipH();
			sendedZneg = &flippedZneg;
		}

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		glBindTexture(GL_TEXTURE_CUBE_MAP, m_Handle);

		const auto tinternal_format = (m_Flags & SIBR_GPU_INTEGER)
			? GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_internal_format
			: GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::internal_format;
		const auto tformat = (m_Flags & SIBR_GPU_INTEGER)
			? GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_format
			: GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::format;
		const auto ttype = GLType<typename PixelFormat::Type>::type;

		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, tinternal_format, xpos.w(), xpos.h(), 0, tformat, ttype, xpos.data());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, tinternal_format, xneg.w(), xneg.h(), 0, tformat, ttype, xneg.data());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, tinternal_format, ypos.w(), ypos.h(), 0, tformat, ttype, ypos.data());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, tinternal_format, yneg.w(), yneg.h(), 0, tformat, ttype, yneg.data());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, tinternal_format, zpos.w(), zpos.h(), 0, tformat, ttype, zpos.data());
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, tinternal_format, zneg.w(), zneg.h(), 0, tformat, ttype, zneg.data());


		bool autoMIPMAP = ((m_Flags & SIBR_GPU_AUTOGEN_MIPMAP) != 0);
		if (autoMIPMAP) {
			glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
		}


	}


	template<typename T_Type, unsigned int T_NumComp>
	void TextureCubeMap<T_Type, T_NumComp>::createFromImages(const PixelImage& xpos, const PixelImage& xneg,
		const PixelImage& ypos, const PixelImage& yneg,
		const PixelImage& zpos, const PixelImage& zneg, uint flags) {
		const int numMipMap = 1;
		sibr::Vector2u maxSize(0, 0);
		/// \todo TODO: check if the six images have the same size.
		m_W = xpos.w();
		m_H = xpos.h();
		m_Flags = flags;
		createCubeMap();
		sendCubeMap(xpos, xneg, ypos, yneg, zpos, zneg);
	}


	template<typename T_Type, unsigned int T_NumComp>
	TextureCubeMap<T_Type, T_NumComp>::~TextureCubeMap(void) {
		CHECK_GL_ERROR;
		glDeleteTextures(1, &m_Handle);
		CHECK_GL_ERROR;
	}

	template<typename T_Type, unsigned int T_NumComp>
	GLuint TextureCubeMap<T_Type, T_NumComp>::handle(void) const { return m_Handle; }

	template<typename T_Type, unsigned int T_NumComp>
	uint TextureCubeMap<T_Type, T_NumComp>::w(void) const { return m_H; }

	template<typename T_Type, unsigned int T_NumComp>
	uint TextureCubeMap<T_Type, T_NumComp>::h(void) const { return m_W; }

} // namespace sibr


