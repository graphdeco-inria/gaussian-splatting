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


# define SIBR_GPU_AUTOGEN_MIPMAP		(1<<0)
# define SIBR_GPU_MULSTISAMPLE			(1<<1)
# define SIBR_GPU_LINEAR_SAMPLING		(1<<2)
# define SIBR_GPU_INTEGER				(1<<4)
# define SIBR_MSAA4X					(1<<5)
# define SIBR_MSAA8X					(1<<6)
# define SIBR_MSAA16X					(1<<7)
# define SIBR_MSAA32X					(1<<8)
# define SIBR_STENCIL_BUFFER			(1<<9)
# define SIBR_CLAMP_UVS					(1<<10)
# define SIBR_CLAMP_TO_BORDER			(1<<11)
# define SIBR_FLIP_TEXTURE				(1<<12)

# define SIBR_COMPILE_FORCE_SAMPLING_LINEAR	0

namespace sibr{


	/**
	* Contain type utilities to match C, cv and sibr types to OpenGL formats.
	* \addtogroup sibr_graphics
	* @{
	*/

	// --- TYPE HELPERS ---------------------------------------------------

	/** Helper building the correspondence between a GL type and a C type. */
	template <typename T> class GLType;

	/** Helper building the correspondence between a GL type and a C type. */
	template <> class GLType<unsigned char> {
	public:
		enum { type = GL_UNSIGNED_BYTE };
	};

	/** Helper building the correspondence between a GL type and a C type. */
	template <> class GLType<unsigned short> {
	public:
		enum { type = GL_UNSIGNED_SHORT };
	};

	/** Helper building the correspondence between a GL type and a C type. */
	template <> class GLType<short> {
	public:
		enum { type = GL_SHORT };
	};

	/** Helper building the correspondence between a GL type and a C type. */
	template <> class GLType<float> {
	public:
		enum { type = GL_FLOAT };
	};

	/** Helper building the correspondence between a GL type and a C type. */
	template <> class GLType<int> {
	public:
		enum { type = GL_INT };
	};

	// --- FORMAT HELPERS -------------------------------------------------------

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <typename T_Type,int T_Num> class GLFormat;

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<unsigned char,1> {
	public:
		enum {
			internal_format = GL_R8,
			format = GL_RED,
			int_internal_format = GL_R8UI,
			int_format = GL_RED_INTEGER,
			isdepth = 0
	};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<unsigned char, 2> {
	public:
		enum {
			internal_format = GL_RG8,
			format = GL_RG,
			int_internal_format = GL_RG8UI,
			int_format = GL_RG_INTEGER,
			isdepth = 0
		};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<unsigned char,3> {
	public:
		enum {
			internal_format = GL_RGB8,
			format = GL_RGB,
			int_internal_format = GL_RGB8UI,
			int_format = GL_RGB_INTEGER,
			isdepth = 0
	};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<unsigned char,4> {
	public:
		enum {
			internal_format = GL_RGBA8,
			format = GL_RGBA,
			int_internal_format = GL_RGBA8UI,
			int_format = GL_RGBA_INTEGER,
			isdepth = 0
	};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<unsigned short,1> {
	public:
		enum {
			internal_format = GL_R16,
			format = GL_R,
			int_internal_format = GL_R16UI,
			int_format = GL_RED_INTEGER,
			isdepth = 0
	};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<unsigned short, 2> {
	public:
		enum {
			internal_format = GL_RG16,
			format = GL_RG,
			int_internal_format = GL_RG16UI,
			int_format = GL_RG_INTEGER,
			isdepth = 0
		};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<unsigned short,3> {
	public:
		enum {
			internal_format = GL_RGB16,
			format = GL_RGB,
			int_internal_format = GL_RGB16UI,
			int_format = GL_RGB_INTEGER,
			isdepth = 0
	};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<unsigned short,4> {
	public:
		enum {
			internal_format = GL_RGBA16,
			format = GL_RGBA,
			int_internal_format = GL_RGBA16UI,
			int_format = GL_RGBA_INTEGER,
			isdepth = 0
	};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<int,1> {
	public:
		enum {
			internal_format = GL_R32I,
			format = GL_RED_INTEGER,
			int_internal_format = GL_R32I,
			int_format = GL_RED_INTEGER,
			isdepth = 0
	};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<int, 2> {
	public:
		enum {
			internal_format = GL_RG32I,
			format = GL_RG_INTEGER,
			int_internal_format = GL_RG32I,
			int_format = GL_RG_INTEGER,
			isdepth = 0
		};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<int, 3> {
	public:
		enum {
			internal_format = GL_RGB32I,
			format = GL_RGB_INTEGER,
			int_internal_format = GL_RGB32I,
			int_format = GL_RGB_INTEGER,
			isdepth = 0
		};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<int, 4> {
	public:
		enum {
			internal_format = GL_RGBA32I,
			format = GL_RGBA_INTEGER,
			int_internal_format = GL_RGBA32I,
			int_format = GL_RGBA_INTEGER,
			isdepth = 0
		};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<float,1> {
	public:
		enum {
			internal_format = GL_R32F,
			format = GL_RED,
			int_internal_format = -1,
			int_format = -1,
			isdepth = 0
	};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<float, 2> {
	public:
		enum {
			internal_format = GL_RG32F,
			format = GL_RG,
			int_internal_format = -1,
			int_format = -1,
			isdepth = 0
		};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<float,3> {
	public:
		enum {
			internal_format = GL_RGB32F,
			format = GL_RGB,
			int_internal_format = -1,
			int_format = -1,
			isdepth = 0
	};
	};

	/** Helper building the correspondence between a GL format and a C type and number of components. */
	template <> class GLFormat<float,4> {
	public:
		enum {
			internal_format = GL_RGBA32F,
			format = GL_RGBA,
			int_internal_format = -1,
			int_format = -1,
			isdepth = 0
	};
	};

	// Depth texture format (unsupported)

	//template <> class GLFormat<depth32,1> {
	//public:
	//	enum {
	//		internal_format     = GL_DEPTH_COMPONENT32F,
	//		format              = GL_DEPTH_COMPONENT,
	//		int_internal_format = -1,
	//		int_format          = -1,
	//		isdepth             =  1};
	//};

	// --- MAT HELPERS -----------------------

	/** Helper building the correspondence between a GL format and a cv::Mat */
	template <typename T_Type, int T_Num> class GLFormatCVmat;

	/** Helper building the correspondence between a GL format and a cv::Mat */
	template <> class GLFormatCVmat<unsigned char, 1> {
	public:
		enum {
			internal_format = GLFormat<uchar, 1>::internal_format,
			format = GLFormat<uchar, 1>::format,
			int_internal_format = GLFormat<uchar, 1>::int_internal_format,
			int_format = GLFormat<uchar, 1>::int_format,
			isdepth = GLFormat<uchar, 1>::isdepth
		};
	};

	/** Helper building the correspondence between a GL format and a cv::Mat */
	template <> class GLFormatCVmat<unsigned char, 3> {
	public:
		enum {
			internal_format = GLFormat<uchar,3>::internal_format,
			format = GL_BGR,
			int_internal_format = GLFormat<uchar, 3>::int_internal_format,
			int_format = GLFormat<uchar, 3>::int_format,
			isdepth = GLFormat<uchar, 3>::isdepth
		};
	};

	/** Helper building the correspondence between a GL format and a cv::Mat */
	template <> class GLFormatCVmat<unsigned char, 4> {
	public:
		enum {
			internal_format = GLFormat<uchar, 4>::internal_format,
			format = GL_BGRA,
			int_internal_format = GLFormat<uchar, 4>::int_internal_format,
			int_format = GLFormat<uchar, 4>::int_format,
			isdepth = GLFormat<uchar, 4>::isdepth
		};
	};
	
	/** Helper building the correspondence between a GL type and a cv::Mat depth. */
	template<typename T> struct OpenCVdepth;

	/** Helper building the correspondence between a GL type and a cv::Mat depth. */
	template<> struct OpenCVdepth<uchar> {
		static const uint value = CV_8U;
	};

	/** Helper building the correspondence between a GL type and a cv::Mat depth. */
	template<> struct OpenCVdepth<float> {
		static const uint value = CV_32F;
	};

	/** Helper building the correspondence between a GL type and a cv::Mat depth. */
	template<> struct OpenCVdepth<double> {
		static const uint value = CV_64F;
	};

	/** Helper to create a cv::Mat type from its depth and number of components. */
	template<typename T, uint N> constexpr uint getOpenCVtype = CV_MAKE_TYPE(OpenCVdepth<T>::value, N);
	
	/** Helper to create a one-channel cv::Mat from its depth. */
	template<typename T> constexpr uint getOpenCVtypeSingleChannel = getOpenCVtype<T, 1>;
	
	/** Helper class to specify for which image type we can find a valid texture format. */
	template<typename ImageType> struct ValidGLTexFormat {
		static const bool value = false;
	};

	/** Helper class to specify for which image type we can find a valid texture format. */
	template<typename ScalarType, uint N> struct ValidGLTexFormat<sibr::Image<ScalarType, N>> {
		static const bool value = true;
	};

	/** Helper class to specify for which image type we can find a valid texture format. */
	template<> struct ValidGLTexFormat<cv::Mat> {
		static const bool value = true;
	};

	/** Helper class to provide, from a generic image type, all the information needed for OpenGL textures
		Right now it can work with all sibr::Image and with cv::Mat (3U8 only)
		You can add more using explicit template instanciation to specify both
		ValidGLTexFormat and the following GLTexFormat properties.
		*/
	template<typename ImageType, typename ScalarType = typename ImageType::Type, uint N = ImageType::e_NumComp> struct GLTexFormat {
		static_assert(ValidGLTexFormat<ImageType>::value, "ImageWrapper currently only specialized for sibr::Image and cv::Mat ");

		/** Flip an image.
		\param img image to flip
		\return the fliped image.
		*/
		static ImageType flip(const ImageType& img);

		/** Resize an image.
		\param img image to resize
		\param w new width
		\param h new height
		\return the resize image.
		*/
		static ImageType resize(const ImageType& img, uint w, uint h);

		/** Get an image width.
		\param img the image
		\return the width
		*/
		static uint width(const ImageType& img);

		/** Get an image height.
		\param img the image
		\return the height
		*/
		static uint height(const ImageType& img);

		/** Get an image data.
		\param img the image
		\return pointer to the beginning of the data.
		*/
		static const void* data(const ImageType& img);

		static const uint internal_format;		///< Internal GL format.
		static const uint format;				///< Generic GL format.
		static const uint int_internal_format;	///< Internal GL format for integer textures.
		static const uint int_format;			///< Generic GL format for integer textures.
		static const uint isdepth;				///< Is it a depth format.
		static const uint type;					 ///< The component GL type.
	};

	/** Helper class to provide, for an sibr::Image, all the information needed for OpenGL textures. */
	template<typename ScalarType, uint N > struct GLTexFormat<sibr::Image<ScalarType, N>, ScalarType, N > {
		using ImageType = sibr::Image<ScalarType, N>;

		/** \copydoc GLTexFormat::flip */
		static ImageType flip(const ImageType& img) {
			ImageType temp = img.clone();
			temp.flipH();
			return temp;
		}

		/** \copydoc GLTexFormat::resize */
		static ImageType resize(const ImageType& img, uint w, uint h) {
			return img.resized(w, h);
		}

		/** \copydoc GLTexFormat::width */
		static uint width(const ImageType& img) {
			return img.w();
		}

		/** \copydoc GLTexFormat::height */
		static uint height(const ImageType& img) {
			return img.h();
		}

		/** \copydoc GLTexFormat::data */
		static const void* data(const ImageType& img) {
			return img.data();
		}

		static const uint internal_format = GLFormat<ScalarType, N>::internal_format; ///< Internal GL format.
		static const uint format = GLFormat<ScalarType, N>::format;  ///< Generic GL format.
		static const uint int_internal_format = GLFormat<ScalarType, N>::int_internal_format; ///< Internal GL format for integer textures.
		static const uint int_format = GLFormat<ScalarType, N>::int_format;  ///< Generic GL format for integer textures.
		static const uint isdepth = GLFormat<ScalarType, N>::isdepth; ///< Is it a depth format.
		static const uint type = GLType<ScalarType>::type;  ///< The component GL type.
	};

	/** Helper class to provide, for an sibr::Image::Ptr, all the information needed for OpenGL textures. */
	template<typename ScalarType, uint N > struct GLTexFormat<ImagePtr<ScalarType, N>, ScalarType, N > {
		using ImageType = ImagePtr<ScalarType, N>;

		/** \copydoc GLTexFormat::flip */
		static ImageType flip(const ImageType& img) {
			ImageType temp = ImageType::fromImg(*img);
			temp->flipH();
			return temp;
		}

		/** \copydoc GLTexFormat::resize */
		static ImageType resize(const ImageType& img, uint w, uint h) {
			return ImageType::fromImg(img->resized(w, h));
		}

		/** \copydoc GLTexFormat::width */
		static uint width(const ImageType& img) {
			return img->w();
		}

		/** \copydoc GLTexFormat::height */
		static uint height(const ImageType& img) {
			return img->h();
		}

		/** \copydoc GLTexFormat::data */
		static const void* data(const ImageType& img) {
			return img->data();
		}

		static const uint internal_format = GLFormat<ScalarType, N>::internal_format; ///< Internal GL format.
		static const uint format = GLFormat<ScalarType, N>::format;  ///< Generic GL format.
		static const uint int_internal_format = GLFormat<ScalarType, N>::int_internal_format; ///< Internal GL format for integer textures.
		static const uint int_format = GLFormat<ScalarType, N>::int_format;  ///< Generic GL format for integer textures.
		static const uint isdepth = GLFormat<ScalarType, N>::isdepth; ///< Is it a depth format.
		static const uint type = GLType<ScalarType>::type;  ///< The component GL type.

	};

	/** Helper class to provide, for a cv::Mat, all the information needed for OpenGL textures. */
	template<typename ScalarType, uint N > struct GLTexFormat<cv::Mat, ScalarType, N> {
		static_assert(std::is_same_v<ScalarType, uchar> && (N == 3 || N == 4 || N == 1) , "GLTexFormat with cv::Mat currently only defined for 3U8 or 4U8");

		/** \copydoc GLTexFormat::flip */
		static cv::Mat flip(const cv::Mat& img) {
			cv::Mat temp;
			cv::flip(img, temp, 0); //0 for flipH
			return temp;
		}

		/** \copydoc GLTexFormat::resize */
		static cv::Mat resize(const cv::Mat& img, uint w, uint h) {
			cv::Mat temp;
			cv::resize(img, temp, cv::Size(w, h));
			return temp;
		}

		/** \copydoc GLTexFormat::width */
		static uint width(const cv::Mat& img) {
			return img.cols;
		}

		/** \copydoc GLTexFormat::height */
		static uint height(const cv::Mat& img) {
			return img.rows;
		}

		/** \copydoc GLTexFormat::data */
		static const void* data(const cv::Mat& img) {
			return img.ptr();
		}

		/** \copydoc GLTexFormat::data */
		static void* data(cv::Mat& img) {
			return img.ptr();
		}

		/** \return the matrix OpenCV type. */
		static uint cv_type() {
			return CV_MAKE_TYPE(cv::DataDepth<ScalarType>::value, N);
		}

		static const uint internal_format = GLFormatCVmat<ScalarType, N>::internal_format; ///< Internal GL format.
		static const uint format = GLFormatCVmat<ScalarType, N>::format;  ///< Generic GL format.
		static const uint int_internal_format = GLFormatCVmat<ScalarType, N>::int_internal_format; ///< Internal GL format for integer textures.
		static const uint int_format = GLFormatCVmat<ScalarType, N>::int_format;  ///< Generic GL format for integer textures.
		static const uint isdepth = GLFormatCVmat<ScalarType, N>::isdepth; ///< Is it a depth format.
		static const uint type = GLType<ScalarType>::type;  ///< The component GL type.
	};

	/** Helper class to provide, for a cv::Mat, all the information needed for OpenGL textures. */
	template<typename ScalarType, uint N >
	struct GLTexFormat<cv::Mat_<cv::Vec<ScalarType, N> >, ScalarType, N>
		: GLTexFormat<cv::Mat, ScalarType, N>
	{
	};

	/*** @} */
}
