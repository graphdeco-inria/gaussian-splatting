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
# include "core/system/ByteStream.hpp"

# pragma warning(push, 0)
#  include <opencv2/core/core.hpp>
#  include <opencv2/imgproc/imgproc.hpp>
#  include <opencv2/highgui/highgui.hpp>
#  include <boost/filesystem.hpp>
# pragma warning(pop)

#include <utility>
#include <vector>

namespace cv
{
	/** Extend OpenCV support for Eigen types. 
	\ingroup sibr_graphics
	*/
	/*template <typename T_Type, int cn>
	class DataType<Eigen::Matrix<T_Type, cn, 1, Eigen::DontAlign> >
	{
	public:
		typedef Eigen::Matrix<T_Type, cn, 1, Eigen::DontAlign> value_type; ///< Vector type.
		typedef Eigen::Matrix<typename DataType<T_Type>::work_type, cn, 1, Eigen::DontAlign> work_type; ///< Wrapper type.
		typedef T_Type channel_type; ///< Component type.
		typedef value_type vec_type; ///< Vector type.
		enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = cn, fmt = ((channels - 1) << 8) + DataDepth<channel_type>::fmt, type = CV_MAKETYPE(depth, channels) };
	};*/
}

namespace sibr
{

	/**
	* \addtogroup sibr_graphics
	* @{
	*/

	namespace opencv
	{
		/** \return the OpenCV type corresponding to a C type. */
		template <typename T_Type>
		SIBR_GRAPHICS_EXPORT int		imageType(void);// { return -1; } // default, unknown
		/** \return the OpenCV type corresponding to a C type. */
		template <> SIBR_GRAPHICS_EXPORT inline int		imageType< uint8 >(void) { return CV_8U; }
		/** \return the OpenCV type corresponding to a C type. */
		template <> SIBR_GRAPHICS_EXPORT inline int		imageType< int8  >(void) { return CV_8S; }
		/** \return the OpenCV type corresponding to a C type. */
		template <> SIBR_GRAPHICS_EXPORT inline int		imageType< uint16>(void) { return CV_16U; }
		/** \return the OpenCV type corresponding to a C type. */
		template <> SIBR_GRAPHICS_EXPORT inline int		imageType< int16 >(void) { return CV_16S; }
		/** \return the OpenCV type corresponding to a C type. */
		template <> SIBR_GRAPHICS_EXPORT inline int		imageType< int32 >(void) { return CV_32S; }
		/** \return the OpenCV type corresponding to a C type. */
		template <> SIBR_GRAPHICS_EXPORT inline int		imageType< float >(void) { return CV_32F; }
		/** \return the OpenCV type corresponding to a C type. */
		template <> SIBR_GRAPHICS_EXPORT inline int		imageType< double>(void) { return CV_64F; }

		/** \return the size of the range of values a type can take when used in OpenCV. */
		template <typename T_Type>
		inline float			imageTypeRange(void) {
			return (float)std::numeric_limits<T_Type>::max();//-std::numeric_limits<T_Type>::min();
		}
		/** \return the size of the range of values a type can take when used in OpenCV. */
		template <> SIBR_GRAPHICS_EXPORT inline float			imageTypeRange< float >(void) { return 1.f; }
		/** \return the size of the range of values a type can take when used in OpenCV. */
		template <> SIBR_GRAPHICS_EXPORT inline float			imageTypeRange< double>(void) { return 1.f; }

		/** Get the size of the range of values an OpenCV type can take.
		\param cvDepth the OpenCV type depth
		\return the size of the range
		*/
		SIBR_GRAPHICS_EXPORT float			imageTypeCVRange(int cvDepth);

		/** Convert a BGR cv::Mat into a RGB cv::Mat, in-place.
		\param dst the matrix to convert
		*/
		SIBR_GRAPHICS_EXPORT void			convertBGR2RGB(cv::Mat& dst);

		/** Convert a BGR cv::Mat into a RGB cv::Mat, in-place.
		\param dst the matrix to convert
		*/
		SIBR_GRAPHICS_EXPORT void			convertRGB2BGR(cv::Mat& dst);
	}

	typedef	Vector4f ColorRGBA;

	/** @} */

	/**
	* Interface virtual class for all the templated image classes.
	* Contains all functions not making reference to the internal type or numComp in their signature/return type
	* \sa Image
	* \ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT IImage {
	public:
		SIBR_CLASS_PTR(IImage);

		/** Load an image from the disk (png, jpeg, exr, etc., see OpenCV cv::imread documentation for more details).
		\param filename the path to the file
		\param verbose display additional informations
		\param warning_if_not_found log if the file doesn't exist, even if verbose is set to false
		\return a success flag
		*/
		virtual bool			load(const std::string& filename, bool verbose = true, bool warning_if_not_found = true) = 0;
		
		/** Load an image from the disk (stored as a raw binary blob).
		\param filename the path to the file
		\param verbose display additional informations
		\return a success flag
		*/
		virtual bool			loadByteStream(const std::string& filename, bool verbose = true) = 0;

		/** Save an image to the disk (png, jpeg, see OpenCV cv::imwrite documentation for more details).
		\param filename the path to the file
		\param verbose display additional informations
		\warning HDR images will be converted to LDR, \sa saveHDR .
		\warning Some image formats can't be stored in some file formats.
		*/
		virtual void			save(const std::string& filename, bool verbose = true) const = 0;

		/** Save an image to the disk (as a raw binary blob).
		\param filename the path to the file
		\param verbose display additional informations
		*/
		virtual void			saveByteStream(const std::string& filename, bool verbose = true) const = 0;

		/** Save an HDR image to the disk (exr, hdr, see OpenCV cv::imwrite documentation for more details).
		\param filename the path to the file
		\param verbose display additional informations
		*/
		virtual void			saveHDR(const std::string& filename, bool verbose = true) const = 0;

		/** \return the image width. */
		virtual uint			w(void) const = 0;

		/** \return the image height. */
		virtual uint			h(void) const = 0;

		/** \return the image size. */
		virtual sibr::Vector2u	size(void) const = 0;

		/** Check if a pixel (x,y) is inside the image boundaries.
		\param xy the pixel coordinates
		\return true if 0<=x<w and 0<=y=<h */
		virtual bool			isInRange(const ::sibr::Vector2i & xy)  const = 0;

		/** Get the value stored at a pixel and convert it to a string representation.
		\param xy the pixel coordinates
		\return a string representation of the pixel value.
		*/
		virtual std::string		pixelStr(const ::sibr::Vector2i & xy)  const = 0;

		/** \return the number of components of the image. */
		virtual uint			numComp(void) const = 0;

		/** \return the size of a pixel value in bytes. */
		virtual uint			sizeOfComp(void) const = 0;

		/** Flip the image along the horizontal axis. */
		virtual void			flipH(void) = 0;

		/** Flip the image along the vertical axis. */
		virtual void			flipV(void) = 0;

		/** \return the image OpenCV type. */
		virtual int				opencvType(void) const = 0;

		/** \return a reference to the underlying OpenCV matrix. */
		virtual const cv::Mat&	toOpenCV(void) const = 0;

		/** \return a reference to the underlying OpenCV matrix. */
		virtual cv::Mat&		toOpenCVnonConst(void) = 0;

		/** \return a copy of the matrix with channels flipped.
		\note Only applies to 3 and 4 channel images.
		*/
		virtual cv::Mat			toOpenCVBGR(void) const = 0;

		/** Replace the content of the image with the content of another matrix.
		\param img the new matrix
		*/
		virtual void			fromOpenCV(const cv::Mat& img) = 0;

		/** Replace the content of the image with the content of another matrix, flipping channels.
		\param img the new matrix
		\note Only applies to 3 and 4 channel images.
		*/
		virtual void			fromOpenCVBGR(const cv::Mat& img) = 0;

		/** Get the size of jpeg image file by reading its header.
		\param file the input filestream, already opened.
		\return The size (width,heighgt) else (-1, -1) if the header cannot be read .
		*/
		static sibr::Vector2i			get_jpeg_size(std::ifstream& file);

		/** Get the size of an image file from its header. Supported file type: {png, jpg, jpeg, bmp, tga}.
		\param file_path the input file path
		\return The size (width,heighgt) else (-1, -1) if the header cannot be read .
		*/
		static sibr::Vector2i			imageResolution(const std::string& file_path);

	};


	/** Wrapper around an image pointer.
	* \ingroup sibr_graphics */
	template<typename T_Type, unsigned int T_NumComp>
	class ImagePtr;

	/**
	* This class is used to store images. Internally, a cv::Mat
	* is used. The template parameter define a fixed size/format that
	* will be used to convert automatically the image format when
	* you load or copy from another image.
	* \warning We disallow copy as we would have to do a costly in-depth copy of the underlying cv::Mat.
	* If you store images in a vector attribute of a class, you might have to SIBR_DISALLOW_COPY of your class.
	* \note OpenCV uses generally BGR channels (e.g. after loading an image file). 
	* However the internal cv::Mat of this class stores
	* RGB channels. You can get RGB cv::Mat with toOpenCV() or use
	* toOpenCVBGR(). (Most of OpenCV's features works with RGB too but
	* not imshow, imwrite, imread.)
	* \ingroup sibr_graphics
	*/
	template<typename T_Type, unsigned int T_NumComp>
	class Image : public IImage {
	public:
		typedef T_Type						Type;
		typedef ImagePtr<T_Type, T_NumComp> Ptr;

		typedef Eigen::Matrix<T_Type, T_NumComp, 1, Eigen::DontAlign> Pixel;
		enum { e_NumComp = T_NumComp };

	public:

		/// Default constructor.
		Image(void);

		/** Constructor.
		\param width image width
		\param height image height
		\warning The image content will be undefined.
		*/
		Image(uint width, uint height);

		/** Constructor.
		\param width image width
		\param height image height
		\param init default value to use for all components of all pixels
		*/
		Image(uint width, uint height, const T_Type& init);

		/** Constructor.
		\param width image width
		\param height image height
		\param init default value to use for all pixels
		*/
		Image(uint width, uint height, const Pixel& init);

		/** Move constructor.
		\param other image to move, don't use after move
		*/
		Image(Image&& other);

		/** Move operator.
		\param other image to move, don't use after move
		*/
		Image& operator=(Image&& other) noexcept;

		Image& fill(Pixel const& value);

		/**
		\copydoc IImage::load
		*/
		bool		load(const std::string& filename, bool verbose = true, bool warning_if_not_found = true);

		/**
		\copydoc IImage::loadByteStream
		*/
		bool		loadByteStream(const std::string& filename, bool verbose = true);
		
		/**
		\copydoc IImage::save
		*/
		void		save(const std::string& filename, bool verbose = true) const;

		/**
		\copydoc IImage::saveByteStream
		*/
		void		saveByteStream(const std::string& filename, bool verbose = true) const;

		/**
		\copydoc IImage::saveHDR
		*/
		void		saveHDR(const std::string& filename, bool verbose = true) const;

		/** Pixel accessor.
		\param x x coordinate
		\param y y coordinate
		\return a reference to the pixel value.
		*/
		const Pixel&	operator()(uint x, uint y) const;

		/** Pixel accessor.
		\param x x coordinate
		\param y y coordinate
		\return a reference to the pixel value.
		*/
		Pixel&			operator()(uint x, uint y);

		/** Pixel accessor.
		\param xy pixel coordinates
		\return a reference to the pixel value.
		*/
		const Pixel&	operator()(const sibr::Vector2i & xy) const;

		/** Pixel accessor.
		\param xy pixel coordinates
		\return a reference to the pixel value.
		*/
		Pixel&			operator()(const sibr::Vector2i & xy);

		/** Pixel accessor.
		\param xy pixel coordinates
		\return a reference to the pixel value.
		*/
		const Pixel&	operator()(const ::sibr::Vector2f & xy) const;

		/** Pixel accessor.
		\param xy pixel coordinates
		\return a reference to the pixel value.
		*/
		Pixel&			operator()(const ::sibr::Vector2f & xy);

		/** \copydoc IImage::pixelStr */
		virtual std::string		pixelStr(const ::sibr::Vector2i & xy)  const;

		/** \return a pointer to the raw image data. */
		const void*		data(void) const;

		/** \return a pointer to the raw image data. */
		void*			data(void);


		/** Convert a pixel value to a 4 components float vector (in 0,1).
		\param x x coordinate
		\param y y coordinate
		\return the normalized expanded value
		*/
		ColorRGBA	 color(uint x, uint y) const;

		/** Set a pixel value from 4 components float vector (in 0,1).
		\param x x coordinate
		\param y y coordinate
		\param c the new value
		*/
		void		 color(uint x, uint y, const ColorRGBA& c);

		/** Helper to convert a 4 components float vector (in 0,1) to the proper pixel format.
		\param rgba the value to convert
		\return the converted value
		*/
		static Pixel color(const ColorRGBA& rgba);

		/** Generate a resized version of the current image.
		\param width the target width
		\param height the target height
		\param cv_interpolation_method the up/down scaling method
		\return the resized image
		*/ 
		Image		resized(int width, int height, int cv_interpolation_method = cv::INTER_LINEAR) const;
		
		/** Generate a resized version of the current image so that the maximum 
		dimension (either width or height) is now equal to maxlen. Preserve the original ratio.
		Example: src is 2048x1024, resizedMax(1024) -> dst is 1024x512
		\param maxlen the target maximum dimension value
		\return the resized image
		*/ 
		Image		resizedMax(int maxlen) const;

		/** \return a deep copy of the image. */
		Image		clone(void) const;

		/** \return a pointer to a deep copy of the image. */
		ImagePtr<T_Type, T_NumComp>	  clonePtr(void) const;

		/** \return the image width. */
		uint			w(void) const;

		/** \return the image height. */
		uint			h(void) const;

		/** \return the image size. */
		sibr::Vector2u size(void) const;

		/** Check if a pixel (x,y) is inside the image boundaries.
		\param x x coordinate
		\param y y coordinate
		\return true if 0<=x<w and 0<=y=<h */
		template <typename T>
		bool			isInRange(T x, T y) const { return (x >= 0 && y >= 0 && x < (T)w() && y < (T)h()); }

		/** Check if a pixel (x,y) is inside the image boundaries.
		\param x x coordinate
		\param y y coordinate
		\return true if 0<=x<w and 0<=y=<h 
		\todo Duplicate call used in inpainting, remove.
		*/
		template <typename T>
		bool			inRange(T x, T y) const { return isInRange(x, y); }

		/** Check if a pixel (x,y) is inside the image boundaries.
		\param xy the pixel coordinates
		\return true if 0<=x<w and 0<=y=<h */
		bool			isInRange(const sibr::Vector2i & xy)  const { return (xy.x() >= 0 && xy.y() >= 0 && xy.x() < (int)w() && xy.y() < (int)h()); }
		
		/** Check if a pixel (x,y) is inside the image boundaries.
		\param xy the pixel coordinates
		\return true if 0<=x<w and 0<=y=<h */
		bool			isInRange(const sibr::Vector2f & xy)  const { return (xy.x() >= 0 && xy.y() >= 0 && xy.x() < (float)w() && xy.y() < (float)h()); }

		/** \copydoc IImage::numComp */
		uint		numComp(void) const;

		/** \copydoc IImage::sizeOfComp */
		uint		sizeOfComp(void) const;

		/** \copydoc IImage::flipH */
		void		flipH(void);

		/** \copydoc IImage::flipV */
		void		flipV(void);

		/** \copydoc IImage::opencvType */
		int				opencvType(void) const { return CV_MAKETYPE(opencv::imageType<T_Type>(), T_NumComp); }

		/** \copydoc IImage::toOpenCV */
		const cv::Mat&	toOpenCV(void) const { return _pixels; }

		/** \copydoc IImage::toOpenCVnonConst */
		cv::Mat&		toOpenCVnonConst(void) { return _pixels; }
		
		/** \copydoc IImage::toOpenCVBGR */
		cv::Mat			toOpenCVBGR(void) const;
		
		/** \copydoc IImage::fromOpenCV */
		void			fromOpenCV(const cv::Mat& img);
		
		/** \copydoc IImage::fromOpenCVBGR */
		void			fromOpenCVBGR(const cv::Mat& img);

		/** Find the component-wise minimum and maximum values contained in the image.
		\param minImage will contain the minimum value
		\param maxImage will contain the maximum value
		*/
		void findMinMax(Pixel& minImage, Pixel& maxImage);

		/** Rescale an image content in a defined range.
		\param minValue the lower value of the range
		\param maxValue the upper value of the range
		*/
		void remap(const Pixel& minValue, const Pixel& maxValue);

		/** Cast into another image type.
		\return the converted image
		*/
		template<class T_Image> T_Image cast() const {
			T_Image b;
			b.fromOpenCV(toOpenCV());
			return b;
		}

		/** Fetch bilinear interpolated value from floating point pixel coordinates.
			\param pixel query position in [0,w[x[0,h[
			\return the interpolated value
		*/
		Pixel bilinear(const sibr::Vector2f & pixel) const;

		/** Fetch bicubic interpolated value from floating point pixel coordinates.
			\param pixelPosition query position in [0,w[x[0,h[
			\return the interpolated value
		*/
		Pixel bicubic(const sibr::Vector2f & pixelPosition) const;

		/** Disallow copy constructor.
		\param other image to copy
		*/
		Image( const Image& other) = delete;

		/** Disallow copy operator.
		\param other image to copy
		*/
		Image& 		operator =(const Image& other) = delete;

	protected:

		/** Helper for bicubic interpolation.
		\param t blend factor
		\param colors colors at the four corners
		\return interpolated value
		*/
		static Eigen::Matrix<float, T_NumComp, 1, Eigen::DontAlign> monoCubic(float t, const Eigen::Matrix<float, T_NumComp, 4, Eigen::DontAlign>& colors);

		cv::Mat			_pixels; ///< Pixels stored in RGB format
	};

	/** Provides a wrapper around a pointer to an image. 
	\ingroup sibr_graphics
	*/
	template<typename T_Type, unsigned int T_NumComp>
	class ImagePtr {
	public:
		
		using ImageType = Image<T_Type, T_NumComp>; ///< Underlying image type.

		std::shared_ptr<Image<T_Type, T_NumComp>> imPtr; ///< Pointer type.
		
		/// Default constructor.
		ImagePtr() { imPtr = std::shared_ptr<Image<T_Type, T_NumComp>>(); };

		/** Constructor from a raw pointer.
		\param imgPtr the raw pointer to wrap
		*/
		ImagePtr(Image<T_Type, T_NumComp>* imgPtr) { imPtr = std::shared_ptr<Image<T_Type, T_NumComp>>(imgPtr); };
		
		/** Constructor from a shared pointer.
		\param imgPtr the shared pointer to wrap
		*/
		ImagePtr(const std::shared_ptr<Image<T_Type, T_NumComp>>& imgPtr)  {imPtr = std::shared_ptr<Image<T_Type, T_NumComp>>(imgPtr); };

		/** Generate a pointer by cloning an image.
		\param img the image to clone
		\return the pointer
		*/
		static ImagePtr fromImg(const ImageType & img) { return ImagePtr(std::make_shared<Image<T_Type, T_NumComp>>(img.clone())); };

		/** Set a new pointee.
		\param ptr the new image pointer
		*/
		void reset(ImageType * ptr) { imPtr.reset(ptr); };

		/** \return the image */
		Image<T_Type, T_NumComp>*	get() { return imPtr.get(); };

		/** Pixel accessor.
		\param x x coordinate
		\param y y coordinate
		\return a reference to the pixel value.
		*/
		const typename Image<T_Type, T_NumComp>::Pixel&			operator()(uint x, uint y) const;

		/** Pixel accessor.
		\param x x coordinate
		\param y y coordinate
		\return a reference to the pixel value.
		*/
		typename Image<T_Type, T_NumComp>::Pixel&				operator()(uint x, uint y);

		/** Pixel accessor.
		\param xy pixel coordinates
		\return a reference to the pixel value.
		*/
		const typename Image<T_Type, T_NumComp>::Pixel&			operator()(const sibr::Vector2i & xy) const;

		/** Pixel accessor.
		\param xy pixel coordinates
		\return a reference to the pixel value.
		*/
		typename Image<T_Type, T_NumComp>::Pixel&				operator()(const sibr::Vector2i & xy);

		/** \return the dereferenced image */
		Image<T_Type, T_NumComp>&								operator * () { return imPtr.operator*(); };

		/** \return the dereferenced image */
		const Image<T_Type, T_NumComp>&							operator * () const { return imPtr.operator*(); };

		/** \return raw pointer to the image */
		Image<T_Type, T_NumComp>*								operator -> () { return imPtr.operator->(); };

		/** \return raw pointer to the image */
		const Image<T_Type, T_NumComp>*							operator -> () const { return imPtr.operator->(); };
		
		/** Assign a shared ptr.
		\param imgShPtr the shared pointer
		\return a reference to the updated pointer
		*/ 
		std::shared_ptr<Image<T_Type, T_NumComp>> & 			operator = (std::shared_ptr<Image<T_Type, T_NumComp>> & imgShPtr) { imPtr = imgShPtr; return &imPtr; };
		
		/** \return true if the image pointer is initialized. */
		operator bool() { return imPtr.get() != nullptr; };

		/** \return true if the image pointer is initialized. */
		operator bool() const { return imPtr.get() != nullptr; };

	};

	/**
	* \addtogroup sibr_graphics
	* @{
	*/

	/// Standard image types
	typedef Image<unsigned char, 3> ImageRGB;
	typedef Image<unsigned char, 4> ImageRGBA;
	typedef Image<unsigned char, 1> ImageL8;
	typedef Image<unsigned char, 2> ImageUV8;
	typedef Image<unsigned short int, 3> ImageRGB16;
	typedef Image<unsigned short int, 1> ImageL16;
	typedef Image<float, 3>         ImageRGB32F;
	typedef Image<float, 3>         ImageFloat3;
	typedef Image<float, 4>         ImageRGBA32F;
	typedef Image<float, 4>         ImageFloat4;
	typedef Image<float, 1>         ImageL32F;
	typedef Image<float, 1>         ImageFloat1;
	typedef Image<float, 2>         ImageFloat2;
	typedef Image<float, 2>         ImageUV32F;
	typedef Image<bool, 1>          ImageBool1;
	typedef Image<double, 1>        ImageDouble1;
	typedef Image<double, 2>        ImageDouble2;
	typedef Image<double, 3>        ImageDouble3;
	typedef Image<double, 4>        ImageDouble4;
	typedef Image<int, 1>        ImageInt1;
	typedef Image<int, 2>        ImageInt2;
	typedef Image<int, 3>        ImageInt3;
	typedef Image<int, 4>        ImageInt4;


	/** Convert an integer ID map to a colored image using a different random color for each ID. Note that 255 is black.
	\param imClass the ID map
	\return a color coded map
	*/
	SIBR_GRAPHICS_EXPORT Image<unsigned char, 3> coloredClass(const Image<unsigned char, 1>::Ptr imClass);

	/** Convert an integer ID map to a colored image using a different random color for each ID. Note that 255 is black.
	\param imClass the ID map
	\return a color coded map
	*/
	SIBR_GRAPHICS_EXPORT Image<unsigned char, 3> coloredClass(const Image<int, 1>::Ptr imClass);

	/** Display a 32F image in a debug window, using the Parula colormap after normalizing the values.
	\param im the float image to display
	\param logScale display log(img)
	\param min optional lower bound for the normalization
	\param max optional upper bound for the normalization
	*/
	SIBR_GRAPHICS_EXPORT void showFloat(const Image<float, 1> & im, bool logScale = false, double min = -DBL_MAX, double max = DBL_MAX);

	/** Convert a L32F into a RGBA image while preserving bit-level representation.
	Useful to save float maps as PNG, and benefit from PNG compression on disk.
	\param imgF the image to convert
	\return the packed RGBA image 
	*/
	SIBR_GRAPHICS_EXPORT sibr::ImageRGBA convertL32FtoRGBA(const sibr::ImageL32F & imgF);

	/** Convert a RGBA into a L32F image while preserving bit-level representation.
	Useful to decode float maps stored as as PNG.
	\param imgRGBA the image to convert
	\return the unpacked float image 
	*/
	SIBR_GRAPHICS_EXPORT sibr::ImageL32F convertRGBAtoL32F(const sibr::ImageRGBA  & imgRGBA);

	/** Convert a RGB32F into a RGBA image (3 times wider) while preserving bit-level representation.
	Useful to save float maps as PNG, and benefit from PNG compression on disk.
	\param imgF the image to convert
	\return the packed RGBA image 
	*/
	SIBR_GRAPHICS_EXPORT sibr::ImageRGBA convertRGB32FtoRGBA(const sibr::ImageRGB32F & imgF);

	/** Convert a RGBA into a RGB32F image while preserving bit-level representation.
	Useful to decode float maps stored as as PNG.
	\param imgRGBA the image to convert
	\return the unpacked float image 
	*/
	SIBR_GRAPHICS_EXPORT sibr::ImageRGB32F convertRGBAtoRGB32F(const sibr::ImageRGBA & imgRGBA);

	/** Convert a RGB32 normal map into a UV16 map storing theta and phi as half floats, packed into a RGBA8.
	\param imgF the XYZ normal map
	\return the packed theta,phi normal map
	*/
	SIBR_GRAPHICS_EXPORT sibr::ImageRGBA convertNormalMapToSphericalHalf(const sibr::ImageRGB32F & imgF);

	/** Convert a RGBA map, packing theta and phi as half floats, into a RGB32 normal map.
	\param imgF packed theta,phi normal map
	\return the XYZ normal map
	*/
	SIBR_GRAPHICS_EXPORT sibr::ImageRGB32F convertSphericalHalfToNormalMap(const sibr::ImageRGBA & imgF);

	/** Create a three channels cv::Mat by repeating a single channel cv::Mat.
	\param c the input cv::Mat
	\return a three channels cv::Mat
	*/
	SIBR_GRAPHICS_EXPORT cv::Mat duplicate3(cv::Mat c);

	/** Display an image into a popup OpenCV window.
	\param img the image to display
	\param windowTitle the window title
	\param closeWindow close the window after key presses
	*/
	template <typename T_Type, unsigned T_NumComp>
	static void		show(const Image<T_Type, T_NumComp> & img, const std::string& windowTitle = "sibr::show()", bool closeWindow = true) {
		cv::namedWindow(windowTitle, cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
		// Note: CV_GUI_EXPANDED does only work with Qt
		
		cv::imshow(windowTitle, img.toOpenCVBGR());
		cv::waitKey(0);
		if (closeWindow) {
			cv::destroyWindow(windowTitle);
		}
	}

	/*** @} */

	// ----- DEFINITIONS -------------

	template<typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>::Image(void) :
		_pixels(0, 0, opencvType()) { }

	template<typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>::Image(uint width, uint height) :
		_pixels(height, width, opencvType()) { }

	template<typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>::Image(uint width, uint height, const T_Type& init) :
		_pixels(height, width, opencvType(), init) { }

	template<typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>::Image(uint width, uint height, const Pixel& init)
	{
		cv::Scalar scal(0);
		for (int i = 0; i < T_NumComp; i++)
			scal(i) = init(i);

		_pixels = cv::Mat(height, width, opencvType(), scal);

	}

	template<typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>::Image(Image<T_Type, T_NumComp>&& other) {
		operator =(std::move(other));
	}

	template<typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>& Image<T_Type, T_NumComp>::operator=(Image<T_Type, T_NumComp>&& other) noexcept {
		_pixels = std::move(other._pixels);
		return *this;
	}

	template <typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>& Image<T_Type, T_NumComp>::fill(Pixel const& value) {
		std::fill(_pixels.begin<Pixel>(), _pixels.end<Pixel>(), value);
		return *this;
	}

	template<typename T_Type, unsigned int T_NumComp>
	const void*			Image<T_Type, T_NumComp>::data(void) const {
		SIBR_ASSERT(_pixels.isContinuous() == true); // if not true, you don't want to use this function
		return _pixels.ptr();
	}

	template<typename T_Type, unsigned int T_NumComp>
	void*			Image<T_Type, T_NumComp>::data(void) {
		SIBR_ASSERT(_pixels.isContinuous() == true); // if not true, you don't want to use this function
		return _pixels.ptr();
	}

	template<typename T_Type, unsigned int T_NumComp>
	cv::Mat			Image<T_Type, T_NumComp>::toOpenCVBGR(void) const {
		cv::Mat out = toOpenCV().clone();
		opencv::convertRGB2BGR(out);
		return out;
	}

	template<typename T_Type, unsigned int T_NumComp>
	void			Image<T_Type, T_NumComp>::fromOpenCVBGR(const cv::Mat& imgSrc) {
		cv::Mat img = imgSrc.clone();
		opencv::convertBGR2RGB(img);
		fromOpenCV(img);
	}

	template<typename T_Type, unsigned int T_NumComp>
	void			Image<T_Type, T_NumComp>::fromOpenCV(const cv::Mat& imgSrc) {
		cv::Mat img = imgSrc.clone();

		if (img.depth() != opencv::imageType<T_Type>())
		{
			img.convertTo(img, opencv::imageType<T_Type>(),
				opencv::imageTypeRange<T_Type>() / opencv::imageTypeCVRange(img.depth()));
		}

		cv::Vec<T_Type, T_NumComp> p;
		if (img.channels() != T_NumComp)
		{
			_pixels = cv::Mat(img.rows, img.cols, opencvType());
			for (int y = 0; y < img.rows; ++y)
			{
				for (int x = 0; x < img.cols; ++x)
				{
					const T_Type* ptr = img.ptr<T_Type>(y, x);
					assert(ptr != nullptr);
					uint i;
					for (i = 0; i < (uint)img.channels() && i < T_NumComp; ++i)
						p[i] = ptr[i];
					for (; i < T_NumComp && i < 3; ++i)
						p[i] = p[0];
					for (; i < T_NumComp && i < 4; ++i)
						p[i] = static_cast<T_Type>(opencv::imageTypeRange<T_Type>());

					_pixels.at<cv::Vec<T_Type, T_NumComp>>(y, x) = p;
				}
			}
		}
		else
			_pixels = img;
	}

	template<typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>		Image<T_Type, T_NumComp>::clone(void) const {
		Image<T_Type, T_NumComp> img;
		img._pixels = _pixels.clone();
		return img;
	}

	template<typename T_Type, unsigned int T_NumComp>
	ImagePtr<T_Type, T_NumComp>		Image<T_Type, T_NumComp>::clonePtr(void) const {
		ImagePtr<T_Type, T_NumComp> img(new Image<T_Type, T_NumComp>());
		img->_pixels = _pixels.clone();
		return img;
	}

	template<typename T_Type, unsigned int T_NumComp>
	bool		Image<T_Type, T_NumComp>::load(const std::string& filename, bool verbose, bool warning_if_not_found) {
		if (verbose)
			SIBR_LOG << "Loading image file '" << filename << "'." << std::endl;
		else
			std::cerr << ".";
		cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED | cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
		if (img.data == nullptr)
		{
			operator =(Image<T_Type, T_NumComp>()); // reset mat

			if (warning_if_not_found) {
				SIBR_WRG << "Image file not found '" << filename << "'." << std::endl;
			}

			return false;
		}
		opencv::convertBGR2RGB(img);
		fromOpenCV(img);
		return true;
	}

	template<typename T_Type, unsigned int T_NumComp>
	bool		Image<T_Type, T_NumComp>::loadByteStream(const std::string& filename, bool verbose) {
		if (verbose)
			SIBR_LOG << "Loading image file '" << filename << "'." << std::endl;
		else
			std::cerr << ".";


		cv::Vec<T_Type, T_NumComp> p;

		sibr::ByteStream bs;
		if (!bs.load(filename))
			SIBR_WRG << "Image file not found '" << filename << "'." << std::endl;

		int wIm;
		int hIm;
		bs >> wIm >> hIm;

		_pixels = cv::Mat(hIm, wIm, opencvType());
		for (int y = 0; y < hIm; ++y)
		{
			for (int x = 0; x < wIm; ++x)
			{
				uint i;
				for (i = 0; i < T_NumComp; ++i)
					bs >> p[i];

				_pixels.at<cv::Vec<T_Type, T_NumComp>>(y, x) = p;
			}
		}

		return true;
	}

	template<typename T_Type, unsigned int T_NumComp>
	void		Image<T_Type, T_NumComp>::save(const std::string& filename, bool verbose) const {
		{ // Create the output dir if doesn't exists
			boost::filesystem::path outdir = boost::filesystem::path(filename).parent_path();
			if (outdir.empty() == false)
				boost::filesystem::create_directories(outdir);
		}

		// Important Note:
		// If you have a problem when saving an image (e.g. black image) then
		// check the targeted image file format manages correctly the T_Type and
		// T_NumpComp you provide.
		// OpenCV doesn't seem to check always for such incompatibility (and just
		// save empty pixels)

		if (verbose)
			SIBR_LOG << "Saving image file '" << filename << "'." << std::endl;

		cv::Mat img;
		if (T_NumComp == 1) {
			cv::cvtColor(toOpenCVBGR(), img, cv::COLOR_GRAY2BGR);
		} /// \todo TODO: support for 2 channels images.
		else {
			// For 3 and 4 channels, leave the image untouched.
			img = toOpenCVBGR();
		}

		cv::Mat finalImage;
		if (T_NumComp == 4) {
			cv::Mat4b imageF_8UC4;
			double scale = 255.0 / (double)opencv::imageTypeRange<T_Type>();
			img.convertTo(imageF_8UC4, CV_8UC4, scale);
			finalImage = imageF_8UC4;
		}
		else {
			cv::Mat3b imageF_8UC3;
			double scale = 255.0 / (double)opencv::imageTypeRange<T_Type>();
			img.convertTo(imageF_8UC3, CV_8UC3, scale);
			finalImage = imageF_8UC3;
		}

		if (img.cols > 0 && img.rows > 0)
		{
			if (cv::imwrite(filename, finalImage) == false)
				SIBR_ERR << "unknown error while saving image '" << filename << "'"
				<< " (do the targeted file format manages correctly the bpc ?)" << std::endl;
		}
		else
			SIBR_WRG << "failed to save (image is empty)" << std::endl;
	}

	template<typename T_Type, unsigned int T_NumComp>
	void		Image<T_Type, T_NumComp>::saveHDR(const std::string& filename, bool verbose) const {
		{ // Create the output dir if doesn't exists
			boost::filesystem::path outdir = boost::filesystem::path(filename).parent_path();
			if (outdir.empty() == false)
				boost::filesystem::create_directories(outdir);
		}

		if (verbose)
			SIBR_LOG << "Saving image file '" << filename << "'." << std::endl;

		cv::Mat img;
		if (T_NumComp == 1) {
			cv::cvtColor(toOpenCVBGR(), img, cv::COLOR_GRAY2BGR);
		} /// \todo TODO: support for 2 channels images.
		else {
			// For 3 and 4 channels, leave the image untouched.
			img = toOpenCVBGR();
		}

		cv::Mat finalImage;
		if (T_NumComp == 4) {
			cv::Mat4f imageF_32FC4;
			double scale = 1.0 / (double)opencv::imageTypeRange<T_Type>();
			img.convertTo(imageF_32FC4, CV_32FC4, scale);
			finalImage = imageF_32FC4;
		}
		else {
			cv::Mat3f imageF_32FC3;
			double scale = 1.0 / (double)opencv::imageTypeRange<T_Type>();
			img.convertTo(imageF_32FC3, CV_32FC3, scale);
			finalImage = imageF_32FC3;
		}

		if (img.cols > 0 && img.rows > 0)
		{
			if (cv::imwrite(filename, finalImage) == false)
				SIBR_ERR << "unknown error while saving image '" << filename << "'"
				<< " (do the targeted file format manages correctly the bpc ?)" << std::endl;
		}
		else
			SIBR_WRG << "failed to save (image is empty)" << std::endl;
	}

	template <typename T_Type, unsigned int T_NumComp>
	void		Image<T_Type, T_NumComp>::saveByteStream(const std::string& filename, bool verbose) const {
		{ // Create the output dir if doesn't exists
			boost::filesystem::path outdir = boost::filesystem::path(filename).parent_path();
			if (outdir.empty() == false)
				boost::filesystem::create_directories(outdir);
		}
		if (verbose)
			SIBR_LOG << "Saving image file '" << filename << "'." << std::endl;

		sibr::ByteStream bs;

		int wIm = w();
		int hIm = h();

		if (wIm > 0 && hIm > 0) {
			bs << wIm << hIm;
			for (int j = 0; j < hIm; j++) {
				for (int i = 0; i < wIm; i++) {
					for (int k = 0; k < T_NumComp; k++) {
						bs << _pixels.at<cv::Vec<T_Type, T_NumComp>>(j, i)[k];
					}
				}
			}
			bs.saveToFile(filename);
		}
		else
			SIBR_WRG << "failed to save (image is empty)" << std::endl;
	}

	template<typename T_Type, unsigned int T_NumComp>
	inline const typename Image<T_Type, T_NumComp>::Pixel&		Image<T_Type, T_NumComp>::operator()(uint x, uint y) const {
#ifndef NDEBUG
		if (!(x < w() && y < h())) {
			std::cout << " access (" << x << " , " << y << ") while size is " << w() << " x " << h() << std::endl;
}
#endif
		SIBR_ASSERT(x < w() && y < h());
		return _pixels.at<typename Image<T_Type, T_NumComp>::Pixel>(y, x);
	}

	template<typename T_Type, unsigned int T_NumComp>
	inline const typename Image<T_Type, T_NumComp>::Pixel & ImagePtr<T_Type, T_NumComp>::operator()(uint x, uint y) const
	{
		return (*imPtr)(x, y);
	}

	template<typename T_Type, unsigned int T_NumComp>
	inline typename Image<T_Type, T_NumComp>::Pixel&		Image<T_Type, T_NumComp>::operator()(uint x, uint y) {
#ifndef NDEBUG
		if (!(x < w() && y < h())) {
			std::cout << " access (" << x << " , " << y << ") while size is " << w() << " x " << h() << std::endl;
		}
#endif
		SIBR_ASSERT(x < w() && y < h());
		return _pixels.at<typename Image<T_Type, T_NumComp>::Pixel>(y, x);
	}

	template<typename T_Type, unsigned int T_NumComp>
	inline typename Image<T_Type, T_NumComp>::Pixel & ImagePtr<T_Type, T_NumComp>::operator()(uint x, uint y)
	{
		return (*imPtr)(x, y);
	}


	template<typename T_Type, unsigned int T_NumComp>
	inline const typename Image<T_Type, T_NumComp>::Pixel& Image<T_Type, T_NumComp>::operator()(const sibr::Vector2i & xy) const {
		return operator()(xy[0], xy[1]);
	}
	template<typename T_Type, unsigned int T_NumComp>
	inline const typename Image<T_Type, T_NumComp>::Pixel & ImagePtr<T_Type, T_NumComp>::operator()(const sibr::Vector2i & xy) const
	{
		return (*imPtr)(xy[0], xy[1]);
	}

	template<typename T_Type, unsigned int T_NumComp>
	inline typename Image<T_Type, T_NumComp>::Pixel& Image<T_Type, T_NumComp>::operator()(const sibr::Vector2i & xy) {
		return operator()(xy[0], xy[1]);
	}
	template<typename T_Type, unsigned int T_NumComp>
	inline typename Image<T_Type, T_NumComp>::Pixel & ImagePtr<T_Type, T_NumComp>::operator()(const sibr::Vector2i & xy)
	{
		return (*imPtr)(xy[0], xy[1]);
	}

	template<typename T_Type, unsigned int T_NumComp>
	inline typename Image<T_Type, T_NumComp>::Pixel& Image<T_Type, T_NumComp>::operator()(const sibr::Vector2f & xy) {
		return operator()((int)xy[0], (int)xy[1]);
	}
	template<typename T_Type, unsigned int T_NumComp>
	inline const typename Image<T_Type, T_NumComp>::Pixel& Image<T_Type, T_NumComp>::operator() (const sibr::Vector2f & xy) const {
		return operator()((int)xy[0], (int)xy[1]);
	}

	template<typename T_Type, unsigned int T_NumComp>
	ColorRGBA	Image<T_Type, T_NumComp>::color(uint x, uint y) const {
		SIBR_ASSERT(x < w() && y < h());
		float scale = 1.f / opencv::imageTypeRange<T_Type>();
		cv::Vec<T_Type, T_NumComp> v = _pixels.at<cv::Vec<T_Type, T_NumComp>>(y, x);

		return ColorRGBA(v.val[0] * scale, v.val[1] * scale, v.val[2] * scale,
			(T_NumComp > 3) ? v.val[3] * scale : 1.f);
	}
	template<typename T_Type, unsigned int T_NumComp>
	void		Image<T_Type, T_NumComp>::color(uint x, uint y, const ColorRGBA& rgba) {
		SIBR_ASSERT(x < w() && y < h());
		float scale = opencv::imageTypeRange<T_Type>();
		cv::Vec<T_Type, T_NumComp> v;//(p.data(), T_NumComp);
		for (uint i = 0; i < T_NumComp; ++i) v[i] = T_Type(rgba[i] * scale);
		_pixels.at<cv::Vec<T_Type, T_NumComp>>(y, x) = v;
	}

	template<typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>	Image<T_Type, T_NumComp>::resized(int width, int height, int cv_interpolation_method) const
	{
		if (width == w() && height == h())
			return clone();
		Image dst;
		cv::resize(toOpenCV(), dst._pixels, cv::Size(width, height), 0, 0, cv_interpolation_method);
		return dst;
	}

	template<typename T_Type, unsigned int T_NumComp>
	Image<T_Type, T_NumComp>		Image<T_Type, T_NumComp>::resizedMax(int maxlen) const
	{
		float newWidth = (w() >= h()) ? maxlen : maxlen * ((float)w() / (float)h());
		float newHeight = (h() >= w()) ? maxlen : maxlen * ((float)h() / (float)w());

		return resized((int)newWidth, (int)newHeight);
	}

	template<typename T_Type, unsigned int T_NumComp>
	typename Image<T_Type, T_NumComp>::Pixel Image<T_Type, T_NumComp>::color(const ColorRGBA& rgba) {
		float scale = opencv::imageTypeRange<T_Type>();
		Pixel v;//(p.data(), T_NumComp);
		for (uint i = 0; i < T_NumComp; ++i) v[i] = T_Type(rgba[i] * scale);
		return v;
	}

	template<typename T_Type, unsigned int T_NumComp>
	std::string Image<T_Type, T_NumComp>::pixelStr(const ::sibr::Vector2i & xy)  const {
		if (isInRange(xy)) {
			std::stringstream ss;
//			ss << "( " << operator()(xy).cast<std::conditional<std::is_same_v<T_Type, uchar>, int, T_Type>::type>().transpose() << " )";
std::cerr << "PIXEL STR PB" << std::endl;
exit(1);
			return  ss.str();
		}
		return "";
	}

	template<typename T_Type, unsigned int T_NumComp>
	uint		Image<T_Type, T_NumComp>::w(void) const {
		return _pixels.cols;
	}

	template<typename T_Type, unsigned int T_NumComp>
	uint		Image<T_Type, T_NumComp>::h(void) const {
		return _pixels.rows;
	}

	template<typename T_Type, unsigned int T_NumComp>
	sibr::Vector2u	Image<T_Type, T_NumComp>::size(void) const {
		return sibr::Vector2u(w(), h());
	}

	template<typename T_Type, unsigned int T_NumComp>
	uint		Image<T_Type, T_NumComp>::numComp(void) const {
		return T_NumComp;
	}

	template<typename T_Type, unsigned int T_NumComp>
	uint		Image<T_Type, T_NumComp>::sizeOfComp(void) const {
		return sizeof(T_Type)*T_NumComp;
	}

	template<typename T_Type, unsigned int T_NumComp>
	void		Image<T_Type, T_NumComp>::flipH(void) {
		cv::flip(_pixels, _pixels, 0 /*!=0 means horizontal*/);
	}
	template<typename T_Type, unsigned int T_NumComp>
	void		Image<T_Type, T_NumComp>::flipV(void) {
		cv::flip(_pixels, _pixels, 1 /*!=1 means vertical*/);
	}

	template<typename T_Type, unsigned int T_NumComp>
	void Image<T_Type, T_NumComp>::findMinMax(Pixel& minImage, Pixel& maxImage) {
		for (uint c = 0; c < T_NumComp; ++c) {
			minImage[c] = T_Type(opencv::imageTypeRange<Type>());
			maxImage[c] = T_Type(-opencv::imageTypeRange<Type>());
		}

		Pixel p;
		for (uint y = 0; y < h(); ++y) {
			for (uint x = 0; x < w(); ++x) {
				Pixel v = operator()(x, y);
				for (uint c = 0; c < T_NumComp; ++c) {
					minImage[c] = std::min(v[c], minImage[c]);
					maxImage[c] = std::max(v[c], maxImage[c]);
				}
			}
		}
	}

	template<typename T_Type, unsigned int T_NumComp>
	void		Image<T_Type, T_NumComp>::remap(const Pixel& minVal, const Pixel& maxVal) {
		Pixel minImage;
		Pixel maxImage;
		findMinMax(minImage, maxImage);

		Pixel p;
		for (uint y = 0; y < h(); ++y) {
			for (uint x = 0; x < w(); ++x) {
				Pixel v = operator()(x, y);
				for (uint i = 0; i < T_NumComp; ++i)
					p[i] = minVal[i] + ((maxVal[i] - minVal[i])*(v[i] - minImage[i])) / (maxImage[i] - minImage[i]);
				operator()(x, y) = p;
			}
		}
	}

	template<typename T_Type, unsigned int T_NumComp>
	Eigen::Matrix<T_Type, T_NumComp, 1, Eigen::DontAlign> Image<T_Type, T_NumComp>::bilinear(const sibr::Vector2f & queryPosition) const
	{
		if (w() < 2 || h() < 2) {
			return Eigen::Matrix<T_Type, T_NumComp, 1, Eigen::DontAlign>();
		}

		const sibr::Vector2i cornerPixel = sibr::Vector2f((queryPosition - 0.5f*sibr::Vector2f(1, 1)).unaryExpr([](float f) { return std::floor(f); })).template cast<int>();

		const sibr::Vector2f ts = queryPosition - (cornerPixel.cast<float>() + 0.5f*sibr::Vector2f(1, 1));

		const sibr::Vector2i topLeft(0, 0), bottomRight(w() - 1, h() - 1);

		const sibr::Vector2i mm = sibr::clamp<int, 2>(cornerPixel + sibr::Vector2i(0, 0), topLeft, bottomRight);
		const sibr::Vector2i pm = sibr::clamp<int, 2>(cornerPixel + sibr::Vector2i(1, 0), topLeft, bottomRight);
		const sibr::Vector2i mp = sibr::clamp<int, 2>(cornerPixel + sibr::Vector2i(0, 1), topLeft, bottomRight);
		const sibr::Vector2i pp = sibr::clamp<int, 2>(cornerPixel + sibr::Vector2i(1, 1), topLeft, bottomRight);
		return (
			operator()(mm).template cast<float>() * (1.0f - ts[0]) * (1.0f - ts[1]) +
			operator()(pm).template cast<float>() * ts[0] * (1.0f - ts[1]) +
			operator()(mp).template cast<float>() * (1.0f - ts[0]) * ts[1] +
			operator()(pp).template cast<float>() * ts[0] * ts[1]
			).template cast<T_Type>();
	}

	template<typename T_Type, unsigned int T_NumComp>
	Eigen::Matrix<float, T_NumComp, 1, Eigen::DontAlign> Image<T_Type, T_NumComp>::monoCubic(float t, const Eigen::Matrix<float, T_NumComp, 4, Eigen::DontAlign> & colors)
	{
		static const Eigen::Matrix<float, 4, 4> M = 0.5f* (Eigen::Matrix<float, 4, 4>() <<
			0, 2, 0, 0,
			-1, 0, 1, 0,
			2, -5, 4, -1,
			-1, 3, -3, 1
			).finished().transpose();

		return colors * (M*Eigen::Matrix<float, 4, 1>(1, t, t*t, t*t*t));
	}

	template<typename T_Type, unsigned int T_NumComp>
	Eigen::Matrix<T_Type, T_NumComp, 1, Eigen::DontAlign> Image<T_Type, T_NumComp>::bicubic(const sibr::Vector2f & queryPosition) const
	{
		static const std::vector<std::vector<sibr::Vector2i> > offsets = {
			{ { -1,-1 },{ 0,-1 } ,{ 1,-1 },{ 2,-1 } },
			{ { -1,0 },{ 0,0 } ,{ 1,0 },{ 2,0 } },
			{ { -1,1 },{ 0,1 } ,{ 1,1 },{ 2,1 } },
			{ { -1,2 },{ 0,2 } ,{ 1,2 },{ 2,2 } }
		};

		typedef Eigen::Matrix<float, T_NumComp, 4, Eigen::DontAlign> ColorStack;

		if (w() < 4 || h() < 4) {
			return Vector<T_Type, T_NumComp>();
		}

		const sibr::Vector2i cornerPixel = (queryPosition - 0.5f*sibr::Vector2f(1, 1)).unaryExpr([](float f) { return std::floor(f); }).template cast<int>();
		const sibr::Vector2f ts = queryPosition - (cornerPixel.cast<float>() + 0.5f*sibr::Vector2f(1, 1));

		ColorStack colorsGrid[4];
		const sibr::Vector2i topLeft(0, 0), bottomRight(w() - 1, h() - 1);
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				const sibr::Vector2i pixelPosition = cornerPixel + offsets[i][j];
				colorsGrid[i].col(j) = operator()(sibr::clamp(pixelPosition, topLeft, bottomRight)).template cast<float>();
			}
		}

		ColorStack bs;
		for (int i = 0; i < 4; ++i) {
			bs.col(i) = monoCubic(ts[0], colorsGrid[i]);
		}

		Vector<float, T_NumComp> resultFloat = monoCubic(ts[1], bs);
		return (resultFloat.unaryExpr([](float f) { return sibr::clamp(f, 0.0f, sibr::opencv::imageTypeRange<T_Type>()); })).template cast<T_Type>();
	}

	template <typename sibr_T, typename openCV_T, int N>
	inline Vector<sibr_T, N> fromOpenCV(const cv::Vec<openCV_T, N> & vec) {
		Vector<sibr_T, N> out;
		for (int i = 0; i < N; ++i) {
			out[i] = static_cast<sibr_T>(vec[i]);
		}
		return out;
	}

	template <typename openCV_T, typename sibr_T, int N>
	inline cv::Vec<openCV_T, N> toOpenCV(const Vector<sibr_T, N> & vec) {
		cv::Vec<openCV_T, N> out;
		for (int i = 0; i < N; ++i) {
			out[i] = static_cast<openCV_T>(vec[i]);
		}
		return out;
	}

} // namespace sibr
