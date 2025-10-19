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

#include "core/graphics/Config.hpp"
#include "core/graphics/Camera.hpp"
#include "core/assets/Config.hpp"

namespace sibr
{
	/** Input camera parameters. Inherits all basic camera functionality from Camera
	*  and adds functions for depth samples from multi-view stereo.
	*
	* \sa Camera, NovelCamera
	* \ingroup sibr_assets
	*/
	class SIBR_ASSETS_EXPORT InputCamera : public Camera 
	{
	public:
		typedef std::shared_ptr<InputCamera> Ptr;

		/** Near/far plane representation. */
		struct Z {

			/** Constructor. */
			Z() {}

			/** Constructor.
			 * \warning Ordering of the values is swapped.
			 * \param f far plane
			 * \param n near plane
			 */
			Z(float f, float n) : far(f), near(n) {}

			float far = 0.0f; ///< Far plane.
			float near = 0.0f; ///< Near plane.
		};

		/** Default constructor. */
		InputCamera() :
			_focal(0.f), _k1(0.f), _k2(0.f), _w(0), _h(0), _id(0), _active(true)
		{ }

		/** Partial constructor
		* \param f focal length in mm
		* \param k1 first distortion parameter
		* \param k2 second distortion parameter
		* \param w  width of input image
		* \param h  height of input image
		* \param id ID of input image
		*/
		InputCamera(float f, float k1, float k2, int w, int h, int id);
		InputCamera(float fy, float fx, float k1, float k2, int w, int h, int id);

		/** Constructor, initialize the input camera.
		* \param id ID of input image
		* \param w  width of input image
		* \param h  height of input image
		* \param position camera position
		* \param rotation camera rotation
		* \param focal focal length in mm
		* \param k1 first distortion parameter
		* \param k2 second distortion parameter
		* \param active  input image active or not
		*/
		InputCamera(int id, int w, int h, sibr::Vector3f & position, sibr::Matrix3f & rotation, float focal, float k1, float k2, bool active);

		/** Constructor, initialize the input camera.
		* \param id ID of input image
		* \param w  width of input image
		* \param h  height of input image
		* \param m  camera parameters resad from Bundler output file
		* \param active  input image active or not
		* \param fovFromFocal: if true, compute fov from focal else use "standard sibr" convention
		* \sa Bundler: http://phototour.cs.washington.edu/bundler/
		* \deprecated Avoid using this legacy constructor.
		*/
		InputCamera(int id, int w, int h, sibr::Matrix4f m, bool active);

		/** Constructor from a basic Camera.
		 * \param c camera
		 * \param w image width
		 * \param h image height
		 */
		InputCamera(const Camera& c, int w, int h);

		/** Copy constructor. */
		InputCamera(const InputCamera&) = default;

		/** Move constructor. */
		InputCamera(InputCamera&&) = default;

		/** Copy operator. */
		InputCamera&	operator =(const InputCamera&) = default;

		/** Move operator. */
		InputCamera&	operator =(InputCamera&&) = default;

		/** Input image width
		* \return width of input image
		*/
		uint w(void) const;

		/** Input image height
		* \return height of input image
		*/
		uint h(void) const;

		/** Check if the input camera active or inactive,
		* camera is completely ignored if set to inactive.
		* \return true if active, false otherwise
		*/
		bool isActive(void) const;

		/** Set camera active status
		 *\param active if true, camera is in use
	     */
		void setActive(bool active) { _active = active ; }

		/** \return the image name */
		inline const std::string&	name(void) const { return _name; }

		/** Set camera name 
		 * \param s the new name
		 */
		inline void					name( const std::string& s ) { _name = s; }

		/** Update image dimensions. Calls \a update() after changing image width and height
		* \param w image width
		* \param h image height
		*/
		void size( uint w, uint h );

		/** \return the camera id */
		uint id() const { return _id; }

		/** Project a world space point into screen space.
		 *\param pt 3d world point
		 *\return screen space position and depth, in (0,w)x(0,h)x(0,1)
		 */
		Vector3f projectScreen( const Vector3f& pt ) const;

		/** \return the focal length */
		float focal() const;

		/** \return the focal length x */
		float focalx() const;

		/** set the focal length ; to be used with caution; focal is usually inferred from the fov*/
		void setFocal(float focal) { _focal = focal; }

		/** \return the k1 distorsion parameter */
		float k1() const;

		/** \return the k2 distorsion parameter */
		float k2() const;

		/** Back-project pixel coordinates and depth.
		* \param pixelPos pixel coordinates p[0],p[1] in [0,w-1]x[0,h-1] 
		* \param depth d in [-1,1]
		* \returns 3D world point
		*/
		Vector3f			unprojectImgSpaceInvertY( const sibr::Vector2i & pixelPos, const float & depth ) const;

		/** Project 3D point using perspective projection.
		* \param point3d 3D point
		* \returns pixel coordinates in [0,w-1]x[0,h-1] and depth d in [-1,1]
		*/
		Vector3f			projectImgSpaceInvertY( const Vector3f& point3d  ) const;

		/** Load from internal binary representation.
		 * \param filename file path
		 * \return success boolean
		 */
		bool				loadFromBinary( const std::string& filename );

		/** Save to disk using internal binary representation.
		 * \param filename file path
		 */
		void				saveToBinary( const std::string& filename ) const;

		/** Save a file in the IBR TopView format.
		* \param outfile the destination file
		*/
		void				writeToFile(std::ostream& outfile) const;

		/** Load a file in the IBR TopView format.
		* \param infile the input file
		*/
		void				readFromFile(std::istream& infile);

		/** Conver to Bundle string.
		 * \param negativeZ should the Z axis be flipped
		 * \recomputeFocal recompute the focal or just set
		 * \return a string that can be used to create a bundle file from this camera
		*/
		std::string toBundleString(bool negativeZ = false, bool recomputeFocal = true) const;


		/** \return A vector of four Vector2i corresponding to the pixels at the camera corners
		*/
		std::vector<sibr::Vector2i> getImageCorners() const;

		/** Return a new camera resized to the specified height
		*/
		sibr::InputCamera resizedH(int h) const;
		/** Return a new camera resized to the specified height
		*/
		sibr::InputCamera resizedW(int w) const;

		/** Return the lookat string of the camera
		*/
		std::string lookatString() const;
		/** save a vector of cameras as lookat
		*/
		static void saveAsLookat(const std::vector<InputCamera::Ptr> & cams, const std::string & fileName);
		/** save a vector of cameras sizes to a file to be read by mitsuba rendering script
		*/
		static void saveImageSizes(const std::vector<InputCamera::Ptr> & cams, const std::string & fileName);


		/** Save a vector of cameras as a bundle file.
		 *\param cams the cameras
		 * \param fileName output bundle file path
		 * \param negativeZ should the Z axis be flipped
		 * \param exportImages should empty images with the proper dimensions be saved in a visualize subdirectory
		 * \param oldFocal: recompute focal, else assign that of camera TODO: fix this
		*/
		static void saveAsBundle(const std::vector<InputCamera::Ptr> & cams, const std::string & fileName, bool negativeZ = false, bool exportImages = false, bool recomputeFocal=true);

		/** Save a vector of cameras as a lookat file.
		 *\param cams the cameras
		 * \param fileName output lookat file path
		*/
		static void saveAsLookat(const std::vector<sibr::Camera> & cams, const std::string & fileName);

		/** Load cameras from a bundler file.
		 *\param datasetPath path to the root of the dataset, should contain bundle.out, list_images.txt and optionally clipping_planes.txt 
		 * \param zNear default near-plane value to use if the clipping_planes.txt file doesn't exist
		 * \param zFar default far-plane value to use if the clipping_planes.txt file doesn't exist
		 * \param bundleName name of the bundle file
		 * \param listName name of the list images file
		 * \returns the loaded cameras
		 */
		static std::vector<InputCamera::Ptr> load( const std::string& datasetPath, float zNear = 0.01f, float zFar = 1000.0f, const std::string & bundleName = "bundle.out", const std::string & listName = "list_images.txt");

		/** Load cameras from a NVM file.
		*\param nvmPath path to the NVM file
		* \param zNear default near-plane value to use
		* \param zFar default far-plane value to use.
		* \param wh will contain the sizes of each camera image
		* \returns the loaded cameras
		*/
		static std::vector<InputCamera::Ptr> loadNVM(const std::string& nvmPath, float zNear = 0.01f, float zFar = 1000.0f, std::vector<sibr::Vector2u> wh = std::vector<sibr::Vector2u>());

		/** Load cameras from a .lookat file generated by our Blender plugin.
		* \param lookatPath path to the lookAt file
		* \param wh indicative size of each camera image
		* \param zNear default near-plane value to use
		* \param zFar default far-plane value to use.
		* \returns the loaded cameras
		*/
		static std::vector<InputCamera::Ptr> loadLookat(const std::string& lookatPath, const std::vector<sibr::Vector2u>& wh= std::vector<sibr::Vector2u>(),float zNear= -1, float zFar= -1);

		static std::vector<InputCamera::Ptr> InputCamera::loadTransform(const std::string& transformPath, int w, int h, std::string extension, const float zNear = 0.01f, const float zFar = 1000.0f, const int offset = 0, const int fovXfovYFlag = 0);

		/** Load cameras from a Colmap txt file.
		* \param colmapSparsePath path to the Colmap sparse directory, should contains cameras.txt and images.txt
		* \param zNear default near-plane value to use
		* \param zFar default far-plane value to use.
		* \param fovXfovYFlag should we use two dimensional fov.
		* \returns the loaded cameras
		* \note the camera frame is internally transformed to be consistent with fribr and RC.
		*/
		static std::vector<InputCamera::Ptr> loadColmap(const std::string& colmapSparsePath, const float zNear = 0.01f, const float zFar = 1000.0f, const int fovXfovYFlag = 0);

		static std::vector<InputCamera::Ptr> loadColmapBin(const std::string& colmapSparsePath, const float zNear = 0.01f, const float zFar = 1000.0f, const int fovXfovYFlag = 0);

		static std::vector<InputCamera::Ptr> loadJSON(const std::string& jsonPath, const float zNear = 0.01f, const float zFar = 1000.0f);

		/** Load cameras from a bundle file.
		* \param bundlerPath path to the bundle file.
		* \param zNear default near-plane value to use
		* \param zFar default far-plane value to use.
		* \param listImagePath path to the list_images.txt file. Will default to a file in the same directory as the bundle.out file.
		* \param path: if this is a path, then you can load more images that those defined in the list_images file of the datset; TODO: possibly should require a list_images.txt for the path
		* \returns the loaded cameras
		*/
		static std::vector<InputCamera::Ptr> loadBundle(const std::string& bundlerPath,  float zNear = 0.01f, float zFar = 1000.0f, const std::string & listImagePath = "", bool path = false);

		/** Load cameras from a bundle file.
		* \param bundlerPath path to the bundle file.
		* \param zNear default near-plane value to use
		* \param zFar default far-plane value to use.
		* \param listImagePath path to the list_images.txt file. Will default to a file in the same directory as the bundle.out file.
		* \returns the loaded cameras
		*/
		static std::vector<InputCamera::Ptr> loadBundleFRIBR(const std::string& bundlerPath, float zNear = 0.01f, float zFar = 1000.0f, const std::string & listImagePath = "");

		/** Load cameras from a Meshrrom SFM cameras.sfm txt file.
		* \param meshroomSFMPath path to the Meshroom StructureFromMotion/{dd63cea98bda0e3b53ec76f17b0753b3e4dde589}/ directory, should contains cameras.sfm 
		* \param zNear default near-plane value to use
		* \param zFar default far-plane value to use.
		* \returns the loaded cameras
		* \note the camera frame is internally transformed to be consistent with fribr and RC.
		*/
		static std::vector<InputCamera::Ptr> loadMeshroom(const std::string& meshroomSFMPath, const float zNear = 0.01f, const float zFar = 1000.0f);

		uint _id; ///< Input camera id

	protected:

		float _focal; ///< focal length
		float _focalx; ///< focal length x, if there is one (colmap typically; -1 by default use with caution)
		float _k1; ///< K1 bundler distorsion parameter
		float _k2; ///< K2 bundler dist parameter
		uint _w; ///< Image width
		uint _h; ///< Image height
		std::string _name; ///< Input image name
		bool _active; ///< is the camera currently in use.
	};

} // namespace sibr
