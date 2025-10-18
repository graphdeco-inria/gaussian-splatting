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
#include "picojson/picojson.hpp"
#include "core/scene/Config.hpp"
#include "core/system/CommandLineArgs.hpp"
#
#include "core/system/Matrix.hpp"
#include "core/assets/ImageListFile.hpp"
#include "core/assets/InputCamera.hpp"


#include <iostream>
#include <vector>
#include <string>


namespace sibr{

	/**
	* Interface used to store the data required for defining an IBR Scene
	* 
	*
	* Members:
	* - _basePathName: Base dataset directory path.
	* - _camInfos: Vector of sibr::InputCamera holding all data attached with the scene cameras.
	* - _meshPath: Filepath of the mesh associated to the scene.
	* - _imgInfos: Vector of sibr::ImageListFile::Infos holding filename, width, height, and id of the input images.
	* - _imgPath: Path to the calibrated images directory.
	* - _activeImages: Vector of bools storing active state of the camera.
	* - _numCameras: Number of cameras associated with the dataset
	* - _datasetType: Type if dataset being used. Currently supported: COLMAP, SIBR_BUNDLER, NVM, MESHROOM
	*
	* \ingroup sibr_scene
	*/

	class SIBR_SCENE_EXPORT IParseData {
		
	public:

		/**
		 * \brief Denotes the type of dataset represented by a IParseData object.
		* \ingroup sibr_scene
		*/
		enum class Type {
			EMPTY, GAUSSIAN, BLENDER, SIBR, COLMAP_CAPREAL, COLMAP, COLMAP2, NVM, MESHROOM, CHUNKED, EXTERNAL
		};

		/**
		* \brief Pointer to the instance of class sibr::IParseData.
		*/
		typedef std::shared_ptr<IParseData>				Ptr;

		/**
		* \brief Function to parse data from a dataset path. Will automatically determine the type of dataset based on the files present.
		* \param myArgs Arguments containing the dataset path and other infos
		* \param customPath additional data path
		*/
		virtual void  getParsedData(const BasicIBRAppArgs & myArgs, const std::string & customPath = "") = 0;

		/**
		* \brief Getter for the information regarding the input images.
		*
		*/
		virtual const std::vector<sibr::ImageListFile::Infos>&	imgInfos(void) const = 0;

		/**
		* \brief Setter for the information regarding the input images.
		*
		*/
		virtual void											imgInfos(std::vector<sibr::ImageListFile::Infos>& infos) = 0;

		/**
		* \brief Getter to the number of cameras defined in the bundle file.
		*
		*/
		virtual const int										numCameras(void) const = 0;

		/**
		* \brief Setter to the number of cameras defined in the bundle file.
		*
		*/
		virtual void											numCameras(int numCams) = 0;

		/**
		* \brief Getter for the list of active cameras/images.
		*
		*/
		virtual const std::vector<bool>&						activeImages(void) const = 0;

		/**
		* \brief Setter for the list of active cameras/images.
		*
		*/
		virtual void											activeImages(std::vector<bool>& activeCams) = 0;

		/**
		* \brief Getter for the base path name where the dataset is located.
		*
		*/
		virtual const std::string&								basePathName(void) const = 0;

		/**
		* \brief Setter for the base path name where the dataset is located.
		*
		*/
		virtual void											basePathName(std::string & path)  = 0;
		
		/**
		* \brief Getter for the mesh path where the dataset is located.
		*
		*/
		virtual const std::string&								meshPath(void) const = 0;

		/**
		* \brief Setter for the mesh path where the dataset is located.
		*
		*/
		virtual void											meshPath(std::string & path)  = 0;

		/**
		* \brief Getter for the dataset type.
		*
		*/
		virtual const IParseData::Type&							datasetType(void) const = 0;

		/**
		* \brief Setter for the dataset type.
		*
		*/
		virtual void											datasetType(IParseData::Type dataType) = 0;

		/**
		* \brief Getter for the camera infos.
		*
		*/
		virtual const std::vector<InputCamera::Ptr>	cameras(void) const = 0;

		/**
		* \brief Setter for the camera infos.
		*
		*/
		virtual void											cameras(std::vector<InputCamera::Ptr>& cams) = 0;

		/**
		* \brief Getter for the image path.
		*
		*/
		virtual const std::string								imgPath(void) const = 0;

		/**
		* \brief Setter for the image path.
		*
		*/
		virtual void											imgPath(std::string& imPath) = 0;
		
	};

}