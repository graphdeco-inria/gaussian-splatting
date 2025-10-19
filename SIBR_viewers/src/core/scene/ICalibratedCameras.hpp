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
# include "core/scene/IParseData.hpp"
#include "core/scene/Config.hpp"

namespace sibr
{
	/**
	\ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT ICalibratedCameras {
	public:

		/**
		* \brief Pointer to the instance of class sibr::CalibratedCameras.
		*/
		typedef std::shared_ptr<ICalibratedCameras>	Ptr;

		// load from a path on disk in a predefined format (or could detect from file extension)

		/**
		* \brief Creates the calibrated cameras for a scene given data parsed from dataset path.
		* 
		* \param data Holds all information required to created a set of calibrated cameras
		*/
		virtual void	setupFromData(const IParseData::Ptr & data) = 0;

		/**
		* \brief Assigns the calibrated cameras for a scene to a list of cameras passed as parameter.
		*
		* \param cams Vector of type sibr::InputCamera to which the scene inputCameras will be set
		*/
		virtual void	setupCamerasFromExisting(const std::vector<InputCamera::Ptr> & cams) = 0;
		
		/**
		* \brief Function to set a camera as active.
		* 
		* \param camId Integer ID of the camera to be set active
		*/
		virtual void	activateCamera(uint camId) = 0;

		/**
		* \brief Function to set a camera as inactive.
		*
		* \param camId Integer ID of the camera to be set inactive
		*/
		virtual void	deactivateCamera(uint camId) = 0;

		/**
		* \brief Function to mark the cameras used for rendering.
		* Generally used for debugging purposes
		* \param selectedCameras list of camera IDs that are used for rendering
		*/
		virtual void	debugFlagCameraAsUsed(const std::vector<uint>& selectedCameras) = 0;

		/**
		* \brief Function to check if the camera is used for rendering.
		*
		* \param camId Integer ID of the cameras to be checked if it is being used for rendering
		* \return true if used for rendering
		*/
		virtual bool	isCameraUsedForRendering(size_t camId) const = 0;
		
		/**
		* \brief Function to set the cameras used for rendering.
		*
		* \param usedCamera Vector to specify which cameras are used for rendering 
		*/
		virtual void	usedCameraForRendering(const std::vector<bool> usedCamera) = 0;

		/**
		* \brief Getter to the vector of input cameras used to create the scene
		*
		*/
		virtual const std::vector<InputCamera::Ptr>&				inputCameras(void) const = 0;


		virtual const void							updateNearsFars(std::vector<sibr::Vector2f> & nearsFars) = 0;

	};
}