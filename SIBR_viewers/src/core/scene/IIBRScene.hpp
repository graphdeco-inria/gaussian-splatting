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

#include <core/system/CommandLineArgs.hpp>
#include "core/raycaster/CameraRaycaster.hpp"
#include "core/scene/ICalibratedCameras.hpp"
#include "core/scene/IParseData.hpp"
#include "core/scene/IProxyMesh.hpp"
#include "core/scene/IInputImages.hpp"
#include "core/scene/RenderTargetTextures.hpp"
#include "core/scene/Config.hpp"
#include "core/system/String.hpp"

namespace sibr {

	/**
	* Interface used to define how an IBR Scene is shaped
	* containing multiple components required to define a scene.
	*
	* Members:
	* - ICalibratedCameras
	* - IInputImages
	* - IProxyMesh
	* - RenderTargetTextures
	* 
	* \ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT IIBRScene
	{

		/**
		* \brief Pointer to the instance of class sibr::IIBRScene.
		*/
		SIBR_CLASS_PTR(IIBRScene);
		
	public:
		/** Scene initialization infos. */
		struct SceneOptions
		{
			bool		renderTargets = true; ///< Load rendertargets?
			bool		mesh = true; ///< Load mesh?
			bool		images = true; ///< Load images?
			bool		cameras = true; ///< Load cameras?
			bool        texture = true; ///< Load texture ?

			SceneOptions() {}
		};

		/**
		* \brief Creates a BasicIBRScene given custom data argument.
		* The scene will be created using the custom data (cameras/images/proxies/textures etc.) provided.
		* \param data to provide data instance holding customized components.
		* \param width the constrained width for GPU texture data.
		* \param myOpts to specify whether to initialize specific parts of the scene (RTs, geometry,...)
		*/
		virtual void createFromCustomData(const IParseData::Ptr & data, const uint width = 0, SceneOptions myOpts = SceneOptions()) = 0;
		
		/**
		 * \brief Function to create a scene directly using the dataset path specified in command-line.
		 */
		virtual void	createFromDatasetPath() = 0;

		/**
		* \brief Function to generate render targets using the _data (regarding cameras, images, proxies ) parsed from metadata file.
		*/
		virtual void	createRenderTargets() = 0;


		/**
		 * \brief Getter for the pointer holding the data related to the scene.
		 * 
		 */
		virtual const IParseData::Ptr						data(void) const = 0;

		/**
		* \brief Setter for the pointer holding the data related to the scene for scene creation.
		* \param data the setup data
		*/
		virtual void										data(const sibr::IParseData::Ptr & data) = 0;


		/**
		 * \brief Getter for the pointer holding cameras related to each input iamge of the scene.
		 *
		 */
		virtual const ICalibratedCameras::Ptr				cameras(void) const = 0;

		/**
		 * \brief Getter for the pointer holding the input images to the scene.
		 *
		 */
		virtual const IInputImages::Ptr						images(void) const = 0;

		/**
		 * \brief Getter for the pointer holding the proxies required by the scene.
		 *
		 */
		virtual const IProxyMesh::Ptr						proxies(void) const = 0;

		/**
		 * \brief Getter for the pointer holding the render targets textures related to the scene.
		 *
		 */
		virtual const RenderTargetTextures::Ptr	&			renderTargets(void) const = 0;
		
		/**
		 * \brief Getter for the pointer holding the render targets textures related to the scene.
		 *
		 */
		virtual RenderTargetTextures::Ptr &					renderTargets(void) = 0;

		/**
		 * \brief Getter for the pointer holding the mesh textures related to the mesh loaded for the scene.
		 *
		 */
		virtual Texture2DRGB::Ptr &							inputMeshTextures(void) = 0;
		
	};
}
