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

#include <core/scene/IIBRScene.hpp>

namespace sibr {

	/**
	* Class used to define a basic IBR Scene 
	* containing multiple components required to define a scene.
	* 
	* \ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT BasicIBRScene: public IIBRScene
	{
		
	public:

		/**
		* \brief Pointer to the instance of class sibr::BasicIBRScene.
		*/
		SIBR_CLASS_PTR(BasicIBRScene);

		/**
		 * \brief Default constructor to create a BasicIBRScene.
		 */
		BasicIBRScene();

		/**
		 * \brief Constructor to create a BasicIBRScene given command line arguments.
		 * The scene may be created using either dataset path, or explicitly specifying individual componenets.
		 * \param myArgs to provide all command line arguments containing path to specific components.
		 * \param noRTs to specify whether to initialize render target textures or not.
		 * \param noMesh skip loading the mesh
		 */
		BasicIBRScene(const BasicIBRAppArgs & myArgs, bool noRTs, bool noMesh = false);

		/**
		 * \brief Constructor to create a BasicIBRScene given command line arguments.
		 * The scene may be created using either dataset path, or explicitly specifying individual componenets.
		 * \param myArgs to provide all command line arguments containing path to specific components.
		 * \param myOpts to specify initialization paramters for the scene.
		 */
		BasicIBRScene(const BasicIBRAppArgs& myArgs, SceneOptions myOpts = SceneOptions());


		/** Destructor. */
		~BasicIBRScene() {};

		/**
		* \brief Creates a BasicIBRScene given custom data argument.
		* The scene will be created using the custom data (cameras/images/proxies/textures etc.) provided.
		* \param data to provide data instance holding customized components.
		* \param width the constrained width for GPU texture data.
		* \param myOpts to specify whether to initialize specific parts of the scene (RTs, geometry,...)
		*/
		void createFromCustomData(const IParseData::Ptr & data, const uint width = 0, SceneOptions myOpts = SceneOptions()) override;
		
		/**
		 * \brief Function to create a scene directly using the dataset path specified in command-line.
		 */
		void createFromDatasetPath() {};

		/**
		* \brief Function to generate render targets using the _data (regarding cameras, images, proxies ) parsed from metadata file.
		*/
		void createRenderTargets() override;
		
				/**
		 * \brief Getter for the pointer holding the data related to the scene.
		 * 
		 */
		const IParseData::Ptr						data(void) const override;

		/**
		* \brief Setter for the pointer holding the data related to the scene for scene creation.
		* \param data the setup data
		*/
		void										data(const sibr::IParseData::Ptr & data) override;


		/**
		 * \brief Getter for the pointer holding cameras related to each input iamge of the scene.
		 *
		 */
		const ICalibratedCameras::Ptr				cameras(void) const override;

		/**
		 * \brief Getter for the pointer holding the input images to the scene.
		 *
		 */
		const IInputImages::Ptr						images(void) const override;

		/**
		 * \brief Getter for the pointer holding the proxies required by the scene.
		 *
		 */
		const IProxyMesh::Ptr						proxies(void) const override;

		/**
		 * \brief Getter for the pointer holding the render targets textures related to the scene.
		 *
		 */
		const RenderTargetTextures::Ptr	&		renderTargets(void) const override;
		
		/**
		 * \brief Getter for the pointer holding the render targets textures related to the scene.
		 *
		 */
		RenderTargetTextures::Ptr &				renderTargets(void) override;

		/**
		 * \brief Getter for the pointer holding the mesh textures related to the mesh loaded for the scene.
		 *
		 */
		Texture2DRGB::Ptr &						inputMeshTextures(void) override;

	protected:
		BasicIBRScene(BasicIBRScene & scene);
		BasicIBRScene& operator =(const BasicIBRScene&) = delete;

		IParseData::Ptr				_data;
		ICalibratedCameras::Ptr		_cams;
		IInputImages::Ptr			_imgs;
		IProxyMesh::Ptr				_proxies;
		Texture2DRGB::Ptr			_inputMeshTexture;
		RenderTargetTextures::Ptr	_renderTargets;
		SceneOptions				_currentOpts;

		/**
		* \brief Creates a BasicIBRScene from the internal stored data component in the scene.
		* The data could be populated either from dataset path or customized by the user externally.
		* \param width the constrained width for GPU texture data.
		*/
		void createFromData(const uint width = 0);

		
	};

	///// INLINE DEFINITIONS /////

	inline const IParseData::Ptr			BasicIBRScene::data(void) const
	{
		return _data;
	}

	inline void BasicIBRScene::data(const IParseData::Ptr & data) 
	{
		_data = data;
	}

	inline const ICalibratedCameras::Ptr BasicIBRScene::cameras(void) const
	{
		return _cams;
	}

	inline const IInputImages::Ptr BasicIBRScene::images(void) const
	{
		return _imgs;
	}

	inline const IProxyMesh::Ptr BasicIBRScene::proxies(void) const
	{
		return _proxies;
	}

	inline const RenderTargetTextures::Ptr & BasicIBRScene::renderTargets(void) const
	{
		return _renderTargets;
	}

	inline RenderTargetTextures::Ptr & BasicIBRScene::renderTargets(void)
	{
		return _renderTargets;
	}

	inline Texture2DRGB::Ptr & BasicIBRScene::inputMeshTextures(void)
	{
		return _inputMeshTexture;
	}

}
