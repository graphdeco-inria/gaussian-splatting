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


#include "BasicIBRScene.hpp"
#include <iostream>
#include <string>

#include "core/scene/CalibratedCameras.hpp"
#include "core/scene/ParseData.hpp"
#include "core/scene/ProxyMesh.hpp"
#include "core/scene/InputImages.hpp"

namespace sibr
{
	
	BasicIBRScene::BasicIBRScene() {
		_data.reset(new ParseData());
		_cams.reset(new CalibratedCameras());
		_imgs.reset(new InputImages());
		_proxies.reset(new ProxyMesh());
		_renderTargets.reset(new RenderTargetTextures());
	}

	BasicIBRScene::BasicIBRScene(const BasicIBRAppArgs & myArgs, bool noRTs, bool noMesh)
	{

		BasicIBRScene();
		// parse metadata file
		_data.reset(new ParseData());
		_currentOpts.renderTargets = !noRTs;
		_currentOpts.mesh = !noMesh;

		_data->getParsedData(myArgs);
		std::cout << "Number of input Images to read: " << _data->imgInfos().size() << std::endl;

		if (_data->imgInfos().size() != _data->numCameras())
			SIBR_ERR << "List Image file size do not match number of input cameras in Bundle file!" << std::endl;

		if (_data->datasetType() != IParseData::Type::EMPTY) {
			createFromData(myArgs.texture_width);
		}
	}

	BasicIBRScene::BasicIBRScene(const BasicIBRAppArgs& myArgs, SceneOptions myOpts)
	{
		BasicIBRScene();
		_currentOpts = myOpts;

		// parse metadata file
		_data.reset(new ParseData());


		_data->getParsedData(myArgs);
		std::cout << "Number of input Images to read: " << _data->imgInfos().size() << std::endl;

		if (_data->imgInfos().size() != _data->numCameras())
			SIBR_ERR << "List Image file size do not match number of input cameras in Bundle file!" << std::endl;

		if (_data->datasetType() != IParseData::Type::EMPTY) {
			createFromData(myArgs.texture_width);
		}
	}

	void BasicIBRScene::createFromCustomData(const IParseData::Ptr & data, const uint width, BasicIBRScene::SceneOptions myOpts)
	{
		_data = data;
		_currentOpts = myOpts;
		createFromData(width);
	}


	void BasicIBRScene::createRenderTargets()
	{
		_renderTargets->initializeDefaultRenderTargets(_cams, _imgs, _proxies);
	}

	BasicIBRScene::BasicIBRScene(BasicIBRScene & scene)
	{
		_data = scene.data();
		_cams = scene.cameras();
		_imgs = scene.images();
		_proxies = scene.proxies();
		_renderTargets = scene.renderTargets();
	}

	void BasicIBRScene::createFromData(const uint width)
	{
		_cams.reset(new CalibratedCameras());
		_imgs.reset(new InputImages());
		_proxies.reset(new ProxyMesh());

		// setup calibrated cameras
		if (_currentOpts.cameras) {
			
			_cams->setupFromData(_data);

			std::cout << "Number of Cameras set up: " << _cams->inputCameras().size() << std::endl;
		}

		// load input images

		uint mwidth = width;
		if (_currentOpts.images) {
			_imgs->loadFromData(_data);
			std::cout << "Number of Images loaded: " << _imgs->inputImages().size() << std::endl;

			if (width == 0) {// default
				if (_imgs->inputImages()[0]->w() > 1920) {
					SIBR_LOG << "Limiting width to 1920 for performance; use --texture-width to override" << std::endl;
					mwidth = 1920;
				}
			}
		}
		_renderTargets.reset(new RenderTargetTextures(mwidth));

		if (_currentOpts.mesh) {
			// load proxy
			_proxies->loadFromData(_data);


			std::vector<InputCamera::Ptr> inCams = _cams->inputCameras();
			float eps = 0.1f;
			if (inCams.size() > 0 && (abs(inCams[0]->znear() - 0.1) < eps || abs(inCams[0]->zfar() - 1000.0) < eps || abs(inCams[0]->zfar() - 100.0) < eps) && _proxies->proxy().triangles().size() > 0) {
				std::vector<sibr::Vector2f>    nearsFars;
				CameraRaycaster::computeClippingPlanes(_proxies->proxy(), inCams, nearsFars);
				_cams->updateNearsFars(nearsFars);
			}

			//// Load the texture.
			sibr::ImageRGB inputTextureImg;

			std::string texturePath, textureImageFileName;

			// Assumes that the texture is stored next to the mesh in the same directory
			// This information comes from Assimp and the mtl file if available
			if ((textureImageFileName = _proxies->proxy().getTextureImageFileName()) != "") {
				texturePath = sibr::parentDirectory(_data->meshPath()) + "/" + textureImageFileName;
				// check if full path given 
				if (!sibr::fileExists(texturePath) && sibr::fileExists(textureImageFileName)) 
					texturePath = textureImageFileName;
			}
			else {
				texturePath = sibr::parentDirectory(_data->meshPath()) + "/mesh_u1_v1.png";
				if (sibr::fileExists(texturePath)) {
					texturePath = sibr::parentDirectory(_data->meshPath()) + "/textured_u1_v1.png";
					if (!sibr::fileExists(texturePath)) 
						texturePath = sibr::parentDirectory(_data->meshPath()) + "/texture.png";
				}
			}


			if (_currentOpts.texture && sibr::fileExists(texturePath)) {
				inputTextureImg.load(texturePath);
				_inputMeshTexture.reset(new sibr::Texture2DRGB(inputTextureImg, SIBR_GPU_LINEAR_SAMPLING));
			}

		}

		if (_currentOpts.renderTargets) {
			createRenderTargets();
		}
	}
	
}
