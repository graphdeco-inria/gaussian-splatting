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

#include "core/graphics/Texture.hpp"
#include "core/scene/ICalibratedCameras.hpp"
#include "core/scene/IInputImages.hpp"
#include "core/scene/IProxyMesh.hpp"
#include "core/assets/Resources.hpp"
# include "core/graphics/Shader.hpp"
#include "core/graphics/Utils.hpp"
#include "core/scene/Config.hpp"


# define SIBR_SCENE_LINEAR_SAMPLING			4


namespace sibr{

	/** 
	\ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT RTTextureSize {

	public:
		RTTextureSize(uint w = 0) : _width(w) {}

		void initSize(uint w, uint h, bool force_aspect_ratio = false);

		bool isInit() const;

	protected:
		uint		_width = 0; //constrained width provided by the command line args, defaults to 0
		uint		_height = 0; //associated height, computed in initSize
		bool		_isInit = false;
		int			_initActiveCam = 0;

	};

	/**
	\ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT RGBDInputTextures : public virtual RTTextureSize {
		SIBR_CLASS_PTR(RGBDInputTextures)
	public:
		const std::vector<RenderTargetRGBA32F::Ptr> & inputImagesRT() const;

		virtual void initializeImageRenderTargets(ICalibratedCameras::Ptr cams, IInputImages::Ptr imgs);
		virtual void initializeDepthRenderTargets(ICalibratedCameras::Ptr cams, IProxyMesh::Ptr proxies, bool facecull);

	protected:
		std::vector<RenderTargetRGBA32F::Ptr> _inputRGBARenderTextures;

	};

	/**
	\ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT DepthInputTextureArray : public virtual RTTextureSize {
		SIBR_CLASS_PTR(DepthInputTextureArray)
	public:
		virtual void initDepthTextureArrays(ICalibratedCameras::Ptr cams, IProxyMesh::Ptr proxies, bool facecull, int flags = SIBR_GPU_LINEAR_SAMPLING);
		const Texture2DArrayLum32F::Ptr &  getInputDepthMapArrayPtr() const;

	protected:
		Texture2DArrayLum32F::Ptr _inputDepthMapArrayPtr;

	};
	/**
	\ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT RGBInputTextureArray : public virtual RTTextureSize {

		SIBR_CLASS_PTR(RGBInputTextureArray)

	public:
		virtual void initRGBTextureArrays(IInputImages::Ptr imgs, int flags = 0, bool force_aspect_ratio=false);
		const Texture2DArrayRGB::Ptr & getInputRGBTextureArrayPtr() const;

	protected:
		Texture2DArrayRGB::Ptr _inputRGBArrayPtr;

	};

	/**
	\ingroup sibr_scene
	*/
	class SIBR_SCENE_EXPORT RenderTargetTextures :
		public virtual RGBDInputTextures,
		public virtual DepthInputTextureArray,
		public virtual RGBInputTextureArray 
	{
		
	public:
		SIBR_CLASS_PTR(RenderTargetTextures)
		
		RenderTargetTextures(uint w = 0) : RTTextureSize(w) {}

		virtual void initRGBandDepthTextureArrays(ICalibratedCameras::Ptr cams, IInputImages::Ptr imgs, IProxyMesh::Ptr proxies, int textureFlags, unsigned int w, unsigned int h, bool faceCull = true);
		// TODO: remove this, not needed
		virtual void initRGBandDepthTextureArrays(ICalibratedCameras::Ptr cams, IInputImages::Ptr imgs, IProxyMesh::Ptr proxies, int textureFlags, int texture_width, bool faceCull = true, bool force_aspect_ratio = false);
		virtual void initRGBandDepthTextureArrays(ICalibratedCameras::Ptr cams, IInputImages::Ptr imgs, IProxyMesh::Ptr proxies, int textureFlags, bool faceCull = true, bool force_aspect_ratio=false);
		virtual void initializeDefaultRenderTargets(ICalibratedCameras::Ptr cams, IInputImages::Ptr imgs, IProxyMesh::Ptr proxies);

	protected:
		void initRenderTargetRes(ICalibratedCameras::Ptr cams);

	};


}
