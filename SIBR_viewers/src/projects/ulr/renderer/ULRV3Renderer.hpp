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

# include "Config.hpp"
# include <core/system/Config.hpp>
# include <core/graphics/Texture.hpp>
# include <core/graphics/Shader.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/renderer/RenderMaskHolder.hpp>
# include <core/scene/BasicIBRScene.hpp>
# include <core/system/SimpleTimer.hpp>

namespace sibr { 
	
	/**
	 * \class ULRV3Renderer
	 * \brief Perform per-pixel Unstructured Lumigraph Rendering (Buehler et al., 2001). No selection is done on the CPU side.
	 * Relies on texture arrays and uniform buffer objects to support a high number of cameras. 
	 */
	class SIBR_EXP_ULR_EXPORT ULRV3Renderer : public RenderMaskHolderArray
	{
		SIBR_CLASS_PTR(ULRV3Renderer);
	
	public:

		/**
		 * Constructor.
		 * \param cameras The input cameras to use.
		 * \param w The width of the internal rendertargets.
		 * \param h The height of the internal rendertargets.
		 * \param fShader An optional name of the fragment shader to use (default to ulr_v3).
		 * \param vShader An optional name of the vertex shader to use (default to ulr_v3).
		 * \param facecull Should the mesh be renderer with backface culling.
		 */
		ULRV3Renderer(const std::vector<InputCamera::Ptr> & cameras, 
			const uint w, const uint h, 
			const std::string & fShader = "ulr/ulr_v3", 
			const std::string & vShader = "ulr/ulr_v3", 
			const bool facecull = true
		);

		/**
		 * Change the shaders used by the ULR renderer.
		 * \param fShader The name of the fragment shader to use.
		 * \param vShader The name of the vertex shader to use.
		 */
		virtual void setupShaders(
			const std::string & fShader = "ulr/ulr_v3",
			const std::string & vShader = "ulr/ulr_v3"
		);

		/**
		 * Performs ULR rendering to a given destination rendertarget.
		 * \param mesh The mesh to use as geometric proxy.
		 * \param eye The novel viewpoint.
		 * \param dst The destination rendertarget.
		 * \param inputRGBs A texture array containing the input RGB images.
		 * \param inputDepths A texture array containing the input depth maps.
		 * \param passthroughDepth If true, depth from the position map will be output to the depth buffer for ulterior passes.
		 */
		virtual void process(
			const sibr::Mesh & mesh,
			const sibr::Camera& eye,
			IRenderTarget& dst,
			const sibr::Texture2DArrayRGB::Ptr & inputRGBs,
			const sibr::Texture2DArrayLum32F::Ptr & inputDepths,
			bool passthroughDepth = false
			);

		/**
		 * Performs ULR rendering to a given destination rendertarget.
		 * \param mesh The mesh to use as geometric proxy.
		 * \param eye The novel viewpoint.
		 * \param dst The destination rendertarget.
		 * \param inputRGBHandle The handle of a texture array containing the input RGB images.
		 * \param inputDepths A texture array containing the input depth maps.
		 * \param passthroughDepth If true, depth from the position map will be output to the depth buffer for ulterior passes.
		 */
		virtual void process(
			const sibr::Mesh & mesh,
			const sibr::Camera& eye,
			IRenderTarget& dst,
			uint inputRGBHandle,
			const sibr::Texture2DArrayLum32F::Ptr & inputDepths,
			bool passthroughDepth = false
		);

		/** 
		 *  Update which cameras should be used for rendering, based on the indices passed.
		 *  \param camIds The indices to enable.
		 **/
		void updateCameras(const std::vector<uint> & camIds);

		/// Set the epsilon occlusion threshold.
		float & epsilonOcclusion() { return _epsilonOcclusion.get(); }

		/// Enable or disable the masks.
		bool & useMasks() { return _useMasks.get(); }

		/// Flip the RGB images before using them.
		bool & flipRGBs() { return _flipRGBs.get(); }

		/// Enable or diable occlusion testing.
		bool& occTest() { return _occTest.get(); }

		/// Show debug weights.
		bool & showWeights() { return _showWeights.get(); }

		/// Set winner takes all weights strategy
		bool & winnerTakesAll() { return _winnerTakesAll.get(); }

		/// Apply gamma correction to the output.
		bool & gammaCorrection() { return _gammaCorrection.get(); }

		/// Apply backface culling to the mesh.
		bool & backfaceCull() { return _backFaceCulling; }

		/** Resize the internal rendertargets.
		 *\param w the new width
		 *\param h the new height
		 **/
		void resize(const unsigned int w, const unsigned int h);

		/// Should the final RT be cleared or not.
		bool & clearDst() { return _clearDst; }

		/// \return The ID of the first pass position map texture.
		uint depthHandle() const { return _depthRT->texture(); }

		void startProfile() { 
			_profiling = true; 
			_depthCost.clear();
			_blendCost.clear();
		}

		void stopProfile();

		/**
		 * Render the world positions of the proxy points in an intermediate rendertarget.
		 * \param mesh the proxy mesh.
		 * \param eye The novel viewpoint.
		 */
		virtual void renderProxyDepth(const sibr::Mesh & mesh, const sibr::Camera& eye);

		/**
		* Perform ULR blending.
		* \param eye The novel viewpoint.
		* \param dst The destination rendertarget.
		* \param inputRGBHandle The handle to a texture array containing the input RGB images.
		* \param inputDepths A texture array containing the input depth maps.
		* \param passthroughDepth If true, depth from the position map will be output to the depth buffer for ulterior passes.
		*/
		virtual void renderBlending(
			const sibr::Camera& eye,
			IRenderTarget& dst,
			uint inputRGBHandle,
			const sibr::Texture2DArrayLum32F::Ptr & inputDepths,
			bool passthroughDepth
		);


	protected:
		/// Shader names.
		std::string fragString, vertexString;

		sibr::GLShader _ulrShader;
		sibr::GLShader _depthShader;

		sibr::RenderTargetRGBA32F::Ptr		_depthRT;
		GLuniform<Matrix4f>					_nCamProj;
		GLuniform<Vector3f>					_nCamPos;

		GLuniform<bool>
			_occTest = true,
			_useMasks = false,
			_discardBlackPixels = true,
			_areMasksBinary = true,
			_invertMasks = false,
			_flipRGBs = false,
			_showWeights = false,
			_winnerTakesAll = false,
			_gammaCorrection = false;

		size_t _maxNumCams = 0;
		GLuniform<int> _camsCount = 0;

		GLuniform<float>					_epsilonOcclusion = 0.01f;
		bool								_backFaceCulling = true;
		bool								_clearDst = true;

		/** Camera infos data structure shared between the CPU and GPU.
			We have to be careful about alignment if we want to send those struct directly into the UBO. */
		struct CameraUBOInfos {	 
			Matrix4f vp; ///< Matrix viewproj.
			Vector3f pos; ///< Camera position.
			int selected = 0; ///< Is the camera selected (0/1).
			Vector3f dir; ///< Camera direction.
			float dummy = 0.0f; ///< Padding to a multiple of 16 bytes for alignment on the GPU.
		};

		std::vector<CameraUBOInfos> _cameraInfos;
		GLuint _uboIndex;

		bool		_profiling = false;
		sibr::Timer	_depthPassTimer;
		sibr::Timer	_blendPassTimer;
		int											_numFramesProfiling = 100;
		std::string									_profileStr = "";
		std::vector<float>							_depthCost, _blendCost;

	};


} /*namespace sibr*/ 

