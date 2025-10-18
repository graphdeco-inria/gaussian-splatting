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

namespace sibr { 

	/** Second version of the ULR render. use separate samplers for each input image.
	 */
	class SIBR_EXP_ULR_EXPORT ULRV2Renderer : public RenderMaskHolder
	{
		SIBR_CLASS_PTR(ULRV2Renderer);

		/** Constructor.
		 *\param cameras input cameras
		 *\param w rendering width
		 *\param h rendering height
		 *\param maxCams maximum number of cameras selcted for rendering a frame
		 *\param fShader name of the fragment shader
		 *\param vShader name of the vertex shader
		 *\param facecull should backface culling be performed during the prepass.
		 **/
		ULRV2Renderer(const std::vector<InputCamera::Ptr> & cameras, const uint w, const uint h, const unsigned int maxCams = 0, const std::string & fShader = "ulr/ulr_v2", const std::string & vShader = "ulr/ulr_v2", const bool facecull = true);

		/** Setup the ULR shaders.
		 *\param fShader name of the fragment shader
		 *\param vShader name of the vertex shader
		 **/
		void setupULRshader(const std::string & fShader = "ulr/ulr_v2", const std::string & vShader = "ulr/ulr_v2");

		/** Render.
		 *\param imgs_ulr vector of selected image IDs
		 *\param eye novel viewpoint
		 *\param scene the scene to render
		 *\param altMesh optional alternative mesh
		 *\param inputRTs the RGBD input images
		 *\param dst destination target
		 */
		void process(const std::vector<uint>& imgs_ulr, const sibr::Camera& eye,
			const sibr::BasicIBRScene::Ptr& scene,
			std::shared_ptr<sibr::Mesh>& altMesh,
			const std::vector<std::shared_ptr<RenderTargetRGBA32F> >& inputRTs,
			IRenderTarget& dst);

		/** Should occlusion testing be performed.
		 *\param val true if testing should occur
		 */
		void doOccl(bool val) { _doOccl = val; }

		/** \return a reference to the occlusion threshold */
		float & epsilonOcclusion() { return _epsilonOcclusion; }

		/** Are the mask smooth values or binary.
		 *\param val true if they are binary
		 */
		void setAreMasksBinary(bool val) { _areMasksBinary = val; }

		/** Should the masks be inverted.
		 *\param val true if they should
		 */
		void setDoInvertMasks(bool val) { _doInvertMasks = val; }

		/** Should black pixels be ignored when accumulating colors.
		 *\param val true if they should be ignored
		 */
		void setDiscardBlackPixels(bool val) { _discardBlackPixels = val; }

		/** Should backface culling be performed.
		 *\param val true if it should
		 */
		void setCulling(bool val) { _shouldCull = val; }

		/** \return a pointer to the soft visibility texture array if it exists */
		Texture2DArrayLum32F * & getSoftVisibilityMaps(void) { return soft_visibility_maps; }

		/** \return a reference to the soft visibility threshold. */
		sibr::GLuniform<float> & getSoftVisibilityThreshold() { return _soft_visibility_threshold; }

		/** \return a pointer to the ULR OpenGL program. */
		sibr::GLShader * getProgram() { return &_ulrShader; }

		/** \return the number of cameras */
		size_t getNumCams() { return _numCams; }

	public:
		sibr::RenderTargetRGBA32F::Ptr _depthRT; ///< the prepass render target.

	private:
		
		sibr::GLShader _ulrShader;
		sibr::GLShader _depthShader;

		std::vector<sibr::GLParameter>	_icamProj;
		std::vector<sibr::GLParameter>	_icamPos;
		std::vector<sibr::GLParameter>	_icamDir;
		std::vector<sibr::GLParameter>	_inputRGB;
		std::vector<sibr::GLParameter>	_masks;
		std::vector<sibr::GLuniform<int> >	_selected_cams;

		Texture2DArrayLum32F * soft_visibility_maps;
		sibr::GLuniform<float> _soft_visibility_threshold;
		sibr::GLuniform<bool> _use_soft_visibility;

		sibr::GLParameter _occTest;
		sibr::GLParameter _areMasksBinaryGL;
		sibr::GLParameter _doInvertMasksGL;
		sibr::GLParameter _discardBlackPixelsGL;
		sibr::GLParameter _doMask;
		sibr::GLParameter _ncamPos;
		sibr::GLParameter _camCount;
		sibr::GLParameter _proj;
		sibr::GLuniform<float> _epsilonOcclusion;

		bool	_doOccl;
		bool	_areMasksBinary;
		bool	_doInvertMasks;
		bool	_discardBlackPixels;
		bool _shouldCull;
		size_t _numCams;
	
   };

} /*namespace sibr*/

