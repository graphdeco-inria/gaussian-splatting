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
# include <core/graphics/RenderTarget.hpp>
# include <core/graphics/Shader.hpp>
# include <core/graphics/Texture.hpp>
# include <core/graphics/Image.hpp>
# include <core/scene/BasicIBRScene.hpp>

namespace sibr { 

	/** Store a set of masks associated to a set of images (dataset input images for instance), on the GPU.
	This version uses a list of R8 rendertargets.
	\note Might want to use textures instead of RTs here.
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT RenderMaskHolder
	{
		typedef	RenderTargetLum::Ptr	MaskPtr;
	public:

		/** Update the masks
		\param masks the new masks to use
		*/
		void							setMasks( const std::vector<MaskPtr>& masks );

		/** \return the masks rendertargets. */
		const std::vector<MaskPtr>&		getMasks( void ) const;

		/** \return true if masks are available (non empty list). */
		bool							useMasks( void ) const;

		/** Load masks from black and white images on disk.
		\param ibrScene the dataset scene associated to the masks
		\param maskDir the masks directory
		\param preFileName mask filename prefix
		\param postFileName mask filename suffix and extension
		\param w target width
		\param h target height
		*/
		void 							loadMasks(
											const sibr::BasicIBRScene::Ptr& ibrScene, 
											const std::string& maskDir, const std::string& preFileName, 
											const std::string& postFileName, int w, int h);

		/** Upload a mask image to the GPU.
		\param img the mask image to upload
		\param i the mask index in the list
		\param masks the uploaded masks list (will be updated)
		\param invert should the mask be inverted
		**/
	    void 							uploadMaskGPU(sibr::ImageL8& img, int i, std::vector<RenderTargetLum::Ptr> & masks, bool invert) ;

	private:

		std::vector<MaskPtr>	_masks; ///< List of masks on the GPU.

	};

	/** Store a set of masks associated to a set of images (dataset input images for instance), on the GPU.
	This version uses a R8 texture array.
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT RenderMaskHolderArray
	{
		using MaskArray = sibr::Texture2DArrayLum;
		using MaskArrayPtr = MaskArray::Ptr;

	public:

		/** Update the masks
		\param masks the new masks to use
		*/
		void							setMasks(const MaskArrayPtr& masks);

		/** \return the masks texture array. */
		const MaskArrayPtr &				getMasks(void) const;

		/** Load masks from black and white images on disk.
		\param ibrScene the dataset scene associated to the masks
		\param maskDir the masks directory
		\param preFileName mask filename prefix
		\param postFileName mask filename suffix and extension
		\param w target width
		\param h target height
		*/
		void 							loadMasks(
			const sibr::BasicIBRScene::Ptr& ibrScene,
			const std::string& maskDir = "", const std::string& preFileName = "masks" ,
			const std::string& postFileName = "", int w = -1, int h = -1
		);

	protected:

		MaskArrayPtr _masks; ///< The masks texture array.

	};

} /*namespace sibr*/ 
