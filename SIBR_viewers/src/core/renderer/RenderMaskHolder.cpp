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


#include <core/renderer/RenderMaskHolder.hpp>
#include <core/assets/Resources.hpp>

namespace sibr { 
	void	RenderMaskHolder::setMasks( const std::vector<MaskPtr>& masks )
	{
		_masks = masks;
	}

	const std::vector<RenderMaskHolder::MaskPtr>&	RenderMaskHolder::getMasks( void ) const
	{
		return _masks;
	}

	bool	RenderMaskHolder::useMasks( void ) const
	{
		return _masks.empty() == false;
	}

	void 	RenderMaskHolder::uploadMaskGPU(sibr::ImageL8& img, int i, std::vector<RenderTargetLum::Ptr> & masks, bool invert) 
	{
		sibr::GLShader textureShader;
		textureShader.init("Texture",
			sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("texture.vp")), 
			invert ? sibr::loadFile(sibr::getShadersDirectory("core") + "/texture-invert.frag") : sibr::loadFile(sibr::getShadersDirectory("core") + "/texture.frag"));

		std::shared_ptr<sibr::RenderTargetLum> maskRTPtr;
		maskRTPtr.reset(new sibr::RenderTargetLum(img.w(), img.h()));

		img.flipH();
		std::shared_ptr<sibr::Texture2DLum> rawInputImage(new sibr::Texture2DLum(img));
		img.flipH();

		glViewport(0,0, img.w(), img.h());
		maskRTPtr->clear();
		maskRTPtr->bind();

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, rawInputImage->handle());

		glDisable(GL_DEPTH_TEST);            
		textureShader.begin();
		sibr::RenderUtility::renderScreenQuad();
		textureShader.end();

		maskRTPtr->unbind();
/*
	maskRTPtr->readBack(img);
	sibr::show(img);
*/
		masks.push_back(maskRTPtr);
	}


	void 	RenderMaskHolder::loadMasks(const sibr::BasicIBRScene::Ptr& ibrScene,  const std::string& maskDir,  
				const std::string& preFileName, const std::string& postFileName, int w, int h)
	{
		if( boost::filesystem::exists(maskDir) ) {

		for(int i=0; i<(int)ibrScene->cameras()->inputCameras().size(); i++ ) {
				sibr::ImageRGB mask;
				std::string filename = maskDir + "/" + preFileName + sibr::imageIdToString(i) + postFileName;
			
				if( boost::filesystem::exists(filename))  {
					mask.load(filename,false);
					// Split the image in its channels and keep only the first.
					cv::Mat channels[3];
					cv::split(mask.toOpenCV(), channels);
					sibr::ImageL8 maskOneChan;					
					maskOneChan.fromOpenCV(channels[0]);
					uploadMaskGPU(maskOneChan, i, _masks, false);
				}
				else {
					if( ibrScene->cameras()->inputCameras()[i]->isActive() ) 
						SIBR_ERR << "[RenderMaskHolder] couldnt find " << filename << std::endl;
					else { /// push back empty mask so array is consistent
						/// \todo TODO GD -- this is wasteful, should fine better way
						std::shared_ptr<sibr::RenderTargetLum> maskRTPtr;
						maskRTPtr.reset(new sibr::RenderTargetLum(w, h));
						_masks.push_back(maskRTPtr);
					}
				}
		}

	}
	else
		SIBR_ERR << "[RenderMaskHolder] Cant find directory " << maskDir << std::endl;
	}

	void	RenderMaskHolderArray::setMasks(const MaskArrayPtr& masks)
	{
		_masks = masks;
	}

	const RenderMaskHolderArray::MaskArrayPtr & RenderMaskHolderArray::getMasks(void) const {
		return _masks;
	}

	void RenderMaskHolderArray::loadMasks(
		const sibr::BasicIBRScene::Ptr& ibrScene,
		const std::string& maskDir, const std::string& preFileName,
		const std::string& postFileName, int w, int h
	) {
		std::string maskdir = (maskDir == "" ? ibrScene->data()->basePathName() + "/images/" : maskDir);

		if (!boost::filesystem::exists(maskdir)) {
			SIBR_ERR << "[RenderMaskHolder] Cant find directory " << maskDir << std::endl;
		} else {
			int numInputImgs = (int)ibrScene->cameras()->inputCameras().size();
			std::vector<cv::Mat> masks(numInputImgs);
			for (int i = 0; i < numInputImgs; i++) {
				std::string filename = maskDir + "/" + preFileName + sibr::imageIdToString(i) + postFileName;

				cv::Mat mask = cv::imread(filename);

				if (mask.empty()) {
					SIBR_ERR << "[RenderMaskHolderArray] couldnt find or read " << filename << std::endl;
				}

				cv::Mat channels[3];
				cv::split(mask, channels);
				if (w > 0 && h > 0) {
					cv::resize(channels[0], channels[0], cv::Size(w, h));
				}
				masks[i] = channels[0];
			}

			_masks = MaskArrayPtr(new MaskArray(masks, SIBR_FLIP_TEXTURE));
		}
	}
} /*namespace sibr*/ 
