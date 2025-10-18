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


# include "Config.hpp"
# include <core/assets/Resources.hpp>
# include <map>
# include "ULRV2Renderer.hpp"
#include "core/system/String.hpp"

namespace sibr {
		ULRV2Renderer::ULRV2Renderer(const std::vector<InputCamera::Ptr> & cameras, const uint w, const uint h, const unsigned int maxCams, const std::string & fShader, const std::string & vShader, const bool facecull)
		{
					
			// Count how many cameras are active in the scene.
			unsigned int numActiveCams = 0;
			for (auto & cam : cameras) {
				if (cam->isActive()) {
					++numActiveCams;
				}
			}
			_numCams = maxCams == 0 ? numActiveCams : std::min(maxCams, numActiveCams);
			
			setupULRshader(fShader,vShader);

			_depthRT.reset(new sibr::RenderTargetRGBA32F(w, h));

			_doOccl = true;
			_areMasksBinary = true;
			_doInvertMasks = false;
			_discardBlackPixels = true;
			_shouldCull = facecull;
			_epsilonOcclusion = 1e-2f;
			_soft_visibility_threshold = 30.0f;
			soft_visibility_maps = nullptr;
		}

		void ULRV2Renderer::setupULRshader(const std::string & fShader, const std::string & vShader)
		{
			std::cerr << "[ULRV2Renderer] Trying to initialize shaders for at most " << _numCams << " cameras." << std::endl;
			/// \todo TODO SR: handle the case were we require more shader texture slots than we are allowed too.
			/// Seems to be around 90 on Quadro K4200. We can either do multiple passes (fi 40 cams per pass),
			/// or try to use texture arrays to avoid this problem.
			/// If this happens to you, lower the maximum number of cameras picked by the ulr algo.

			GLShader::Define::List defines;
			defines.emplace_back("NUM_CAMS", _numCams);
			_ulrShader.init("ULRV2",
				sibr::loadFile(sibr::getShadersDirectory("") + "/" + vShader + ".vert"),
				sibr::loadFile(sibr::getShadersDirectory("") + "/" + fShader + ".frag", defines));
			_depthShader.init("ULRV2Depth",
				sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr_intersect.vert"),
				sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr_intersect.frag", defines));

			_proj.init(_depthShader, "proj");
			_ncamPos.init(_ulrShader, "ncam_pos");
			_occTest.init(_ulrShader, "occ_test");
			_areMasksBinaryGL.init(_ulrShader, "is_binary_mask");
			_doInvertMasksGL.init(_ulrShader, "invert_mask");
			_discardBlackPixelsGL.init(_ulrShader, "discard_black_pixels");
			_doMask.init(_ulrShader, "doMasking");
			_camCount.init(_ulrShader, "camsCount");
			_use_soft_visibility.init(_ulrShader, "useSoftVisibility");
			_soft_visibility_threshold.init(_ulrShader, "softVisibilityThreshold");
			_epsilonOcclusion.init(_ulrShader, "epsilonOcclusion");

			_icamProj.resize(_numCams);
			_icamPos.resize(_numCams);
			_icamDir.resize(_numCams);
			_inputRGB.resize(_numCams);
			_masks.resize(_numCams);
			_selected_cams.resize(_numCams);

			_ulrShader.begin();
			for (uint i = 0; i<(uint)_numCams; i++)
			{
				_icamProj[i].init(_ulrShader, sibr::sprint("icam_proj[%d]", i));
				_icamPos[i].init(_ulrShader, sibr::sprint("icam_pos[%d]", i));
				_icamDir[i].init(_ulrShader, sibr::sprint("icam_dir[%d]", i));
				_selected_cams[i].init(_ulrShader, sibr::sprint("selected_cams[%d]", i));
				_inputRGB[i].init(_ulrShader, sibr::sprint("input_rgb[%d]", i));
				_inputRGB[i].set(i + 2);  // location 0 and 1 reserved.s
				_masks[i].init(_ulrShader, sibr::sprint("masks[%d]", i));
				_masks[i].set(GLuint(_numCams + i + 2));

			}
			_ulrShader.end();

		}
		
		void
			ULRV2Renderer::process(const std::vector<uint>& imgs_ulr, const sibr::Camera& eye,
				const sibr::BasicIBRScene::Ptr& scene,
				std::shared_ptr<sibr::Mesh>& altMesh,
				const std::vector<std::shared_ptr<RenderTargetRGBA32F> >& inputRTs,
				IRenderTarget& dst)
		{
			// Get a new camera with z_near ~ 0
			sibr::Camera new_cam = eye;
			//new_cam.znear(0.001f);


			glViewport(0, 0, _depthRT->w(), _depthRT->h());
			_depthRT->bind();
			glClearColor(0, 0, 0, 1);
			glClearDepth(1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			_depthShader.begin();
			_proj.set(new_cam.viewproj());
			if (altMesh != nullptr) {
				altMesh->render(true, _shouldCull); // enable depth test - disable back culling
			} else {
				scene->proxies()->proxy().render(true, _shouldCull);
			}
			_depthShader.end();
			_depthRT->unbind();
			
			glViewport(0, 0, dst.w(), dst.h());
			dst.clear();
			dst.bind();

			_ulrShader.begin();
			
			_ncamPos.set(eye.position());
			_occTest.set(_doOccl);
			_areMasksBinaryGL.set(_areMasksBinary);
			_doInvertMasksGL.set(_doInvertMasks);
			_discardBlackPixelsGL.set(_discardBlackPixels);
			_doMask.set(useMasks());
			_epsilonOcclusion.send();

			CHECK_GL_ERROR

			_use_soft_visibility.set(soft_visibility_maps != nullptr && soft_visibility_maps->handle());

			CHECK_GL_ERROR

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, _depthRT->texture());

			CHECK_GL_ERROR
			
			if (_use_soft_visibility) {
				//std::cout << "using soft visib" << std::endl;
				_soft_visibility_threshold.send();

				CHECK_GL_ERROR;

				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D_ARRAY, soft_visibility_maps->handle());

				CHECK_GL_ERROR;
			}

			CHECK_GL_ERROR;

			int usedCamerasCount = 0;
			
			for (int i = 0; i < std::min(imgs_ulr.size(), _numCams); ++i) {
				
				if (!scene->cameras()->inputCameras()[imgs_ulr[i]]->isActive()) {
					continue;
				}
				
				auto& cam = *scene->cameras()->inputCameras()[imgs_ulr[i]];
				_icamPos[usedCamerasCount].set(cam.position());
				_icamDir[usedCamerasCount].set(cam.dir());
				_icamProj[usedCamerasCount].set(cam.viewproj());
				_selected_cams[usedCamerasCount].set((int)imgs_ulr[i]);
				glActiveTexture(GL_TEXTURE0 + usedCamerasCount + 2);
				glBindTexture(GL_TEXTURE_2D, inputRTs[imgs_ulr[i]]->texture());

				if (useMasks()) {
					glActiveTexture(GL_TEXTURE0 + (int)_numCams + usedCamerasCount + 2);
					glBindTexture(GL_TEXTURE_2D, getMasks()[imgs_ulr[i]]->texture());
				}
				++usedCamerasCount;
			}

			CHECK_GL_ERROR;

			_camCount.set(usedCamerasCount);
			
			CHECK_GL_ERROR;

			//glDisable(GL_DEPTH_TEST);
			RenderUtility::renderScreenQuad();

			CHECK_GL_ERROR;

			_ulrShader.end();
			dst.unbind();

		}



	} /*namespace sibr*/
