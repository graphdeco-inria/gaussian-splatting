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


#include <projects/ulr/renderer/ULRV3Renderer.hpp>



sibr::ULRV3Renderer::ULRV3Renderer(const std::vector<InputCamera::Ptr> & cameras, const uint w, const uint h, const std::string & fShader, const std::string & vShader, const bool facecull)
{
	_backFaceCulling = facecull;
	fragString = fShader;
	vertexString = vShader;
	_maxNumCams = cameras.size();
	_camsCount = int(_maxNumCams);

	// Populate the cameraInfos array (will be uploaded to the GPU).
	_cameraInfos.clear();
	_cameraInfos.resize(_maxNumCams);
	for (size_t i = 0; i < _maxNumCams; ++i) {
		const auto & cam = *cameras[i];
		_cameraInfos[i].vp = cam.viewproj();
		_cameraInfos[i].pos = cam.position();
		_cameraInfos[i].dir = cam.dir();
		_cameraInfos[i].selected = cam.isActive();
	}

	// Compute the max number of cameras allowed.
	GLint maxBlockSize = 0, maxSlicesSize = 0;
	glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &maxBlockSize);
	glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxSlicesSize);
	// For each camera we store a matrix, 2 vecs3, 2 floats (including padding).
	const unsigned int bytesPerCamera = 4 * (16 + 2 * 3 + 2);
	const unsigned int maxCamerasAllowed = std::min((unsigned int)maxSlicesSize, (unsigned int)(maxBlockSize / bytesPerCamera));
	std::cout << "[ULRV3Renderer] " << "MAX_UNIFORM_BLOCK_SIZE: " << maxBlockSize << ", MAX_ARRAY_TEXTURE_LAYERS: " << maxSlicesSize << ", meaning at most " << maxCamerasAllowed << " cameras." << std::endl;

	// Create UBO.
	_uboIndex = 0;
	glGenBuffers(1, &_uboIndex);
	glBindBuffer(GL_UNIFORM_BUFFER, _uboIndex);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(CameraUBOInfos)*_maxNumCams, &_cameraInfos[0], GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	// Setup shaders and uniforms.
	setupShaders(fragString, vertexString);

	// Create the intermediate rendertarget.
	_depthRT.reset(new sibr::RenderTargetRGBA32F(w, h));

	CHECK_GL_ERROR;
}


void sibr::ULRV3Renderer::setupShaders(const std::string & fShader, const std::string & vShader)
{
	// Create shaders.
	std::cout << "[ULRV3Renderer] Setting up shaders for " << _maxNumCams << " cameras." << std::endl;
	GLShader::Define::List defines;
	defines.emplace_back("NUM_CAMS", _maxNumCams);
	defines.emplace_back("ULR_STREAMING", 0);

	_ulrShader.init("ULRV3",
		sibr::loadFile(sibr::getShadersDirectory("") + "/" + vShader + ".vert"),
		sibr::loadFile(sibr::getShadersDirectory("") + "/" + fShader + ".frag", defines));
	_depthShader.init("ULRV3Depth",
		sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr_intersect.vert"),
		sibr::loadFile(sibr::getShadersDirectory("ulr") + "/ulr_intersect.frag", defines));

	// Setup uniforms.
	_nCamProj.init(_depthShader, "proj");
	_nCamPos.init(_ulrShader, "ncam_pos");
	_occTest.init(_ulrShader, "occ_test");
	_useMasks.init(_ulrShader, "doMasking");
	_discardBlackPixels.init(_ulrShader, "discard_black_pixels");
	_epsilonOcclusion.init(_ulrShader, "epsilonOcclusion");
	_areMasksBinary.init(_ulrShader, "is_binary_mask");
	_invertMasks.init(_ulrShader, "invert_mask");
	_flipRGBs.init(_ulrShader, "flipRGBs");
	_showWeights.init(_ulrShader, "showWeights");
	_winnerTakesAll.init(_ulrShader, "winner_takes_all");
	_camsCount.init(_ulrShader, "camsCount");
	_gammaCorrection.init(_ulrShader, "gammaCorrection");

	CHECK_GL_ERROR;
}

void sibr::ULRV3Renderer::process(
	const sibr::Mesh & mesh,
	const sibr::Camera & eye,
	IRenderTarget & dst,
	const sibr::Texture2DArrayRGB::Ptr & inputRGBs,
	const sibr::Texture2DArrayLum32F::Ptr & inputDepths,
	bool passthroughDepth
) {
	// Render the proxy positions in world space.
	process(mesh, eye, dst, inputRGBs->handle(), inputDepths, passthroughDepth);
}

void sibr::ULRV3Renderer::process(
	const sibr::Mesh & mesh,
	const sibr::Camera & eye,
	IRenderTarget & dst,
	uint inputRGBHandle,
	const sibr::Texture2DArrayLum32F::Ptr & inputDepths,
	bool passthroughDepth
) {
	if (_profiling) {
		_depthPassTimer.tic();
	}
	// Render the proxy positions in world space.
	renderProxyDepth(mesh, eye);
	if (_profiling) {
		glFinish();
		//std::cout << "\nDepth Pass: " << _depthPassTimer.deltaTimeFromLastTic() << " ms" << std::endl;
		_depthCost.push_back(_depthPassTimer.deltaTimeFromLastTic());
	}
	if (_profiling) {
		_blendPassTimer.tic();
	}
	// Perform ULR blending.
	renderBlending(eye, dst, inputRGBHandle, inputDepths, passthroughDepth);
	if (_profiling) {
		glFinish();
		//std::cout << "\nBlend Pass: " << _blendPassTimer.deltaTimeFromLastTic() << " ms" << std::endl;
		_blendCost.push_back(_blendPassTimer.deltaTimeFromLastTic());
	}
}

void sibr::ULRV3Renderer::updateCameras(const std::vector<uint> & camIds) {
	// Reset all cameras.
	for(auto & caminfos : _cameraInfos) {
		caminfos.selected = 0;
	}
	// Enabled the ones passed as indices.
	for (const auto & camId : camIds) {
		_cameraInfos[camId].selected = 1;
	}

	// Update the content of the UBO.
	glBindBuffer(GL_UNIFORM_BUFFER, _uboIndex);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(CameraUBOInfos)*_maxNumCams, &_cameraInfos[0]);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void sibr::ULRV3Renderer::stopProfile()
{
	const std::vector<std::string> names = { "Depth Cost: ", "Blend Cost: "};
	const  std::vector<std::vector<float>> counts = {
		_depthCost, _blendCost};
	std::string profileStr = "";
	//profileStr = "";

	for (int i = 0; i < names.size(); ++i) {
		// Compute metrics: min, max, avg, variance.
		double miniF = std::numeric_limits<double>::max();
		double maxiF = 0.0;
		double avgF = 0.0;
		for (size_t tid = 0; tid < counts[i].size(); ++tid) {
			const double ft = double(counts[i][tid]);
			avgF += ft;
			miniF = std::min(miniF, ft);
			maxiF = std::max(maxiF, ft);
		}
		avgF /= double(counts[i].size());
		double varF = 0.0;
		for (size_t tid = 0; tid < counts[i].size(); ++tid) {
			const double residualF = double(counts[i][tid]) - avgF;
			varF += residualF * residualF;
		}
		varF /= double(int(counts[i].size()) - 1);
		profileStr += "-----------\n";
		profileStr += names[i] + " num frames: " + std::to_string(counts[i].size()) + "\n";
		profileStr += names[i] + " min/max: " + std::to_string(miniF) + "/" + std::to_string(maxiF) + "\n";
		profileStr += names[i] + " avg/stddev: " + std::to_string(avgF) + "/" + std::to_string(std::sqrt(varF)) + "\n";
	}

	std::cout << profileStr << std::endl;
}

void sibr::ULRV3Renderer::renderProxyDepth(const sibr::Mesh & mesh, const sibr::Camera & eye)
{
	// Bind and clear RT.
	_depthRT->bind();
	glViewport(0, 0, _depthRT->w(), _depthRT->h());
	glClearColor(0, 0, 0, 1);
	glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Render the mesh from the current viewpoint, output positions.
	_depthShader.begin();
	_nCamProj.set(eye.viewproj());

	mesh.render(true, _backFaceCulling);
	
	_depthShader.end();
	_depthRT->unbind();
}

void sibr::ULRV3Renderer::renderBlending(
	const sibr::Camera & eye,
	IRenderTarget & dst,
	uint inputRGBHandle,
	const sibr::Texture2DArrayLum32F::Ptr & inputDepths,
	bool passthroughDepth
) {
	// Bind and clear destination rendertarget.
	glViewport(0, 0, dst.w(), dst.h());
	if (_clearDst) {
		dst.clear();
	}
	dst.bind();

	_ulrShader.begin();

	// Uniform values.
	_nCamPos.set(eye.position());
	_occTest.send();
	_areMasksBinary.send();
	_invertMasks.send();
	_discardBlackPixels.send();
	_useMasks.send();
	_epsilonOcclusion.send();
	_flipRGBs.send();
	_showWeights.send();
	_camsCount.send();
	_winnerTakesAll.send();
	_gammaCorrection.send();

	// Textures.
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _depthRT->handle());

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D_ARRAY, inputRGBHandle);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D_ARRAY, inputDepths->handle());

	// Pass the masks if enabled and available.
	if (_useMasks && _masks.get()) {
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D_ARRAY, _masks->handle());
	}

	// Bind UBO to shader, after all possible textures.
	glBindBuffer(GL_UNIFORM_BUFFER, _uboIndex);
	glBindBufferBase(GL_UNIFORM_BUFFER, 4, _uboIndex);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	if (passthroughDepth) {
		glEnable(GL_DEPTH_TEST);
	} else {
		glDisable(GL_DEPTH_TEST);
	}

	// Perform ULR rendering.
	RenderUtility::renderScreenQuad();
	glDisable(GL_DEPTH_TEST);

	_ulrShader.end();
	dst.unbind();
}

void sibr::ULRV3Renderer::resize(const unsigned w, const unsigned h) {
	_depthRT.reset(new sibr::RenderTargetRGBA32F(w, h));
}
