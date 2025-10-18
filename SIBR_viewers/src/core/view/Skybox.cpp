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



# include "core/assets/Resources.hpp"
# include "core/view/Skybox.hpp"

namespace sibr
{
	bool	Skybox::load(const std::string& skyFolder)
	{
		if (!sibr::directoryExists(skyFolder))
			return false;

		_shader.init("Skybox",
			sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("skybox.vp")),
			sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("skybox.fp")));
		_paramView.init(_shader, "in_View");
		_paramAspect.init(_shader, "in_Aspect");

		std::array<const char*, 6> filenames = {
			"right.jpg"		,
			"left.jpg"		,
			"top.jpg"		,
			"bottom.jpg"	,
			"forward.jpg"	,
			"back.jpg"		
		};

		std::array<ImageRGB, filenames.size()>	images;

		for (uint i = 0; i < filenames.size(); ++i)
		{
			std::string file = (skyFolder + "/") + filenames[i];
			if (images[i].load(file) == false)
			{
				SIBR_ERR << "cannot open " << file << " (loading the skybox)" << std::endl;
			}
		}
		_cubemap.reset(new TextureCubeMapRGB(images[0], images[1], images[2], images[3], images[4], images[5]));

		return true;
	}


	void	Skybox::render(const Camera& eye, const sibr::Vector2u& imgSize)
	{
		if (_cubemap == nullptr)
			return;


		glDisable(GL_DEPTH_TEST);

		CHECK_GL_ERROR;
		_shader.begin();
		CHECK_GL_ERROR;
		_paramAspect.set(Vector2f(float(imgSize.x())/float(imgSize.y()), float(imgSize.y())/float(imgSize.x())));
		CHECK_GL_ERROR;
		_paramView.set(Matrix4f(eye.view().inverse()));
		CHECK_GL_ERROR;
		// cube map texture should already be bound
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, _cubemap->handle());
		CHECK_GL_ERROR;

		RenderUtility::useDefaultVAO();
		const unsigned char indices[] = { 0, 1, 2, 3 };
		glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_BYTE, indices);
		CHECK_GL_ERROR;

		_shader.end();
	}
} // namespace sibr
