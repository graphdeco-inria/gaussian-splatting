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

# include "core/view/Config.hpp"
# include "core/graphics/Shader.hpp"
# include "core/graphics/Texture.hpp"
# include "core/graphics/Camera.hpp"

namespace sibr
{
	/** A skybox object for rendering a cubemap texture.
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT Skybox
	{
		SIBR_CLASS_PTR(Skybox);

	public:

		/** Load skybox faces from a directory. The files should be named: {right, left, top, bottom, forward, back}.jpg
		\param skyFolder directory path
		\return a success boolean 
		*/
		bool	load(const std::string& skyFolder);

		/** Render in the current RT.
		\param eye current viewpoint
		\param imgSize the destination RT size
		*/
		void	render(const Camera& eye, const sibr::Vector2u& imgSize);

	private:

		GLShader		_shader; ///< Skybox shader.
		GLParameter		_paramView; ///< VP parameter.
		GLParameter		_paramAspect; ///< Aspect ratio parameter.

		TextureCubeMapRGB::Ptr	_cubemap = nullptr; ///< Cubemap texture.

	};


} // namespace sibr
