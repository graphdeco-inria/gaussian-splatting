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

# include <core/graphics/Window.hpp>
# include <core/graphics/Shader.hpp>
# include <core/graphics/Texture.hpp>

# include <core/renderer/Config.hpp>

namespace sibr { 

	/** Copy the content of an input texture to another rendertarget or to the window.
	If you need a basic copy, prefer using blit.
	\sa sibr::blit
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT CopyRenderer
	{
	public:
		typedef std::shared_ptr<CopyRenderer>	Ptr;

	public:

		/** Constructor. You can specify custom shaders, refer to noproj.vert and copy.frag for examples.
		\param vertFile pah to the vertex shader file
		\param fragFile pah to the fragment shader file
		*/
		CopyRenderer(
			const std::string& vertFile = sibr::getShadersDirectory("core") + "/noproj.vert",
			const std::string& fragFile = sibr::getShadersDirectory("core") + "/copy.frag"
		);

		/** Copy input texture to the output texture, copy also the input alpha into depth.
		\param textureID the texture to copy
		\param dst the destination
		\param disableTest disable depth testing (depth won't be written)
		*/
		void	process( uint textureID, IRenderTarget& dst,
			bool disableTest=true);

		/** Copy input texture to a window.
		\param textureID the texture to copy
		\param dst the destination window
		*/
		void	copyToWindow( uint textureID, Window& dst);

		/** \return option to flip the texture when copying. */
		bool & flip() { return _flip.get(); }

	private:
		
		GLShader			_shader; ///< Copy shader.
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
	};

} /*namespace sibr*/ 
