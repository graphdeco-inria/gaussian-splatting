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

# include <core/graphics/Shader.hpp>
# include <core/graphics/Texture.hpp>

# include <core/renderer/Config.hpp>

namespace sibr { 

	/** Blur on color edges present in a texture.
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT BlurRenderer
	{
	public:
		typedef std::shared_ptr<BlurRenderer>	Ptr;

	public:

		/// Constructor.
		BlurRenderer( void );

		/** Process the texture.
		\param textureID the texture to blur
		\param textureSize the texture dimensions
		\param dst the destination rendertarget
		*/
		void	process(
			/*input*/	uint textureID,
			/*input*/	const Vector2f& textureSize,
			/*output*/	IRenderTarget& dst );

	private:

		GLShader			_shader; ///< Blur shader.
		GLParameter			_paramImgSize; ///< Texture size uniform.

	};

} /*namespace sibr*/ 
