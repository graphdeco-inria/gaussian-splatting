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

#include "Config.hpp"

# include <core/graphics/Shader.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/graphics/Texture.hpp>
# include <core/graphics/Camera.hpp>

# include <core/renderer/Config.hpp>


namespace sibr
{
	/** Render the world space positions of a mesh surface.
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT PositionRenderer
	{
		SIBR_CLASS_PTR(PositionRenderer);

	public:

		/** Constructor with a target size.
		\param w the target width
		\param h the target height
		*/
		PositionRenderer(int w,int h);

		/** Render the mesh world positions.
		\param cam the viewpoint to use
		\param mesh the mesh to render
		\param backFaceCulling should backface culling be performed
		\param frontFaceCulling flip the culling test orientation
		*/
		void render( const sibr::Camera &cam, const Mesh& mesh, bool backFaceCulling=false, bool frontFaceCulling=false);

		/** \return the result rendertarget containing world space positions. */
		const sibr::RenderTargetRGB32F::Ptr & getPositionsRT() { return _RT; }

	private:

		sibr::GLShader							_shader; ///< The positions shader.
		sibr::GLuniform<sibr::Matrix4f>			_MVP; ///< MVP uniform.
		sibr::RenderTargetRGB32F::Ptr			_RT; ///< Destination render target.
		
	};

} // namespace

