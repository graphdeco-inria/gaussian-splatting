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
# include <core/graphics/Mesh.hpp>
# include <core/graphics/Camera.hpp>

# include <core/renderer/Config.hpp>

namespace sibr { 

	/** Render a mesh colored using the per-vertex color attribute.
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT ColoredMeshRenderer
	{
	public:
		typedef std::shared_ptr<ColoredMeshRenderer>	Ptr;

	public:

		/// Constructor.
		ColoredMeshRenderer( void );

		/** Render the mesh using its vertices colors, interpolated over triangles.
		\param mesh the mesh to render
		\param eye the viewpoint to use
		\param dst the destination rendertarget
		\param mode the rendering mode of the mesh
		\param backFaceCulling should backface culling be performed
		*/
		void	process(
			/*input*/	const Mesh& mesh,
			/*input*/	const Camera& eye,
			/*output*/	IRenderTarget& dst,
			/*mode*/    sibr::Mesh::RenderMode mode = sibr::Mesh::FillRenderMode,
			/*BFC*/     bool backFaceCulling = true);

	private:

		GLShader			_shader; ///< Color shader.
		GLParameter			_paramMVP; ///< MVP uniform.
		
	};

} /*namespace sibr*/ 
