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
# include <core/graphics/Mesh.hpp>
# include <core/graphics/Texture.hpp>
# include <core/graphics/Camera.hpp>

# include <core/renderer/Config.hpp>

namespace sibr { 

	/** Render a Point Cloud with colors
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT PointBasedRenderer
	{
	public:
		typedef std::shared_ptr<PointBasedRenderer>	Ptr;

	public:

		/** Constructor.
		*/
		PointBasedRenderer();

		void	meshToDevice(const Mesh& mesh);

		/** Render the textured mesh.
		\param mesh the mesh to render (should have UV attribute)
		\param eye the viewpoint to use
		\param dst destination rendertarget
		\param backfaceCull should backface culling be performed
		*/
		void	process(const Mesh& mesh, const Camera& eye, IRenderTarget& dst, bool backfaceCull = true);

		/** Render the textured mesh.
		\param mesh the mesh to render (should have UV attribute)
		\param eye the viewpoint to use
		\param model additional transformation matrix
		\param dst destination rendertarget
		\param backfaceCull should backface culling be performed
		*/
		void process(const Mesh & mesh, const Camera & eye, const sibr::Matrix4f & model, IRenderTarget & dst, bool backfaceCull = true);

	protected:

		GLShader			_shader; ///< The point based shader.
		GLuniform<Matrix4f> 	_paramMVP; ///< MVP uniform.
		GLuniform<float> 	_paramAlpha; ///< Alpha uniform.
		GLuniform<int>		_paramRadius; ///< Radius uniform.
		GLuniform<Vector3f>	_paramUserColor;
	};

} /*namespace sibr*/ 
