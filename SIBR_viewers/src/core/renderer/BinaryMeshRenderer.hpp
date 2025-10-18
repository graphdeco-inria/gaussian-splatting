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

	/** Render a binary mask of a mesh, with options to limit Z-fighting.
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT BinaryMeshRenderer
	{
		SIBR_CLASS_PTR(BinaryMeshRenderer);

	public:

		/// Constructor.
		BinaryMeshRenderer();

		/** Render the mesh mask. 
		Regions covered by the mesh will be filled with (1,1,1,1).
		\param mesh the mesh to render
		\param eye the viewpoint to use
		\param dst the destination rendertarget
		*/
		void	process( const Mesh& mesh, const Camera& eye, IRenderTarget& dst );

		/** Shift that can be used to modify the depth written, 
		to avoid Z-fighting when rendering multiple masks of 
		the same mesh or combining masks.
		If set to 0.0: no shift, if set to 1.0: all vertices sent to depth 0.0.
		\return a reference to the shift
		*/
		float & getEpsilon() {
			return epsilon.get();
		}

	private:

		GLShader					_shader; ///< Mask shader.
		GLuniform<sibr::Matrix4f>	_paramMVP; ///< MVP uniform.
		GLuniform<float>			epsilon = 0; ///< Epsilon uniform.
		
	};

} /*namespace sibr*/ 
