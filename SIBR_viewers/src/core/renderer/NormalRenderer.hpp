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

# include <core/renderer/Config.hpp>

# include "core/assets/InputCamera.hpp"
# include "core/graphics/Texture.hpp"
# include "core/graphics/Camera.hpp"
# include "core/graphics/RenderUtility.hpp"
# include "core/assets/Resources.hpp"
# include "core/graphics/Shader.hpp"
# include "core/graphics/Mesh.hpp"


namespace sibr
{

	/** Render the world or view space normals of a mesh.
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT NormalRenderer
	{

	public:
		
		/** Constructor.
		\param w target width
		\param h target height
		\param generate if true, use a geoemtry shader to compute normals on the fly, else use vertex normals
		\param useFloats if true, render in a 32F rendertarget, else use 8U
		\param imSpace if true, render view space normals, else render world space normals
		*/
		NormalRenderer(int w,int h, bool generate = true, bool useFloats = false, bool imSpace = false) ;
		
		/// Destructor.
		~NormalRenderer();

		/** Render the mesh normals in the internal render target.
		\param cam the viewpoint to use
		\param mesh the mesh to render
		\param modelMat additional model matrix
		\param clear clear the rendertarget before rendering
		*/
		void render( const sibr::InputCamera &cam, const Mesh& mesh, const Matrix4f &modelMat = Matrix4f::Identity(), bool clear=true);
		
		/** Resize the internal rendertarget.
		\param w the new width
		\param h the new height
		*/
		void setWH(int w, int h);
		
		std::shared_ptr<sibr::RenderTargetRGB> _normal_RT; ///< The low-precision normal result rendertarget (used if useFloats is false).
		std::shared_ptr<sibr::RenderTargetRGBA32F> _normal_RT_32F; ///< The high-precision normal result rendertarget (used if useFloats is true).

	private:

		sibr::GLShader				_normalShader; ///< Normal shader.
		sibr::GLParameter			_normalShader_proj; ///< Projection matrix uniform.
		sibr::GLParameter			_normalShader_view; ///< View matrix uniform.
		sibr::GLParameter			_normalShader_model; ///< Model matrix uniform.
		sibr::GLParameter			_normalShader_projInv; ///< Inverse projection matrix uniform.
		sibr::GLParameter			_normalShader_imSpace; ///< View space toggle uniform.
		bool _generate; ///< Should normals be generated on the fly.
		bool _useFloats; ///< Should the normals be rendered to a 32F precision target.
		
	};


} // namespace

