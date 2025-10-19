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

# include "Config.hpp"

# include <iostream>
# include <vector>
# include <memory>

# include <core/graphics/Config.hpp>
# include <core/graphics/Texture.hpp>
# include "core/graphics/Shader.hpp"

namespace sibr { 

	/**
	* Hole filling by poisson synthesis on an input texture;
	* contains all shaders, render targets and render passes.
	* All black pixels on the input texture are considered holes
	* and Poisson synthesis affects these pixels only, all
	* other pixels are treated at Dirichlet boundary conditions.
	* \ingroup sibr_renderer
	*/
	class SIBR_EXP_RENDERER_EXPORT PoissonRenderer
	{
	public:
		typedef std::shared_ptr<PoissonRenderer>	Ptr;

	public:

		/**
		* Initialize Poisson solvers render targets and shaders.
		* \param w width of highest resolution multigrid level
		* \param h height of highest resolution multigrid level
		*/
		PoissonRenderer ( uint w, uint h );

		/** Perform poisson filling.
		\param src source rendertarget, black pixels will be filled
		\param dst destination rendertarget
		*/
		void	process(
			/*input*/	const RenderTargetRGBA::Ptr& src,
			/*ouput*/	RenderTargetRGBA::Ptr& dst );

		/** Perform poisson filling.
		\param texID source texture handle, black pixels will be filled
		\param dst destination rendertarget
		*/
		void	process(
			/*input*/	uint texID,
			/*ouput*/	RenderTargetRGBA::Ptr& dst );

		/**
		* \return the size used for in/out textures (defined in ctor)
		*/
		const Vector2i&		getSize( void ) const;

		/** If true, fix a bug caused by erroneous viewport when initializing the internal pyramid.
			Left exposed for retrocompatibility reasons.
		\return a reference to the bugfix toggle. */
		bool & enableFix() { return _enableFix; }

	private:
		/**
		* Render the full Poisson synthesis on the holes in texture 'tex'.
		* \param tex OpenGL texture handle of input texture
		* \returns OpenGL texture handle of texture containing Poisson synthesis solution
		*/
		uint render( uint tex );

		/** Size defined in the ctor */
		Vector2i		_size;

		/** Shader to perform Jacobi relaxations */
		sibr::GLShader	_jacobiShader;

		/** Shader to downsample input texture and boundary conditions from
		* higher multigrid level to next lower level */
		sibr::GLShader	_restrictShader;

		/** Shader to interpolate Poisson synthesis solution from
		* lower multigrid level to next higher level */
		sibr::GLShader	_interpShader;

		/** Shader to compute divergence (second derivative) field of input texture */
		sibr::GLShader	_divergShader;

		/** Render target to store Poisson synthesis result */
		RenderTargetRGBA::Ptr  _poisson_RT;

		/** Helper render target for \p _poisson_RT to
		* perform ping-pong render passes during Jacobi relaxations */
		RenderTargetRGBA::Ptr  _poisson_tmp_RT;

		/** Dirichlet constraints for each multigrid level */
		std::vector<RenderTargetRGBA::Ptr> _poisson_div_RT;

		/** Jacobi step parameters. */
		sibr::GLParameter _jacobi_weights, _jacobi_scale, _restrict_scale;
		/** Interpolation scale. */
		sibr::GLParameter _interp_scale;

		/** Enable the "weird large regions of color" bugfix. */
		bool _enableFix = true;
	};

} /*namespace sibr*/ 
