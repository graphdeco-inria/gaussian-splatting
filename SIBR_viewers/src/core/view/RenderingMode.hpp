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

# include "core/graphics/Camera.hpp"
# include "core/graphics/Viewport.hpp"
# include "core/graphics/Texture.hpp"
# include "core/view/Config.hpp"
# include "core/view/ViewBase.hpp"
# include "core/graphics/Image.hpp"
# include "core/graphics/Shader.hpp"
# include "core/assets/InputCamera.hpp"

namespace sibr
{
	/**
	*	Rendering mode manages the rendertarget and camera fed to an IBR view. Can be used to render a view using a stereoscopic mode (anaglyph or VR).
	*   \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT IRenderingMode
	{
		SIBR_CLASS_PTR(IRenderingMode);

	public:
		typedef RenderTargetRGB RenderTarget;
	public:
		/// Destructor.
		virtual ~IRenderingMode( void ) { }

		/** Perform rendering of a view.
		 *\param view the view to render
		 *\param eye the current camera
		 *\param viewport the current viewport
		 *\param optDest an optional destination RT 
		 */
		virtual void	render( 
			ViewBase& view, const sibr::Camera& eye, const sibr::Viewport& viewport, 
			IRenderTarget* optDest = nullptr) = 0;

		/** Get the current rendered image as a CPU image
		 *\param current_img will contain the content of the RT */
		virtual void destRT2img( sibr::ImageRGB& current_img ) = 0;

	protected:
		std::unique_ptr<RenderTargetRGB>	_prevL, _prevR; ///< prev RT to link renderers across different views in multipass

	public:
		bool _clear; ///< Should the dst RT be cleared before rendering.

		/** Set common previous step RT.
		 *\param p the RT
		 */
		void	setPrev(const std::unique_ptr<RenderTargetRGB>& p) { std::cerr<<"ERROR " << std::endl; }
		/** Set left and right previous step RTs.
		 *\param pl the left eye RT
		 *\param pr the right eye RT
		 */
		void	setPrevLR(const std::unique_ptr<RenderTargetRGB>& pl, const std::unique_ptr<RenderTargetRGB>& pr) { std::cerr<<"ERROR " << std::endl;}

		/** \return the left eye (or common) RT. */
		virtual const std::unique_ptr<RenderTargetRGB>&	lRT() = 0;
		/** \return the right eye (or common) RT. */
		virtual const std::unique_ptr<RenderTargetRGB>&	rRT() = 0;

	};

	/** Default rendering mode: monoview, passthrough.
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT MonoRdrMode : public IRenderingMode
	{
	public:

		/// Constructor.
		MonoRdrMode( void );

		/** Perform rendering of a view.
		 *\param view the view to render
		 *\param eye the current camera
		 *\param viewport the current viewport
		 *\param optDest an optional destination RT
		 */
		void	render( ViewBase& view, const sibr::Camera& eye, const sibr::Viewport& viewport, IRenderTarget* optDest = nullptr);

		/** Get the current rendered image as a CPU image
		 *\param current_img will contain the content of the RT */
		void destRT2img( sibr::ImageRGB& current_img )
		{
			_destRT->readBack(current_img);
			return;
		}

		/** \return the common RT. */
		virtual const std::unique_ptr<RenderTargetRGB>&	lRT() { return _destRT; }
		/** \return the common RT. */
		virtual const std::unique_ptr<RenderTargetRGB>&	rRT() { return _destRT; }

	private:
		sibr::GLShader							_quadShader; ///< Passthrough shader.
		std::unique_ptr<RenderTarget>		_destRT; ///< Common destination RT.
	};

	/**
	 *Stereo rendering mode: two slightly shifted views are rendered and composited as anaglyphs.
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT StereoAnaglyphRdrMode : public IRenderingMode
	{
	public:

		/// Constructor.
		StereoAnaglyphRdrMode( void );

		/** Perform rendering of a view.
		 *\param view the view to render
		 *\param eye the current camera
		 *\param viewport the current viewport
		 *\param optDest an optional destination RT
		 */
		void	render( ViewBase& view, const sibr::Camera& eye, const sibr::Viewport& viewport, IRenderTarget* optDest = nullptr);

		/** Set the focal distance.
		\param focal focal distance
		*/
		void	setFocalDist(float focal) { _focalDist = focal; }

		/** Set the distance between the two eyes.
		\param iod intra-ocular distance
		*/
		void	setEyeDist(float iod) { _eyeDist = iod; }

		/** \return the focal distance */
		float	focalDist()	{ return _focalDist; }
		/** \return the intra-ocular distance */
		float	eyeDist()	{ return _eyeDist; }

		/** Get the current rendered image as a CPU image (empty).
		 *\param current_img will contain the content of the RT */
		void destRT2img( sibr::ImageRGB& current_img ){};

		/** \return the left eye RT. */
		virtual const std::unique_ptr<RenderTargetRGB>&	lRT() { return _leftRT; }
		/** \return the right eye RT. */
		virtual const std::unique_ptr<RenderTargetRGB>&	rRT() { return _rightRT; }

	private:
		sibr::GLShader		_stereoShader; ///< Anaglyph shader.
		RenderTarget::UPtr	_leftRT, _rightRT; ///< Each eye RT.
		float				_focalDist, _eyeDist; ///< Focal and inter-eyes distances.
	};

	///// DEFINITIONS /////

} // namespace sibr
