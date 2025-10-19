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

# include <type_traits>

# include "core/graphics/Texture.hpp"
# include "core/graphics/Camera.hpp"
# include "core/view/Config.hpp"
//# include "core/view/IBRScene.hpp"
#include "core/graphics/Input.hpp"
#include "core/graphics/Window.hpp"

namespace sibr
{

	/** Basic view representation. All views should inherit from it. 
	* Can be added as a subview in a multi-window system.
	* \sa MultiViewBase
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT ViewBase
	{
	public:

		typedef std::shared_ptr<ViewBase> Ptr;

		/** Constructor.
		 *\param w view width
		 *\param h view height
		 */
		ViewBase( const unsigned int w=720, const unsigned int h=480);

		/** Destructor. */
		virtual ~ViewBase() = default;

		/* Update state based on user input.
		 * \param input user input
		 */
		virtual void	onUpdate(Input& input) { }

		/** Render content in a window.
		 *\param win destination window
		 */
		virtual void	onRender( Window& win)		{ }

		/** Render content in a given rendertarget.
		 *\param dst destination RT
		 *\param eye current viewpoint
		 *\sa IRenderingMode
		 */
		virtual void	onRenderIBR(IRenderTarget& dst, const Camera& eye) {};

		/** Display GUI. */
		virtual void	onGUI() { }

		/** Render content in the currently bound RT, using a specific viewport.
		 * \param vpRender destination viewport
		 * \note Used when the view is in a multi-view system.
		 */
		virtual void	onRender(const Viewport & vpRender) {  }

		/** Update state based on user input.
		 * \param input user input
		 * \param vp input viewport
		 * \note Used when the view is in a multi-view system.
		 */
		virtual void	onUpdate(Input& input, const Viewport & vp);

		/** Legacy: Used to mix with previous pass.
		 * \param prev the previous step RT
		 */
		virtual void	preRender(RenderTargetRGB& prev) {} ;

		/** Legacy: Set the internal RT to use.
		 *\param i RT index */
		virtual void		whichRT(uint i)			{ _whichRT=i; }
		/** Legacy: \return the current selected RT ID. */
		virtual uint		whichRT(void)			{ return _whichRT; }

		/** Set the view resolution.
		\param size the new resolution, in pixels.
		*/
		void				setResolution(const Vector2i& size);
		/**\return the current resolution. */
		const Vector2i&		getResolution( void ) const;

		/** Toggle view status.
		\param act if true, the view is active.
		*/
		void				active(bool act) { _active = act; }
		/** \return true if the view is currently active. */
		bool				active() { return _active; }

		/** Toggle view focus (ie the user is interacting with it).
		\param focus if true, the view is currently focused 
		*/
		void				setFocus(bool focus);
		/** \return true if the view is currently focused (ie the user is interacting with it). */
		bool				isFocused(void) const;

		
		/** Define the name of the view (used for disambiguation of GUI, etc.).
		 * \param name the new name
		 */
		void				setName(const std::string & name);

		/** \return the name of the view. */
		const std::string & name() const;
		
	protected:
		uint			_whichRT; ///< Selected RT id.
		std::vector<RenderTargetLum::Ptr>	_masks; ///< Rendering masks that can beused by some views/renderers.

		bool			_active = true; ///< Is the view active.
		Vector2i		_resolution; ///< View resolution.
		bool			_focus = false; ///< Is the view focused.
		std::string		_name = ""; ///< View name.

	};

} // namespace sibr
