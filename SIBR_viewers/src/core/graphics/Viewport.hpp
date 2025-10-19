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

# include "core/graphics/Config.hpp"
# include "core/system/Vector.hpp"

namespace sibr
{

	/** Represent an on-screen viewport using normalized coordinates, which can be nested into another viewport.
	* \ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT Viewport
	{
	public:

		/** Default constructor: unit viewport. */
		Viewport( void ):
			_parent(nullptr),
			_left(0.f), _top(0.f), _right(1.f), _bottom(1.f) { }

		/** Constructor from extents.
		 *\param left left extent
		 *\param top top extent
		 *\param right right extent
		 *\param bottom bottom extent
		 */
		Viewport( float left, float top, float right, float bottom ) :
			_parent(nullptr),
			_left(left), _top(top), _right(right), _bottom(bottom) { }

		/** Constructor from a parent and relative extents.
		 *\param parent_ the parent viewport
		 *\param left left relative extent
		 *\param top top relative extent
		 *\param right right relative extent
		 *\param bottom bottom relative extent
		 */
		Viewport( const Viewport* parent_, float left, float top, float right, float bottom ) :
			_left(left), _top(top), _right(right), _bottom(bottom) { parent(parent_); }

		/** Constructor from a parent and relative extents.
		 *\param parent_ the parent viewport
		 *\param left left relative extent
		 *\param top top relative extent
		 *\param right right relative extent
		 *\param bottom bottom relative extent
		 */
		Viewport(const Viewport & parent_, float left, float top, float right, float bottom) :
			Viewport(&parent_, left, top, right, bottom) {
			*this = Viewport(finalLeft(), finalTop(), finalRight(), finalBottom());
		}

		/** \return the relative left extent. */
		inline float	left( void ) const { return _left; }
		/** \return the relative top extent. */
		inline float	top( void ) const { return _top; }
		/** \return the relative right extent. */
		inline float	right( void ) const { return _right; }
		/** \return the relative bottom extent. */
		inline float	bottom( void ) const { return _bottom; }

		/** \return the relative viewport width */
		inline float	width( void ) const { return _right-_left; }
		/** \return the relative viewport height */
		inline float	height( void ) const { return _bottom-_top; }

		/** \return the absolute left extent. */
		float	finalLeft( void ) const;
		/** \return the absolute top extent. */
		float	finalTop( void ) const;
		/** \return the absolute right extent. */
		float	finalRight( void ) const;
		/** \return the absolute bottom extent. */
		float	finalBottom( void ) const;

		/** \return the absolute viewport width. */
		float	finalWidth( void ) const;
		/** \return the absolute viewport height. */
		float	finalHeight( void ) const;

		/** \return the absolute viewport size. */
		sibr::Vector2f finalSize() const;
		/** \return the absolute cooridnates of the top left corner. */
		Vector2f finalTopLeft() const;


		/** Compute the absolute pixel coordinates based on relative normalized coordinates.
		 *\param uv the normalized UVs 
		 *\return the pixel coordinates 
		 */
		Vector2f pixAt(const Vector2f & uv) const;

		/** Check if a point is inside the viewport.
		 *\param x horizontal coordinate
		 *\param y vertical coordinate
		 *\return true if the point is inside
		 */
		bool	contains( float x, float y ) const;

		/** Check if a point is inside the viewport.
		 *\param x horizontal coordinate
		 *\param y vertical coordinate
		 *\return true if the point is inside
		 */
		bool	contains( int x, int y ) const;

		/** Check if a point is inside the viewport.
		 *\param xy coordinates
		 *\return true if the point is inside
		 */
		bool	contains(const Vector2f & xy) const;

		/** Bind an OpenGL viewport whose values are determined based on the viewport final dimensions and the target size.
		 *\param screenWidth the width of the rendertarget
		 *\param screenHeight the height of the rendertarget
		 */
		void			bind( uint screenWidth, uint screenHeight ) const;

		/** Bind an OpenGL viewport  whose values are determined based on the viewport final dimensions. */
		void			bind( void ) const;

		/** Perform a full OpenGL clear of the region defined by the viewport in the currently bound target.
		 *\param bgColor clear color
		 */
		void			clear( const Vector3f& bgColor=Vector3f(0.f, 0.f, 0.f) ) const;

		/** Set the viewport parent
		 *\param view the new parent
		 */
		void				parent( const Viewport* view );

		/** \return the parent viewport if it exists or nullptr. */
		const Viewport*		parent( void ) const;

		/** \return true if the viewport is empty (0x0). */
		bool isEmpty() const;

	private:
		const Viewport*	_parent; ///< (optional)

		float	_left; ///< Left extent.
		float	_top; ///< Top extent.
		float	_right; ///< Right extent.
		float	_bottom; ///< Bottom extent.

	};

	///// DEFINITIONS /////

	inline void				Viewport::parent( const Viewport* view ) { 
		_parent = view; 

		//if (_parent == this) // means 'is the root'
		//	_parent = nullptr;
	}
	inline const Viewport*		Viewport::parent( void ) const { 
		return _parent; 
	}

	inline float	Viewport::finalLeft( void ) const {
		return (_parent)? (_parent->finalLeft() + _parent->finalWidth()*left()) : left();
	}

	inline float	Viewport::finalTop( void ) const {
		return (_parent)? ( _parent->finalTop() + _parent->finalHeight()*top() ) : top();
	}

	inline float	Viewport::finalRight( void ) const {
		return (_parent)? (_parent->finalLeft() + _parent->finalWidth()*right()) : right();
	}

	inline float	Viewport::finalBottom( void ) const {
		return (_parent)? (_parent->finalTop() + _parent->finalHeight()*bottom()) : bottom();
	}

	inline float	Viewport::finalWidth( void ) const {
		return (_parent)? _parent->finalWidth()*width() : width();
	}

	inline float	Viewport::finalHeight( void ) const {
		return (_parent)? _parent->finalHeight()*height() : height();
	}

	inline sibr::Vector2f	Viewport::finalSize(void) const {
		return sibr::Vector2f(finalWidth(),finalHeight());
	}

	inline Vector2f Viewport::finalTopLeft() const {
		return { finalLeft(), finalTop() };
	}


} // namespace sibr
