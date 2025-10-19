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



#include "core/graphics/Window.hpp"
#include "core/graphics/Viewport.hpp"

namespace sibr
{
	void			Viewport::bind( uint screenWidth, uint screenHeight ) const
	{
		glViewport(
			(GLint)(left()*screenWidth), (GLint)(top()*screenHeight),
			(GLsizei)(width()*screenWidth), (GLsizei)(height()*screenHeight));
	}

	void			Viewport::clear( const Vector3f& bgColor ) const
	{
		//if (width() < 1.f)
		//	return;

		GLint l = (GLint)finalLeft();
		GLint t = (GLint)finalTop();
		GLsizei w = (GLsizei)finalWidth();
		GLsizei h = (GLsizei)finalHeight();

		glViewport(l, t, w, h);
		glScissor(l, t, w, h);
		glEnable(GL_SCISSOR_TEST);
		glClearColor(bgColor[0], bgColor[1], bgColor[2], 0.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDisable(GL_SCISSOR_TEST);
		
		glViewport(l, t, w, h);
	}

	void			Viewport::bind( void ) const
	{
		//assert((_parent != nullptr || width() > 1.f) 
		//	&& "Too small viewport detected (Set a parent viewport from a window using Viewport::parent(...) and Window::viewport()");

		glViewport(
			(GLint)(finalLeft()), (GLint)(finalTop()),
			(GLsizei)(finalWidth()), (GLsizei)(finalHeight()));
	}
	
	bool	Viewport::contains( float x, float y ) const
	{
		return (x > finalLeft() && x < finalRight() && y > finalTop() && y < finalBottom());
	}
	
	bool	Viewport::contains( int x, int y ) const
	{
		return (x > (int)finalLeft() && x < (int)finalRight() && y > (int)finalTop() && y < (int)finalBottom());
	}

	bool Viewport::contains(const Vector2f & xy) const
	{
		return contains(xy.x(), xy.y());
	}

	bool Viewport::isEmpty() const {
		return width() == 0.0 && height() == 0.0;
	}

	Vector2f Viewport::pixAt(const Vector2f & uv) const {
		return uv.cwiseProduct(finalSize()) + finalTopLeft();
	}

} // namespace sibr
