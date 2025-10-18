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



#include "core/graphics/Input.hpp"

namespace sibr
{

	/*static*/ Input&	Input::global( void )
	{
		/// \todo TODO: add warning if no windows have been created
		static Input	instance;
		instance._empty = false;
		return instance;
	}

	/*static*/ void		Input::poll( void )
	{
		sibr::Input::global().swapStates();
		glfwPollEvents();
	}

	Input Input::subInput(const sibr::Input & global, const sibr::Viewport & viewport, const bool mouseOutsideDisablesKeyboard)
	{
		Input sub = global;
		sub._mousePrevPos -= sibr::Vector2i(viewport.finalLeft(), viewport.finalTop());
		sub._mousePos -= sibr::Vector2i(viewport.finalLeft(), viewport.finalTop());

		if (!global.isInsideViewport(viewport)) {
			sub._mouseButton = MouseButton();
			sub._mouseScroll = 0;

			if (mouseOutsideDisablesKeyboard) {
				sub._keyboard = Keyboard();				
			} 
			return sub;		
		} 
		
		return sub;	
	}

	bool Input::isInsideViewport(const sibr::Viewport & viewport) const
	{
		Eigen::AlignedBox2i subBox;
		subBox.extend(Vector2i(viewport.finalLeft(), viewport.finalTop()));
		subBox.extend(Vector2i(viewport.finalRight(), viewport.finalBottom()));

		return subBox.contains(mousePosition());
	}

	KeyCombination::KeyCombination() : numKeys(0), isTrue(true) { }
	KeyCombination::KeyCombination(int n, bool b) : numKeys(n), isTrue(b) { }

	KeyCombination::operator bool() const
	{
		return isTrue && ( numKeys == sibr::Input::global().key().getNumActivated() );
	}	

	KeyCombination operator&& ( const KeyCombination & combA, const KeyCombination & combB)
	{
		return KeyCombination(combA.numKeys + combB.numKeys, combA.isTrue && combB.isTrue); 
	}

} // namespace sibr
