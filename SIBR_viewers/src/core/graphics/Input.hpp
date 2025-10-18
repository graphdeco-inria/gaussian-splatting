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

# include <array>

//#define GLEW_STATIC
#include <GL/glew.h>
# define GLFW_INCLUDE_GLU
# include <GLFW/glfw3.h>

# include "core/graphics/Config.hpp"
# include "core/system/Vector.hpp"
# include "core/graphics/Viewport.hpp"

namespace sibr
{
	namespace Key
	{
		/** Key codes (based on GLFW codes). */
		enum Code
		{
			Unknown = 0 /*GLFW_KEY_UNKNOWN*/,   
			Space = GLFW_KEY_SPACE,   
			Apostrophe = GLFW_KEY_APOSTROPHE,   
			Comma = GLFW_KEY_COMMA,   
			Minus = GLFW_KEY_MINUS,   
			Period = GLFW_KEY_PERIOD,   
			Slash = GLFW_KEY_SLASH,   
			Num0 = GLFW_KEY_0,   
			Num1 = GLFW_KEY_1,   
			Num2 = GLFW_KEY_2,   
			Num3 = GLFW_KEY_3,   
			Num4 = GLFW_KEY_4,   
			Num5 = GLFW_KEY_5,   
			Num6 = GLFW_KEY_6,   
			Num7 = GLFW_KEY_7,   
			Num8 = GLFW_KEY_8,   
			Num9 = GLFW_KEY_9,   
			Semicolon = GLFW_KEY_SEMICOLON,   
			Equal = GLFW_KEY_EQUAL,   
			A = GLFW_KEY_A,   
			B = GLFW_KEY_B,   
			C = GLFW_KEY_C,   
			D = GLFW_KEY_D,   
			E = GLFW_KEY_E,   
			F = GLFW_KEY_F,   
			G = GLFW_KEY_G,   
			H = GLFW_KEY_H,   
			I = GLFW_KEY_I,   
			J = GLFW_KEY_J,   
			K = GLFW_KEY_K,   
			L = GLFW_KEY_L,   
			M = GLFW_KEY_M,   
			N = GLFW_KEY_N,   
			O = GLFW_KEY_O,   
			P = GLFW_KEY_P,   
			Q = GLFW_KEY_Q,   
			R = GLFW_KEY_R,   
			S = GLFW_KEY_S,   
			T = GLFW_KEY_T,   
			U = GLFW_KEY_U,   
			V = GLFW_KEY_V,   
			W = GLFW_KEY_W,   
			X = GLFW_KEY_X,   
			Y = GLFW_KEY_Y,   
			Z = GLFW_KEY_Z,   
			LeftBracket = GLFW_KEY_LEFT_BRACKET,   
			Backslash = GLFW_KEY_BACKSLASH,   
			RightBracket = GLFW_KEY_RIGHT_BRACKET,   
			GraveAccent = GLFW_KEY_GRAVE_ACCENT,   
			World1 = GLFW_KEY_WORLD_1,   
			World2 = GLFW_KEY_WORLD_2,   
			Escape = GLFW_KEY_ESCAPE,   
			Enter = GLFW_KEY_ENTER,   
			Tab = GLFW_KEY_TAB,   
			Backspace = GLFW_KEY_BACKSPACE,   
			Insert = GLFW_KEY_INSERT,   
			Delete = GLFW_KEY_DELETE,   
			Right = GLFW_KEY_RIGHT,   
			Left = GLFW_KEY_LEFT,   
			Down = GLFW_KEY_DOWN,   
			Up = GLFW_KEY_UP,   
			Page_up = GLFW_KEY_PAGE_UP,   
			Page_down = GLFW_KEY_PAGE_DOWN,   
			Home = GLFW_KEY_HOME,   
			End = GLFW_KEY_END,   
			CapsLock = GLFW_KEY_CAPS_LOCK,   
			ScrollLock = GLFW_KEY_SCROLL_LOCK,   
			NumLock = GLFW_KEY_NUM_LOCK,   
			PrintScreen = GLFW_KEY_PRINT_SCREEN,   
			Pause = GLFW_KEY_PAUSE,   
			F1 = GLFW_KEY_F1,   
			F2 = GLFW_KEY_F2,   
			F3 = GLFW_KEY_F3,   
			F4 = GLFW_KEY_F4,   
			F5 = GLFW_KEY_F5,   
			F6 = GLFW_KEY_F6,   
			F7 = GLFW_KEY_F7,   
			F8 = GLFW_KEY_F8,   
			F9 = GLFW_KEY_F9,   
			F10 = GLFW_KEY_F10,   
			F11 = GLFW_KEY_F11,   
			F12 = GLFW_KEY_F12,   
			F13 = GLFW_KEY_F13,   
			F14 = GLFW_KEY_F14,   
			F15 = GLFW_KEY_F15,   
			F16 = GLFW_KEY_F16,   
			F17 = GLFW_KEY_F17,   
			F18 = GLFW_KEY_F18,   
			F19 = GLFW_KEY_F19,   
			F20 = GLFW_KEY_F20,   
			F21 = GLFW_KEY_F21,   
			F22 = GLFW_KEY_F22,   
			F23 = GLFW_KEY_F23,   
			F24 = GLFW_KEY_F24,   
			F25 = GLFW_KEY_F25,   
			KPNum0 = GLFW_KEY_KP_0,   
			KPNum1 = GLFW_KEY_KP_1,   
			KPNum2 = GLFW_KEY_KP_2,   
			KPNum3 = GLFW_KEY_KP_3,   
			KPNum4 = GLFW_KEY_KP_4,   
			KPNum5 = GLFW_KEY_KP_5,   
			KPNum6 = GLFW_KEY_KP_6,   
			KPNum7 = GLFW_KEY_KP_7,   
			KPNum8 = GLFW_KEY_KP_8,   
			KPNum9 = GLFW_KEY_KP_9,   
			KPDecimal = GLFW_KEY_KP_DECIMAL,   
			KPDivide = GLFW_KEY_KP_DIVIDE,   
			KPMultiply = GLFW_KEY_KP_MULTIPLY,   
			KPSubtract = GLFW_KEY_KP_SUBTRACT,   
			KPAdd = GLFW_KEY_KP_ADD,   
			KPEnter = GLFW_KEY_KP_ENTER,   
			KPEqual = GLFW_KEY_KP_EQUAL,   
			LeftShift = GLFW_KEY_LEFT_SHIFT,   
			LeftControl = GLFW_KEY_LEFT_CONTROL,   
			LeftAlt = GLFW_KEY_LEFT_ALT,   
			LeftSuper = GLFW_KEY_LEFT_SUPER,   
			RightShift = GLFW_KEY_RIGHT_SHIFT,   
			RightControl = GLFW_KEY_RIGHT_CONTROL,   
			RightAlt = GLFW_KEY_RIGHT_ALT,   
			RightSuper = GLFW_KEY_RIGHT_SUPER,   
			Menu = GLFW_KEY_MENU,  

			count // this one is a 'tricks' to automatically get the number
			// of elements in this enum (just type sibr::Key::count).
		};
	} // namespace Key

	namespace Mouse
	{
		/** Mouse button codes (based on GLFW codes). */
		enum Code
		{
			Button1 = GLFW_MOUSE_BUTTON_1, 
			Button2 = GLFW_MOUSE_BUTTON_2,
			Button3 = GLFW_MOUSE_BUTTON_3,
			Button4 = GLFW_MOUSE_BUTTON_4,
			Button5 = GLFW_MOUSE_BUTTON_5,
			Button6 = GLFW_MOUSE_BUTTON_6,
			Button7 = GLFW_MOUSE_BUTTON_7,
			Button8 = GLFW_MOUSE_BUTTON_8,
			Last = GLFW_MOUSE_BUTTON_LAST,

			Left = GLFW_MOUSE_BUTTON_LEFT,
			Middle = GLFW_MOUSE_BUTTON_MIDDLE,
			Right = GLFW_MOUSE_BUTTON_RIGHT,

			Unknown,
			count
		};
	} // namespace Mouse

	/** Helper keeping track of the number of keys currently pressed. */
	struct SIBR_GRAPHICS_EXPORT KeyCombination 
	{
		/// Default constructor.
		KeyCombination();

		/** Constructor.
		\param n number of keys pressed
		\param b are they active or not
		*/
		KeyCombination(int n, bool b); 
		
		/** \return true if they are numKeys pressed keys and their combination is active. */
		operator bool() const; 

		int numKeys; ///< Number of pressed keys.
		bool isTrue; ///< Activations status.
	};
	
	/** Merge two set of pressed keys.
	\param combA  first set
	\param combB second set
	\return the union set
	*/
	KeyCombination SIBR_GRAPHICS_EXPORT operator&&( const KeyCombination & combA, const KeyCombination & combB);

	/** Keep track of the pressed/active/released state of a set of keys/buttons.
	\sa Key::Code, Mouse::Code
	*/
	template <int TNbState, typename TEnum>
	class InputState
	{
	public:

		/** Is an item currently active.
		\param code the item code (key or mouse)
		\return true if the item is active at this frame
		*/
		bool	isActivated( TEnum code ) const {
			return _currentStates[(size_t)code];
		}

		/** Is an item released (lasts one frame).
		\param code the item code (key or mouse)
		\return true if the item is released at this frame
		*/
		bool	isReleased( TEnum code ) const {
			return _lastStates[(size_t)code] \
				&& !_currentStates[(size_t)code];
		}

		/** Is an item pressed at this frame (lasts one frame).
		\sa isActivated
		\param code the item code (key or mouse)
		\return true if the item is pressed at this frame
		*/
		bool	isPressed( TEnum code ) const {
			return !_lastStates[(size_t)code] \
				&& _currentStates[(size_t)code];
		}
		
		/** Is an item currently pressed and only this one (lasts one frame).
		\sa isActivated
		\param code the item code (key or mouse)
		\return true if the item is the only one pressed
		*/
		KeyCombination isPressedOnly( TEnum code ) const {
			return KeyCombination(1,isPressed(code));
		}

		/** Is an item currently active and only this one.
		\param code the item code (key or mouse)
		\return true if the item is the only one active
		*/
		KeyCombination isActivatedOnly( TEnum code ) const {
			return KeyCombination(1,isActivated(code));
		}

		/** Declare an item as pressed at this frame.
		\param code the item code (Key or Mouse).
		*/
		void	press( TEnum code ) {
			_currentStates[(size_t)code] = true;
		}

		/** Declare an item as released at this frame.
		\param code the item code (Key or Mouse).
		*/
		void	release( TEnum code ) {
			_currentStates[(size_t)code] = false;
			_lastStates[(size_t)code] = true;
		}

		/** Mute an item.
		\param code the item code (Key or Mouse).
		*/
		void	silent( TEnum code ) {
			_currentStates[(size_t)code] = \
				_lastStates[(size_t)code] = false;
		}

		/** Reset all items state.
		*/
		void	clearStates( void ) {
			std::fill(_currentStates.begin(), _currentStates.end(), false);
			std::fill(_lastStates.begin(), _lastStates.end(), false);
		}

		/** Update previous frame states with the current frame ones. */
		void	swapStates( void ) {
			_lastStates = _currentStates;
		}

		/** \return the number of keys currently activated. */
		int getNumActivated( void ) const {
			int n=0;
			for(int i=0; i<TNbState; ++i){
				n += (int)_currentStates[i];
			}
			return n;
		}

	private:
		std::array<bool, TNbState>			_currentStates; ///< Current frame state.
		std::array<bool, TNbState>			_lastStates; ///< Last frame state.
	};

	/** Maintain the complete state of user interactions (mouse, keyboard) for a given view or window.
	All coordinates are recaled with respect to the associated view.
	To check if the B key is currently held:
		input.key().isActivated(Key::B);
	To check if the right mouse was just released:
		input.mouseButton().isReleased(Mouse::Right);

	\ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT Input
	{
	public:
		typedef InputState<Key::count, Key::Code>		Keyboard;
		typedef InputState<Mouse::count, Mouse::Code>	MouseButton;

	public:

		/** \return the global system input (all others are derived from this one) */
		static Input&	global( void );

		/** Generate a new Input object based on a parent one and a viewport. Events (clicks) happening outside 
			the viewport will be ignored, mouse coordinates will be recentered with respect to the viewport.
			\param global the parent input
			\param viewport the viewport to retrict the input to
			\param mouseOutsideDisablesKeyboard if set to true, keyboard inputs are ignored when the mouse is outside the viewport
			\return the new restricted input
		*/
		static Input subInput(const sibr::Input & global, const sibr::Viewport & viewport, const bool mouseOutsideDisablesKeyboard = true);

		/** Is the mouse inside a given viewport.
		\param viewport the viewport to test against
		\return true if the mouse is inside.
		*/
		bool isInsideViewport(const sibr::Viewport & viewport) const;

		/** Update internal state based on GLFW, call once per frame. */
		static void		poll( void );

		/** \return the keyboard state. */
		const Keyboard&	key( void ) const {
			return _keyboard;
		}

		/** \return the keyboard state. */
		Keyboard&	key( void ) {
			return _keyboard;
		}

		/** \return the mouse buttons state. */
		const MouseButton&	mouseButton( void ) const {
			return _mouseButton;
		}

		/** \return the mouse buttons state. */
		MouseButton&	mouseButton( void ) {
			return _mouseButton;
		}

		/** \return the current mouse position */
		const Vector2i&	mousePosition( void ) const {
			return _mousePos;
		}

		/** Set the current mouse position.
		\param mousePos the new position
		*/
		void mousePosition( Vector2i mousePos ) {
			_mousePos = mousePos;
		}

		/** \return the change in mouse position since last frame. */
		Vector2i mouseDeltaPosition( void ) const {
			return _mousePrevPos-_mousePos;
		}
		
		/** If any number key is pressed, return the lowest one.
		\return the smallest pressed number, or -1 if none is pressed.
		*/
		int pressedNumber() const {
			static const std::vector<sibr::Key::Code> keys = {
				Key::Num0, Key::Num1, Key::Num2, Key::Num3, Key::Num4,
				Key::Num5, Key::Num6, Key::Num7, Key::Num8, Key::Num9
			};
		
			for (int i = 0; i < 10; ++i){
				if (_keyboard.isPressed(keys[i]) ){
					return i;
				}
			}
			return -1;
		}

		/** Update last frame state with the current one. Call at the end of each frame. */
		void swapStates( void ) {
			key().swapStates();
			mouseButton().swapStates();
			_mousePrevPos = _mousePos;
			_mouseScroll = 0.0;
		}

		/** \return the scroll amount along the vertical axis. */
		double			mouseScroll( void ) const {
			return _mouseScroll;
		}

		/** Set the scroll amount.
		\param  v the scroll amount.
		*/
		void			mouseScroll(double v) {
			_mouseScroll = v;
		}

		/** \return true if the input is associated to an empty view/window. */
		bool empty() const {
			return _empty;
		}

	private:

		Keyboard			_keyboard; ///< Keyboard state.
		MouseButton			_mouseButton; ///< Mouse state.

		Vector2i			_mousePos = {0, 0}; ///< Current mouse  position.
		Vector2i			_mousePrevPos = { 0, 0 }; ///< Previous mouse position.
		double				_mouseScroll = 0.0; ///< Current scroll amount.
		bool				_empty = true; ///< Is the input associated to an empty view/window.

	};

	///// DEFINITIONS /////


} // namespace sibr
