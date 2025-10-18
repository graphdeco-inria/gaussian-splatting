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

# include "core/system/Config.hpp"
# include "core/system/Vector.hpp"

namespace sibr
{
	/** Represents a simple aligned axis rectangle
	
	I use left, top, right, bottom and not x1, x2, y1, y2
	or x, y, w, h for being free of coordinate systems.
	(so you can use this class with an origin starting in
	the top left corner or the bottom right corner; this
	should be the same).
	
	Note this class is currently what I call a constclass:
	once built you cannot modify it.
	* \ingroup sibr_system
	*/
	template <typename T>
	class Rect
	{
	public:
		typedef	Eigen::Matrix<T, 2, 1,Eigen::DontAlign>			Vec;

	public:

		/// Default constructor (null everything)
		Rect( void );
		
		/** Constructor
		 \param left x-coordinate of the left border
		 \param top y-coordinate of the top border
		 \param right x-coordinate of the right border
		 \param bottom y-coordinate of the bottom border
		*/
		Rect( T left, T top, T right, T bottom );

		/// \return the position of the left side
		inline T		left( void ) const;
		/// \return the position of the right side
		inline T		right( void ) const;
		/// \return the position of the top side
		inline T		top( void ) const;
		/// \return the position of the bottom side
		inline T		bottom( void ) const;

		/// \return the width
		inline T		width( void ) const;
		/// \return the height
		inline T		height( void ) const;

		/// \return the top left rectangle corner location.
		inline Eigen::Matrix<T, 2, 1,Eigen::DontAlign>	cornerLeftTop( void ) const;
		/// \return the bottom left rectangle corner location.
		inline Eigen::Matrix<T, 2, 1,Eigen::DontAlign>	cornerLeftBottom( void ) const;
		/// \return the bottom right rectangle corner location.
		inline Eigen::Matrix<T, 2, 1,Eigen::DontAlign>	cornerRightBottom( void ) const;
		/// \return the top right rectangle corner location.
		inline Eigen::Matrix<T, 2, 1,Eigen::DontAlign>	cornerRightTop( void ) const;

	private:
		T	_left;		///< x-coordinate of the left border
		T	_top;		///< y-coordinate of the top border
		T	_right;		///< x-coordinate of the right border
		T	_bottom;	///< y-coordinate of the bottom border
	};

	///// EXPORT DEFAULT TYPES /////

	typedef Rect<float>		Rectf;
	typedef Rect<int>		Recti;

	///// DEFINITION /////

	template <typename T>
	Rect<T>::Rect( void )
	: _left(T(0)), _top(T(0)), _right(T(0)), _bottom(T(0)) {
	}
	template <typename T>
	Rect<T>::Rect( T left, T top, T right, T bottom )
	: _left(left), _top(top), _right(right), _bottom(bottom) {
	}

	template <typename T>
	T		Rect<T>::left( void ) const {
		return _left;
	}
	template <typename T>
	T		Rect<T>::right( void ) const {
		return _right;
	}
	template <typename T>
	T		Rect<T>::top( void ) const {
		return _top;
	}
	template <typename T>
	T		Rect<T>::bottom( void ) const {
		return _bottom;
	}

	template <typename T>
	T		Rect<T>::width( void ) const {
		T v = _right - _left;
		return (v<0.f)? -v : v;
	}
	template <typename T>
	T		Rect<T>::height( void ) const {
		T v = _top - _bottom;
		return (v<0.f)? -v : v;
	}

	template <typename T>
	Eigen::Matrix<T, 2, 1,Eigen::DontAlign>	Rect<T>::cornerLeftTop( void ) const {
		return Eigen::Matrix<T, 2, 1,Eigen::DontAlign>(left(),top());
	}
	template <typename T>
	Eigen::Matrix<T, 2, 1,Eigen::DontAlign>	Rect<T>::cornerLeftBottom( void ) const {
		return Eigen::Matrix<T, 2, 1,Eigen::DontAlign>(left(),bottom());
	}
	template <typename T>
	Eigen::Matrix<T, 2, 1,Eigen::DontAlign>	Rect<T>::cornerRightBottom( void ) const {
		return Eigen::Matrix<T, 2, 1,Eigen::DontAlign>(right(),bottom());
	}
	template <typename T>
	Eigen::Matrix<T, 2, 1,Eigen::DontAlign>	Rect<T>::cornerRightTop( void ) const {
		return Eigen::Matrix<T, 2, 1,Eigen::DontAlign>(right(),top());
	}

} // namespace sibr
