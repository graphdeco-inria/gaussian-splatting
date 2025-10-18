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

# include <vector>
#include <core/system/Vector.hpp>

namespace sibr
{
	template <typename T> class Array2d;


	///
	/// Used to store custom data on a map using pixel position 
	/// (2d unsigned pos).
	/// Internally, this simply use a one dimensional std::vector. 
	/// This class abstract convert operation from 2d to 1d for 
	/// making code easy to read.
	/// \ingroup sibr_system
	///
	template <typename T>
	class Array2d
	{
	public:
		// We use 'reference' defined by std::vector (and STL) and they are
		// a bit special. For example, I suppose that some compiler should
		// be able to replace a heave 'const bool&' by a simple 'bool' (in
		// this case a copy is far more cheaper). And what about 'bool&' ?
		typedef typename std::vector<T>::const_reference	const_reference;
		typedef typename std::vector<T>::reference			reference;

		/// Build from the given size.
		/// Note you can't resize your Array2d (just create a new one 
		/// if you need).
		Array2d( uint width=0, uint height=0 );
		/// Build from the given size and using the given default value.
		/// Note you can't resize your Array2d (just create a new one 
		/// if you need).
		Array2d( uint width, uint height, const_reference defaultValue );

		/// Destructor.
		~Array2d( );

		/// Return the width of this Array2d
		uint	width( void ) const;
		/// Return the height of this Array2d
		uint	height( void ) const;
		/// Return the width of this Array2d
		uint	w(void) const;
		/// Return the height of this Array2d
		uint	h(void) const;

		/// Return TRUE if is empty
		bool	empty( void ) const;

		/// Return data about a pixel at given coordinates
		const_reference		operator ()( uint x, uint y ) const;
		/// Access data about a pixel at given coordinates
		reference			operator ()( uint x, uint y );

		/// Return data about a pixel at given coordinates
		const_reference		operator ()(const sibr::Vector2i & coords) const;
		/// Access data about a pixel at given coordinates
		reference			operator ()(const sibr::Vector2i & coords);

		/// Return data about a pixel at given index
		const_reference		operator []( size_t i ) const;
		/// Access data about a pixel at given index
		reference			operator []( size_t i );

		/// Return the total size of the one dimensional array
		size_t				size( void ) const;

		/// Return data accessible in a one array form
		/// \deprecated Use Array2d<T>::vector( void ) instead.
		const std::vector<T>&	operator () ( void ) const;
		std::vector<T>&			operator () ( void );

		/// Return the internally used std::vector (so you
		/// can use STL algos).
		const std::vector<T>&	vector( void ) const;
		std::vector<T>&			vector( void );

		/// Return a pointer to the first byte a stored
		/// data.
		void*					data( void ) const;
		void*					data( void );



		/// Return the element index for the given coordinates
		inline uint		index( uint x, uint y ) const;

		/// Return FALSE if x,y are out of range. (DON'T print error)
		inline bool		inRange( uint x, uint y ) const;
		inline bool		isInRange( uint x, uint y) const;


	protected:
		/// Return FALSE if x,y are out of range. (print error)
		bool	checkSizeFor( uint x, uint y ) const;

		uint				_width;		///< Width of the pixel map
		uint				_height;	///< Height of the pixel map
		std::vector<T>		_data;		///< data of the pixel map
	};
	

	///// DEFINITION /////
	
	template<typename T>
	Array2d<T>::Array2d( uint width, uint height )
	: _width(width), _height(height), _data(_width*_height) {
	}

	template<typename T>
	Array2d<T>::Array2d( uint width, uint height, const_reference defaultValue )
	: _width(width), _height(height), _data(_width*_height, defaultValue) {
	}

	template<typename T>
	Array2d<T>::~Array2d( )
	{
		_data.clear();
	}

	template<typename T>
	uint	Array2d<T>::width( void ) const {
		return _width;
	}

	template<typename T>
	uint	Array2d<T>::height( void ) const {
		return _height;
	}

	template<typename T>
	uint	Array2d<T>::w(void) const {
		return _width;
	}

	template<typename T>
	uint	Array2d<T>::h(void) const {
		return _height;
	}


	template<typename T>
	typename Array2d<T>::const_reference		Array2d<T>::operator ()( uint x, uint y ) const {
		checkSizeFor(x, y);
		return _data.at(index(x, y));
	}
	template<typename T>
	typename Array2d<T>::reference			Array2d<T>::operator ()( uint x, uint y) {
		checkSizeFor(x, y);
		return _data[index(x, y)];
	}

	template<typename T>
	typename Array2d<T>::const_reference		Array2d<T>::operator ()(const sibr::Vector2i & coords) const {
		return _data[index(coords[0], coords[1])];
	}
	template<typename T>
	typename Array2d<T>::reference			Array2d<T>::operator ()(const sibr::Vector2i & coords) {
		return _data[index(coords[0], coords[1])];
	}

	template<typename T>
	const std::vector<T>&	Array2d<T>::operator () ( void ) const {
		return _data;
	}
	template<typename T>
	std::vector<T>&	Array2d<T>::operator () ( void ) {
		return _data;
	}


	template<typename T>
	const std::vector<T>&	Array2d<T>::vector( void ) const {
		return _data;
	}
	template<typename T>
	std::vector<T>&	Array2d<T>::vector( void ) {
		return _data;
	}
	
	template<typename T>
	void*					Array2d<T>::data( void ) const {
		return vector().empty()? nullptr : &vector()[0];
	}
	template<typename T>
	void*					Array2d<T>::data( void ) {
		return vector().empty()? nullptr : &vector()[0];
	}

	template<typename T>
	bool	Array2d<T>::empty( void ) const {
		return vector().empty();
	}

	
	template<typename T>
	typename Array2d<T>::const_reference		Array2d<T>::operator []( size_t i ) const {
		return _data.at(i);
	}

	template<typename T>
	typename Array2d<T>::reference			Array2d<T>::operator []( size_t i ) {
		return _data[i];
	}

	template<typename T>
	size_t				Array2d<T>::size( void ) const {
		return _data.size();
	}


	template<typename T>
	uint		Array2d<T>::index( uint x, uint y ) const {
		return y*_width + x;
	}

	template<typename T>
	bool	Array2d<T>::inRange( uint x, uint y ) const {
		return (x < _width && y < _height);
	}

	template<typename T>
	bool	Array2d<T>::isInRange(uint x, uint y) const {
		return (x < _width && y < _height);
	}

	template<typename T>
	bool	Array2d<T>::checkSizeFor( uint x, uint y ) const {
		if (inRange(x, y))
			return true;
		//else
		SIBR_ERR << "invalid pixelmap range at " << x << ", " << y
			<< "(current size: " << _width << ", " << _height << "). "
			<< std::endl;

		assert(false);// else it will crash because of std::vector
		return false;
	}

} // namespace sibr
