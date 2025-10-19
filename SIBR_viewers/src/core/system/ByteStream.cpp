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



#include <vector>
#include <iomanip>
#include <fstream>

#ifndef SIBR_OS_WINDOWS
# include <string.h>
#endif

#include "core/system/ByteStream.hpp"


namespace sibr
{
	void			ByteStream::memoryDump( void ) const 
	{
		const unsigned char* data = reinterpret_cast<const unsigned char*>(&_buffer[0]);
		std::cout << "Readable size: " << readableSize() << std::endl;
		std::cout << "Real size: " << bufferSize() << std::endl;
		std::cout << std::hex << std::setfill('0') << std::setw(2);
		for (unsigned i = 0; i < _buffer.size(); ++i)
		{
			const int blocksize = 2;
			for (unsigned j = 0; i < _buffer.size() && j < blocksize; ++j, ++i)
				std::cout << uint(data[i]);
			std::cout << ' ';
			for (unsigned j = 0; i < _buffer.size() && j < blocksize; ++j, ++i)
				std::cout << uint(data[i]);
			std::cout << ' ';
			for (unsigned j = 0; i < _buffer.size() && j < blocksize; ++j, ++i)
				std::cout << uint(data[i]);
			std::cout << ' ';
			for (unsigned j = 0; i < _buffer.size() && j < blocksize; ++j, ++i)
				std::cout << uint(data[i]);
			std::cout << ' ';
			std::cout << std::endl;
		}
		std::cout << std::dec << std::setw(0);
	}

	uint64	ByteStream::htonll(uint64 n)
	{
		if (ByteStream::systemIsBigEndian())
			return n;
		// Else we are on a little endian system
		uint32 out = 0;
		out |= (n & 0xFF00000000000000) >> 56;
		out |= (n & 0x00FF000000000000) >> 40;
		out |= (n & 0x0000FF0000000000) >> 24;
		out |= (n & 0x000000FF00000000) >> 8;
		out |= (n & 0x00000000FF000000) << 8;
		out |= (n & 0x0000000000FF0000) << 24;
		out |= (n & 0x000000000000FF00) << 40;
		out |= (n & 0x00000000000000FF) << 56;
		return out;
	}

	uint32	ByteStream::htonl( uint32 n ) 
	{
		if (ByteStream::systemIsBigEndian())
			return n;
		// Else we are on a little endian system
		uint32 out = 0;

		out |= (n & 0xFF000000) >> 24;
		out |= (n & 0x00FF0000) >> 8;
		out |= (n & 0x0000FF00) << 8;
		out |= (n & 0x000000FF) << 24;
		return out;
	}

	uint16	ByteStream::htons( uint16 n ) 
	{
		if (ByteStream::systemIsBigEndian())
			return n;
		// Else we are on a little endian system
		uint16 out = 0;

		out |= (n & 0xFF00) >> 8;
		out |= (n & 0x00FF) << 8;
		return out;
	}

	bool ByteStream::systemIsBigEndian( void ) 
	{
		static int16 isBigEndian = -1;
		if (isBigEndian == -1)
		{
			/// \todo TODO: check the compiler is not too smart and that it does not
			/// optimize this at compile-time with a const.
			int16 word = 0x0001;
			char *b = (char *) &word;
			isBigEndian = (b[0]? 0 /*little endian*/ : 1 /*big endian*/);
		}
		return isBigEndian != 0;
	}

	bool	ByteStream::load( const std::string& filename ) 
	{
		std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);

		if (file)
		{
			file.seekg(0, file.end);
			auto len = file.tellg();
			file.seekg(0, file.beg);

			_buffer.resize(len);
			file.read(reinterpret_cast<char*>(&_buffer[0]), len);

			file.close();
			return true;
		}
		else
			SIBR_WRG << "cannot load ByteStream from file '" << filename << "'." << std::endl;
		return false;
	}

	void	ByteStream::saveToFile( const std::string& filename ) 
	{
		if (bufferSize() == 0)
			return;

		std::ofstream	file(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);

		if (file)
		{
			file.write((char*)&_buffer[0], _buffer.size() * sizeof(_buffer[0]));
			file.close();
		}
		else
			SIBR_LOG << "ERROR: cannot write to the file '" << filename << "'" << std::endl;
	}

	void ByteStream::push(const void* data, uint size) 
	{
		assert(data != nullptr && size > 0);

		size_t curpos = _buffer.size();
		_buffer.resize(curpos + size);
		memcpy(&_buffer[curpos], data, size);
	}


	ByteStream& ByteStream::operator <<( bool b ) 
	{
		return ByteStream::operator << (static_cast<uint8>(b));
	}

	ByteStream& ByteStream::operator <<( int8 i ) 
	{
		push(&i, sizeof(i));
		return *this;
	}

	ByteStream& ByteStream::operator <<( int16 i ) 
	{
		int16 netorder = htons(i);
		push(&netorder, sizeof(netorder));
		return *this;
	}

	ByteStream& ByteStream::operator <<( int32 i ) 
	{
		int32 netorder = htonl(i);
		push(&netorder, sizeof(netorder));
		return *this;
	}

	ByteStream& ByteStream::operator <<( int64 i )
	{
		int64 netorder = htonll(i);
		push(&netorder, sizeof(netorder));
		return *this;
	}

	ByteStream& ByteStream::operator <<( uint8 i ) 
	{
		push(&i, sizeof(i));
		return *this;
	}

	ByteStream& ByteStream::operator <<( uint16 i ) 
	{
		uint16 netorder = htons(i);
		push(&netorder, sizeof(netorder));
		return *this;
	}

	ByteStream& ByteStream::operator <<( uint32 i ) 
	{
		uint32 netorder = htonl(i);
		push(&netorder, sizeof(netorder));
		return *this;
	}

	ByteStream& ByteStream::operator <<(uint64 i)
	{
		uint64 netorder = htonll(i);
		push(&netorder, sizeof(netorder));
		return *this;
	}
	
	ByteStream& ByteStream::operator <<( const std::string& str )
	{
		uint32 size = static_cast<uint32_t>(str.size());
		operator << (size);
		push(str.data(), sizeof(char)*size);
		return *this;
	}

	ByteStream& ByteStream::operator <<( float f )
	{
		int32* p = (int32*)&f;
		return ByteStream:: operator<<( *p );
	}

	ByteStream& ByteStream::operator <<(double d)
	{
		int64* p = (int64*)&d;
		return ByteStream:: operator<<(*p);
	}

} // namespace sibr
