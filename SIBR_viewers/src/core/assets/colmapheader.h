// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_UTIL_ENDIAN_H_
#define COLMAP_SRC_UTIL_ENDIAN_H_

#include <vector>
#include <algorithm>
#include <iostream>

//namespace colmap {

// Reverse the order of each byte.
template <typename T>
T ReverseBytes(const T& data);

// Check the order in which bytes are stored in computer memory.
bool IsLittleEndian();
bool IsBigEndian();

// Convert data between endianness and the native format. Note that, for float
// and double types, these functions are only valid if the format is IEEE-754.
// This is the case for pretty much most processors.
template <typename T>
T LittleEndianToNative(const T x);
template <typename T>
T BigEndianToNative(const T x);
template <typename T>
T NativeToLittleEndian(const T x);
template <typename T>
T NativeToBigEndian(const T x);

// Read data in little endian format for cross-platform support.
template <typename T>
T ReadBinaryLittleEndian(std::istream* stream);
template <typename T>
void ReadBinaryLittleEndian(std::istream* stream, std::vector<T>* data);

// Write data in little endian format for cross-platform support.
template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const T& data);
template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const std::vector<T>& data);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
T ReverseBytes(const T& data) {
  T data_reversed = data;
  std::reverse(reinterpret_cast<char*>(&data_reversed),
               reinterpret_cast<char*>(&data_reversed) + sizeof(T));
  return data_reversed;
}

inline bool IsLittleEndian() {
#ifdef BOOST_BIG_ENDIAN
  return false;
#else
  return true;
#endif
}

inline bool IsBigEndian() {
#ifdef BOOST_BIG_ENDIAN
  return true;
#else
  return false;
#endif
}

template <typename T>
T LittleEndianToNative(const T x) {
  if (IsLittleEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T BigEndianToNative(const T x) {
  if (IsBigEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T NativeToLittleEndian(const T x) {
  if (IsLittleEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T NativeToBigEndian(const T x) {
  if (IsBigEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T ReadBinaryLittleEndian(std::istream* stream) {
  T data_little_endian;
  stream->read(reinterpret_cast<char*>(&data_little_endian), sizeof(T));
  return LittleEndianToNative(data_little_endian);
}

template <typename T>
void ReadBinaryLittleEndian(std::istream* stream, std::vector<T>* data) {
  for (size_t i = 0; i < data->size(); ++i) {
    (*data)[i] = ReadBinaryLittleEndian<T>(stream);
  }
}

template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const T& data) {
  const T data_little_endian = NativeToLittleEndian(data);
  stream->write(reinterpret_cast<const char*>(&data_little_endian), sizeof(T));
}

template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const std::vector<T>& data) {
  for (const auto& elem : data) {
    WriteBinaryLittleEndian<T>(stream, elem);
  }
}

//}  // namespace colmap

#include <algorithm>
#include <cstdarg>
#include <fstream>
#include <sstream>

bool IsNotWhiteSpace(const int character) {
  return character != ' ' && character != '\n' && character != '\r' &&
         character != '\t';
}

bool StringStartsWith(const std::string& str, const std::string& prefix) {
  return !prefix.empty() && prefix.size() <= str.size() &&
         str.substr(0, prefix.size()) == prefix;
}

void StringLeftTrim(std::string* str) {
  str->erase(str->begin(),
             std::find_if(str->begin(), str->end(), IsNotWhiteSpace));
}

void StringRightTrim(std::string* str) {
  str->erase(std::find_if(str->rbegin(), str->rend(), IsNotWhiteSpace).base(),
             str->end());
}

void StringTrim(std::string* str) {
  StringLeftTrim(str);
  StringRightTrim(str);
}

#endif  // COLMAP_SRC_UTIL_ENDIAN_H_
