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



#include "core/system/String.hpp"
#include <cstdarg>
#include <chrono>
#include <iomanip>

namespace sibr
{
	std::string strSearchAndReplace( const std::string& src, const std::string& search, const std::string& replaceby )
	{
		size_t pos = src.find(search);
		if (pos != std::string::npos)
		{
			std::string out;
			out = src.substr(0, pos) + replaceby + src.substr(pos+search.size(), src.size()-pos+search.size());
			return out;
		}
		return src;
	}

	std::string removeExtension(const std::string & str)
	{
		return  str.substr(0, str.find_last_of('.'));
	}

	std::string getExtension(const std::string & str)
	{
		const std::string::size_type dotPos = str.find_last_of('.');
		if(dotPos == std::string::npos) {
			return "";
		}
		return str.substr(dotPos+1);
	}

	std::string parentDirectory(const std::string & str)
	{
		const char kPathSeparator =
#ifdef _WIN32
				'\\';
#else
				'/';
#endif
		const std::string::size_type pos = str.find_last_of("/\\");
		// If no separator, return empty path.
		if(pos == std::string::npos) {
			return str + kPathSeparator + "..";
		}
		// If the separator is not trailing, we are done. 
		if(pos < str.size()-1) {
			return str.substr(0, pos);
		}
		// Else we have to look for the previous one.
		const std::string::size_type pos1 = str.find_last_of("/\\", pos-1);
		return str.substr(0, pos1);
	}

	SIBR_SYSTEM_EXPORT std::string getFileName(const std::string & str)
	{
		const std::string::size_type pos = str.find_last_of("/\\");
		if (pos == std::string::npos) {
			return str;
		}
		return str.substr(pos+1);
	}

	bool strContainsOnlyDigits(const std::string& str)
	{
		for (char c : str)
			if (c < '0' || c > '9')
				return false;
		return true;
	}

	std::vector<std::string>	split(const std::string& str, char delim)
	{
		std::stringstream	ss(str);
		std::string			to;
		std::vector<std::string>	out;

		if (str.empty())
			return out;

		while (std::getline(ss, to, delim))
			out.push_back(to);
		return out;
	}

	/// Wrapper around sibr::sprintf that returns a string
	std::string sprint(const char *msg, ...)
	{
#define TEMP_STR_SIZE 4096
		va_list args;
		va_start(args, msg);
		char s_StrSingle[TEMP_STR_SIZE];
#ifdef WIN32
		vsprintf_s(s_StrSingle, TEMP_STR_SIZE, msg, args);
#else
		vsnprintf(s_StrSingle, TEMP_STR_SIZE, msg, args);
#endif
		va_end(args);
		return std::string(s_StrSingle);
#undef TEMP_STR_SIZE
	}

	int 		sprintf(char* buffer, size_t size, const char* format, ...)
	{
		va_list args;
		int ret = 0;
		va_start(args, format);
#ifdef WIN32
		ret = vsprintf_s(buffer, size, format, args);
#else
		ret = vsnprintf(buffer, size, format, args);
#endif
		va_end(args);
		return ret;
	}

	SIBR_SYSTEM_EXPORT std::string to_lower(const std::string& str)
	{
		std::string out;
		out.reserve(str.length());

		for (size_t i = 0; i < str.length(); ++i)
			out.push_back(tolower(str[i]));

		return out;
	}


	SIBR_SYSTEM_EXPORT bool find_any(const std::vector<std::string>& needles, const std::string& haystack)
	{
		for (std::string needle : needles)
		{
			if (haystack.find(needle) != std::string::npos)
				return true;
		}

		return false;
	}

	std::string timestamp(const std::string & format) {
		auto now = std::time(nullptr);
#ifdef SIBR_OS_WINDOWS
		tm ltm = { 0,0,0,0,0,0,0,0,0 };
		localtime_s(&ltm, &now);
#else
		tm ltm = *(std::localtime(&now));
#endif
		std::stringstream buffer;
		buffer << std::put_time(&ltm, format.c_str());
		return buffer.str();
	}

} // namespace sirb

