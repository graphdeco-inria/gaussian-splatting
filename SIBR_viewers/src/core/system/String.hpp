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

namespace sibr
{

	/**
	* \addtogroup sibr_system
	* @{
	*/

	/**
	* Replaces all occurences of a substring with another substring.
	* \param src the string to perform replacements in
	* \param search the substring to replace
	* \param replaceby the new substring to substitute
	* \return the string with the substitutions performed.
	*/
	SIBR_SYSTEM_EXPORT std::string strSearchAndReplace( const std::string& src, const std::string& search, const std::string& replaceby );

	/**
	* Process a string (a filename or path) to remove any extension if it exists.
	* \param str the string to remove the extension from
	* \return the string without extension
	*/
	SIBR_SYSTEM_EXPORT std::string removeExtension(const std::string& str);

	/**
	* Process a string (a filename or path) to extract the file extension if it exists.
	* \param str the string to get the extension from
	* \return the extension string (without the leading dot)
	*/
	SIBR_SYSTEM_EXPORT std::string getExtension(const std::string& str);

	/**
	* Process a string (a path) to return the parent directory.
	* \param str the string to process
	* \return the string with the last component removed
	* \note Will return the empty string if no separator was found.
	*/
	SIBR_SYSTEM_EXPORT std::string parentDirectory(const std::string& str);

	/**
	* Process a string (a path) to return the file name.
	* \param str the string to process
	* \return the string with all but the last component removed
	* \note Will return the full string if no separator was found.
	*/
	SIBR_SYSTEM_EXPORT std::string getFileName(const std::string& str);

	/**
	* Check if a string only contains digits.
	* \param str the string to check
	* \return true if it only contains digits
	*/
	SIBR_SYSTEM_EXPORT bool strContainsOnlyDigits(const std::string& str);

	/** Split string into sub-strings delimited by a given character. 
	 * \param str the input string
	 * \param delim the delimiting characters
	 * \return a list of split substrings
	 */
	SIBR_SYSTEM_EXPORT std::vector<std::string>	split(const std::string& str, char delim = '\n');

	/** Wrapper around sibr::sprintf that returns a string 
	 * \param msg the string with C placeholders
	 * \param ... the values for each placeholder
	 * \return the string with the formatted values inserted
	 */
	SIBR_SYSTEM_EXPORT std::string	sprint(const char *msg, ...);

	/** Write a formatted string with inserted values to a buffer.
	 * \param buffer the destination string
	 * \param size the size of the format string
	 * \param format the string with C placeholders
	 * \param ... the values for each placeholder
	 * \return a status code similar to sprintf
	 */
	SIBR_SYSTEM_EXPORT int 		sprintf(char* buffer, size_t size, const char* format, ...);

	/** Convert the input string to lowert case.
	 * \param str the input string
	 * \return the input string in lower case
	 */
	SIBR_SYSTEM_EXPORT std::string					to_lower(const std::string& str);

	/** Find if a list of substring is present in a given string.
	 * \param needles the list of substring
	 * \param haystack the search string
	 * \return true if any substring is present in the search string, else false
	 */
	SIBR_SYSTEM_EXPORT bool					find_any(const std::vector<std::string>& needles, const std::string& haystack);

	/** Write the current timestamp to a string.
	 * \param format the formatting to use for the timestamp (see default value for an example)
	 * \return a string containing the timestamp
	 */
	SIBR_SYSTEM_EXPORT std::string timestamp(const std::string & format = "%Y_%m_%d_%H_%M_%S");


	/*** @} */

} // namespace sibr
