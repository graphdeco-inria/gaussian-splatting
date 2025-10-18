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

#include <rapidxml/rapidxml.hpp>
#include <string>
#include "Config.hpp"

namespace sibr {

	/** Wrapper of rapidxml xml_document<> class so that the string associated to the xml file stays in memory.
	Needed to access nodes by their names.
	* \ingroup sibr_system
	*/
	class SIBR_SYSTEM_EXPORT XMLTree : public rapidxml::xml_document<>
	{
	public:
		/** Construct an XML structure from the content of a file.
		\param path the file path
		*/
		XMLTree(const std::string & path);

		/** Destructor. */
		~XMLTree(void);

		/** Save the XML structure to a file as a string representation.
		\param path output path
		\return a success flag
		*/
		bool save(const std::string & path) const;

	private:
		std::string xmlString; //< Internal copy of the laoded string.
	};
}
