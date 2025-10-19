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


#include "XMLTree.h"
#include "rapidxml/rapidxml_print.hpp"
#include <iostream>
#include <fstream>
#include <sstream>


namespace sibr {
	XMLTree::XMLTree(const std::string &  path)
	{
		std::cout << "Parsing xml file < " << path << " > : ";
		std::ifstream file(path.c_str());
		if (file) {
			std::stringstream buffer;
			buffer << file.rdbuf();
			file.close();
			xmlString = std::move(std::string(buffer.str()));
			this->parse<0>(&xmlString[0]);
			std::cout << "success " << std::endl;
		}
		else {
			std::cout << "error, cant open file " << std::endl;
		}
	}


	XMLTree::~XMLTree(void)
	{
	}


	bool XMLTree::save(const std::string & path) const {
		std::ofstream file(path);
		if(!file.is_open()) {
			SIBR_WRG << "Unable to save XML to path \"" << path << "\"." << std::endl;
			return false;
		}

		file << *this;
		file.close();
		return true;
	}

}
