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


#include <algorithm>
#include <string> 
#include <sstream> 
#include <iomanip>
#include "core/view/UIShortcuts.hpp"

namespace sibr
{
	/*static*/ UIShortcuts& UIShortcuts::global( void )
	{
		static UIShortcuts instance;
		return instance;
	}

	void	UIShortcuts::list( void )
	{
		// Sort elements in alphabetical order.
		std::vector<std::pair<std::string, const char *>> elems(_shortcuts.begin(), _shortcuts.end());
		std::sort(elems.begin(), elems.end(), [](std::pair<std::string, const char *> a, std::pair<std::string, const char *> b) { return b.first < a.first; });
		
		std::ostringstream oss;
		for (auto& pair: elems)
			oss << "  " << std::setw(24)<< std::left << pair.first << " : " << pair.second << std::endl;
		SIBR_LOG << "List of Shortcuts:\n" << oss.str() << std::endl;
	}

	void	UIShortcuts::add( const std::string& shortcut, const char* desc )
	{
		std::string lshortcut = shortcut;
		std::transform(lshortcut.begin(), lshortcut.end(), lshortcut.begin(), ::tolower);

		if (_shortcuts.find(lshortcut) == _shortcuts.end())
			_shortcuts[lshortcut] = desc;
		else
		{
			const char* current = _shortcuts[lshortcut];
			if (current != desc)
			{
				SIBR_ERR << "conflict with shortcuts.\n"
					"Trying to register:\n"
					"[" << shortcut << "] : " << desc
					<< "\nBut already exists as:\n"
					"[" << shortcut << "] : " << current
					<< std::endl;
			}
		}
	}


} // namespace sibr
