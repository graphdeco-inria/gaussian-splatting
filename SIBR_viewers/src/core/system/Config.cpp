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



#include <mutex>
#include "core/system/Config.hpp"

std::mutex	gLogMutex;

namespace sibr
{ 

	LogExit::LogExit(void) :
		lock(gLogMutex)
	{ }

	void LogExit::operator <<=( const std::ostream& /*stream*/ )
	{
		// do exit, only profit a the rules of 'operator precedence'
		// to be executed after operator << when writing to the stream
		// itself.
		// So that this class is evaluated after writing the output and
		// it will exit (see dtor)
		//exit(EXIT_FAILURE);
		throw std::runtime_error("See log for message errors");
	}

	DebugScopeProfiler::~DebugScopeProfiler( void )
	{
		double t = double(clock() - _t0) / CLOCKS_PER_SEC;
		SIBR_LOG << "[PROFILER] Scope '" << _name <<
			"' completed in " << t << "sec." << std::endl;
	}

	DebugScopeProfiler::DebugScopeProfiler( const std::string& name )
		: _name(name)
	{ 
		_t0 = clock();
	}

} // namespace sirb