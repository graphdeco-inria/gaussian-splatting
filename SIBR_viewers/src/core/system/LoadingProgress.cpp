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



#include "core/system/LoadingProgress.hpp"

namespace sibr
{

	LoadingProgress::LoadingProgress( size_t maxIteration,
		const std::string& status, float interval )
		: _currentStep(0), _maxProgress(maxIteration), _status(status), _interval(interval)
	{
		_lastReport = clock::now();
	}

	void				LoadingProgress::walk( size_t step )
	{
		std::lock_guard<std::mutex> l(_mutex);

		_currentStep += step;
		if (std::chrono::duration<float>(clock::now()-_lastReport).count() >= _interval
			|| _currentStep >= _maxProgress)
		{
			report();
			_lastReport = clock::now();
		}

	}

	float				LoadingProgress::current( void ) const
	{
		if (_maxProgress <= 0)
			return 1.f;
		return (float)_currentStep/(float)_maxProgress;
	}

	void				LoadingProgress::report( void ) const
	{
		if (_status.empty())
			SIBR_LOG << "Progression [ "<< current()*100.f <<"% ]" << std::endl;
		else
			SIBR_LOG << "Progression [ "<< current()*100.f <<"% ] - " << _status << std::endl;
	}

} // namespace sibr
