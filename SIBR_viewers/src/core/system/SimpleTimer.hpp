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
#include <vector>
#include <chrono>

namespace sibr
{
	/**
	* Timer to monitor performance of a section of code.
	* \ingroup sibr_system
	*/
	class Timer
	{
	public:
		typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
		typedef std::chrono::nanoseconds nano;
		typedef std::chrono::microseconds micro;
		typedef std::chrono::milliseconds milli;
		typedef std::chrono::seconds s;

		const double timeResolution = (double)std::chrono::high_resolution_clock::period::num
			/ std::chrono::high_resolution_clock::period::den; ///< Timer resolution.

		/** Constructor. Create a timer.
		\param start_now start to measure time at creation
		*/
		Timer(bool start_now = false ) : hasStarted(false)
		{
			if (start_now) {
				tic();
			}
		}

		/** Copy constructor
		\param timer another timer
		*/
		Timer(const Timer & timer) {
			hasStarted = timer.hasStarted;
			current_tic = timer.current_tic;
		}

		/** Start measuring elapsed time.
		 * \warning This will clear existing recorded times.
		*/
		void tic()
		{
			tocs.resize(0);
			hasStarted = true;
			current_tic = std::chrono::high_resolution_clock::now();
		}

		/** Save currently elapsed time.
		 * \note You can call toc multiple times in a row.
		*/
		void toc()
		{
			auto toc = std::chrono::high_resolution_clock::now();
			tocs.push_back(toc);
		}

		/** Get the time elapsed since the last tic, with a precisiond etemrined by the tempalte argument.
			\return the measured time (default: in ms)
		*/
		template<typename T = Timer::milli>
		double deltaTimeFromLastTic() const
		{
			if (!hasStarted) { return std::numeric_limits<double>::max(); }
			auto toc = std::chrono::high_resolution_clock::now();
			
			double deltaTime = 1;
			if (!getDeltaTime<T>(current_tic, toc, deltaTime)) {
				std::cout << "[SIBR - Timer] : below time reslution " << std::endl;
			}

			return deltaTime;
		}

		/** Print a list of all the recorded tocs, with the precision specified as a template argument (by default in ms).
			\param toc_now should a toc be generated right now.
		*/
		template<typename T = Timer::milli>
		void display(bool toc_now = false)
		{
			if (toc_now) {
				toc();
			}
			const int n = (int)tocs.size();
			if (!hasStarted || n == 0) {
				std::cout << "[SIBR - Timer] : no tic or no toc" << std::endl;
			}
			else {
				double deltaTime;
				for (auto & toc : tocs) {
					if (getDeltaTime<T>(current_tic,toc,deltaTime) ) {
						std::cout << "[SIBR - Timer] : " << deltaTime << std::endl;
					}
					else {
						std::cout << "[SIBR - Timer] : below time reslution " << std::endl;
					}
				}
			}
		}

		/** Get the time elapsed between two points in time, using the precision specified as a template argument (default to ms).
		\param tic first time point
		\param toc second time point
		\param deltaTime will contain the computed duration
		\return false if the elapsed time was below the timer precision.
		*/
		template<typename T = Timer::milli>
		bool getDeltaTime(const time_point & tic, const time_point & toc, double & deltaTime) const {
			double timediff_nanoSeconds = (double)std::chrono::duration_cast<Timer::nano>(toc - tic).count();
			if (timediff_nanoSeconds < Timer::timeResolution) {
				return false;
			}
			else {
				deltaTime = (double)std::chrono::duration_cast<T>(toc - tic).count();
				return true;
			}
		}

	private:
		time_point current_tic; ///< Initial tic.
		std::vector<time_point> tocs; ///< Recorded time points.
		bool hasStarted; ///< Is the timer currently running.
	};
	

} // namespace sibr
