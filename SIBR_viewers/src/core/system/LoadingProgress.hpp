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

# include <functional>
# include <chrono>
# include <mutex>
# include "core/system//Config.hpp"


namespace sibr
{
	///
	/// Simple utility class for reporting on the standard output
	/// a loading progess. (So users know your heavy computations
	/// didn't crash)
	///
	/// Instructions:
	/// 1) Instantiate just before a loop (for or while), providing
	/// the max number of iterations.
	/// 2) Call walk() once in a the loop.
	/// \ingroup sibr_system
	///
	class SIBR_SYSTEM_EXPORT LoadingProgress
	{
	public:
		typedef std::chrono::steady_clock						clock;
		typedef clock::time_point								time_point;
		typedef std::function<void (float, const std::string&)>	ExternalCallback;

		/** Create a progress bar.
		\param maxIteration total number of iterations
		\param status a message that will be inserted in next reports
		\param interval an interval of time between each report.
		*/
		LoadingProgress( size_t maxIteration,
			const std::string& status="", float interval=1.f );

		/// Make the loading progress by the given number of steps.
		/// \param step number of steps
		void				walk( size_t step = 1);
		///	\return the current progress in a range [0.0, 1.0]
		float				current( void ) const;

		/// \return the time interval used
		inline float				interval( void ) const;
		/// Change the frequency of each report
		/// \param interval the new step interval to use
		inline void					interval( float interval );

		/// \return the status message used
		inline const std::string&	status( void ) const;
		/// Insert a message in printed reports
		/// \param message the message to insert
		inline void					status( const std::string& message );

	private:
		/// Print a report
		void				report( void ) const;

		size_t		_currentStep;	///< current number of iterations
		size_t		_maxProgress;	///< number of iterations before reaching 100%
		std::string	_status;		///< inserted into a report (you can update it)
		float		_interval;		///< time interval before next report (sec)
		time_point	_lastReport;	///< time point saved during the last report
		std::mutex	_mutex;			///< used ot thread-safe this class (not heavly tested!)
	};

	///// DEFINITIONS /////

	float				LoadingProgress::interval( void ) const {
		return _interval;
	}
	void				LoadingProgress::interval( float interval ) {
		_interval = interval;
	}


	const std::string&	LoadingProgress::status( void ) const {
		return _status;
	}
	void				LoadingProgress::status( const std::string& message ) {
		_status = message;
	}


} // namespace sibr
