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

# include <atomic>
# include <thread>
# include <mutex>
# include <queue>

# include "core/system/Config.hpp"

namespace sibr
{
	/** Class used to work concurrently on multiple tasks/instructions.
	 The only shared object is a queue (TaskIds) that
	 contains ids of remaining tasks to perform.
	
	 Typically, you use this id to access a const array
	 (input) and write results to another array using once
	 again this id. The output array is already resized
	 at the begin so that you can freely modify its
	 element without hurting other threads.
	
	Code Example:

		std::vector<ThreadIdWorker>	workers(MASKPATCH_NBTHREADS);

		// Launch all threads
		for (ThreadIdWorker& t: workers)
		t = std::move(ThreadIdWorker(taskId, workFunc));

		// Wait for all threads
		for (ThreadIdWorker& t: workers)
		if (t.joinable())
		t.join();

	 \ingroup sibr_system
	*/
	class /*SIBR_SYSTEM_EXPORT*/ ThreadIdWorker : public std::thread
	{
	public:
		typedef	std::queue<uint>	TaskIds;
	public:
		/// Build an empty worker (placeholder)
		ThreadIdWorker( void );

		/** Move constructor
		 *\param other worker to move
		 */
		ThreadIdWorker( ThreadIdWorker&& other ) noexcept;

		/** Constructor. Will call the passed function for each given task ID.
		 \param ids a list of task ids
		 \param func a function receiving a task ID as parameter returning either FALSE for signaling the worker to stop or TRUE for keep going.
		*/
		ThreadIdWorker( TaskIds& ids, std::function<bool(uint)> func );

		/** Move operator.
		 *\param other worker to assign
		 *\return the current worker
		 */
		ThreadIdWorker& operator =( ThreadIdWorker&& other ) noexcept;

		/// Deleted copy operator.
		ThreadIdWorker(const ThreadIdWorker&) = delete;

	private:

		/** Will pull the next task or automatically stop.
		 \param ids a list of task ids
		 \param func a function receiving a task ID as parameter
		 */
		void		taskPuller( TaskIds& ids, std::function<bool(uint)> func );

		SIBR_SYSTEM_EXPORT static std::mutex	g_mutex;	///< used to protect the common shared TaskIds list
	};

	///// INLINES /////
	inline ThreadIdWorker::ThreadIdWorker( void ) {
	}

	inline ThreadIdWorker::ThreadIdWorker( ThreadIdWorker&& other ) noexcept :
		std::thread(std::move((std::thread&)other)) {
	}

	inline ThreadIdWorker::ThreadIdWorker( TaskIds& ids, std::function<bool(uint)> func )
		: std::thread( [this, &ids, &func]() { taskPuller(ids, std::move(func)); } ) {
	}

	inline ThreadIdWorker& ThreadIdWorker::operator =( ThreadIdWorker&& other ) noexcept {
		((std::thread*)this)->operator=(std::move(other)); return *this;
	}

	inline void ThreadIdWorker::taskPuller( TaskIds& ids, std::function<bool(uint)> func ) {
		uint id = 0;
		bool stop = false;
		while (!stop)
		{
			{ // Pop next id
				std::lock_guard<std::mutex>	lock(g_mutex);
				stop = ids.empty();

				if (!stop)
				{
					id = ids.front();
					ids.pop();
				}
			}

			if (!stop)
				stop = !func(id);
		}
	}

} // namespace sibr
