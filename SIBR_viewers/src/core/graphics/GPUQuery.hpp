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
#include <core/system/Config.hpp>
#include <core/graphics/Config.hpp>

namespace sibr { 

	/**
	 * Provide a buffered wrapper around an OpenGL query object, avoiding manual synchronization.
	 * See section 4.2 of the OpenGL 4.6 specification for more details on the types of queries available
	 * (time elapsed, number of primitives, number of fragment writes...).
	 * 
	 * For example, to get the processing time of a mesh draw call, you can use the following.
	 * In renderer initialisation:
	 *		GPUQuery query(GL_TIME_ELAPSED);
	 * In the rendering loop:
	 *		query.begin();
	 *		mesh.draw();
	 *		query.end();
	 * In your GUI loop:
	 *		const uint64 time = query.value();
	 *		//... display it.
	 *	
	 * \warning Because the query is using buffering to avoid stalling when querying the value, 
	 * you SHOULD NOT use the same query object for multiple timings in the same frame. 
	 * It should also be use for multiple consecutive frames ; because of buffering again, 
	 * the first time value() is queried, it might be erroneous.
	 *
	 * \note If you want to create a query inline (for a one shot measurement), set the buffer 
	 * count to 1, and know that it will introduce a stall when querying the value.
	* \ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT GPUQuery
	{
		SIBR_CLASS_PTR(GPUQuery);

	public:

		/** Create a query of a given type.
		\param type the OpenGL enum type
		\param count number of buffered internal queries (ideally >= 2).
		*/
		GPUQuery(GLenum type, size_t count = 2);

		/** Start measuring. */
		void begin();

		/** Stop measuring. */
		void end();

		/** Obtain the raw value (time in nanoseconds, number of primitives,...) for the query before last.
		This allows for buffering from one frame to the next and avoid stalls (except if count is set to 1).
		\return the query value.
		*/
		uint64 value();

	private:
		
		std::vector<GLuint> _ids; ///< Internal queries IDs.
		const size_t _count; ///< Number of queries.
		GLenum _type; ///< Type of query.
		size_t _current = 0; ///< Current internla query used.
		bool _observing = false; ///< Are we currently measuring.
	};

} 
