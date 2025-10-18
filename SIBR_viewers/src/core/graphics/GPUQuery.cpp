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


#include "GPUQuery.hpp"

using namespace sibr;

GPUQuery::GPUQuery(GLenum type, size_t count) : _count(count), _type(type){
	if(count < 2) {
		SIBR_WRG << "Using a buffer of size >= 2 is recommended to avoid synchronization problems." << std::endl;
	}
	_ids.resize(count);
	glGenQueries(GLsizei(_ids.size()), &_ids[0]);
	_current = _count - 1;
	// Dummy initial query.
	for (int i = 0; i < _count; ++i)
	{
		begin();
		end();
	}
}

void GPUQuery::begin() {
	if(_observing) {
		SIBR_WRG << "Query already started..." << std::endl;
		return;
	}
	_current = (_current + 1) % _count;
	glBeginQuery(_type, _ids[_current]);
	_observing = true;
}

void GPUQuery::end() {
	if (!_observing) {
		SIBR_WRG << "Query not running..." << std::endl;
		return;
	}
	glEndQuery(_type);
	_observing = false;
}

uint64 GPUQuery::value() {
	if (_observing) {
		SIBR_WRG << "Query still running, ending it first..." << std::endl;
		end();
	}
	// We want the ID of the previous frame, taking into account that we have incremented the counter once more when ending it. So minus 1.
	// Except if you have only one query, in which case we query this one, but it will stall the GPU.
	const size_t previous = (_current - 1) % _count;
	GLuint64 data = 0;
	glGetQueryObjectui64v(_ids[previous], GL_QUERY_RESULT, &data);
	//CHECK_GL_ERROR;
	return data;
}