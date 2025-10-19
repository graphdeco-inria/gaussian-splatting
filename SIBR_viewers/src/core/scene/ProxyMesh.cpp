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


#include "ProxyMesh.hpp"


namespace sibr {

	void ProxyMesh::loadFromData(const IParseData::Ptr & data)
	{
		_proxy.reset(new Mesh());
		// GD HACK
		if (boost::filesystem::extension(data->meshPath()) == ".bin") {
			if (!_proxy->loadSfM(data->meshPath(), data->basePathName())) {
				SIBR_WRG << "proxy model not found at " << data->meshPath() << std::endl;
			}
		}
		else if (!_proxy->load(data->meshPath(), data->basePathName()) && !_proxy->load(removeExtension(data->meshPath()) + ".ply") && !_proxy->load(removeExtension(data->meshPath()) + ".obj")) {
			if (!_proxy->loadSfM(data->meshPath(), data->basePathName())) {
				SIBR_WRG << "proxy model not found at " << data->meshPath() << std::endl;
			}
		}
		if (!_proxy->hasNormals()) {
			_proxy->generateNormals();
		}
	}

	void ProxyMesh::replaceProxy(Mesh::Ptr newProxy)
	{
		_proxy.reset(new Mesh());
		_proxy->vertices(newProxy->vertices());
		_proxy->normals(newProxy->normals());
		_proxy->colors(newProxy->colors());
		_proxy->triangles(newProxy->triangles());
		_proxy->texCoords(newProxy->texCoords());

		// Used by inputImageRT init() and debug rendering
		if (!_proxy->hasNormals())
		{
			_proxy->generateNormals();
		}

	}

	void ProxyMesh::replaceProxyPtr(Mesh::Ptr newProxy)
	{
		_proxy = newProxy;
	}


}

