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



#include "Raycaster.hpp"

namespace sibr
{
	/*static*/ SIBR_RAYCASTER_EXPORT const Raycaster::geomId		Raycaster::InvalidGeomId = RTC_INVALID_GEOMETRY_ID;
	/*static*/ bool													Raycaster::g_initRegisterFlag = false;
	/*static*/ Raycaster::RTCDevicePtr								Raycaster::g_device = nullptr;

	/*static*/ void Raycaster::rtcErrorCallback(void* userPtr, RTCError code, const char* msg)
	{
		std::string err;

		switch (code)
		{
		case RTC_ERROR_UNKNOWN: err = std::string("RTC_ERROR_UNKNOWN"); break;
		case RTC_ERROR_INVALID_ARGUMENT: err = std::string("RTC_ERROR_INVALID_ARGUMENT"); break;
		case RTC_ERROR_INVALID_OPERATION: err = std::string("RTC_ERROR_INVALID_OPERATION"); break;
		case RTC_ERROR_OUT_OF_MEMORY: err = std::string("RTC_ERROR_OUT_OF_MEMORY"); break;
		case RTC_ERROR_UNSUPPORTED_CPU: err = std::string("RTC_ERROR_UNSUPPORTED_CPU"); break;
		case RTC_ERROR_CANCELLED: err = std::string("RTC_ERROR_CANCELLED"); break;
		default: err = std::string("invalid error code"); break;
		}

		SIBR_ERR << "Embree reported the following issue - "
			<< "[" << err << "]'" << msg << "'" << std::endl;
	}

	Raycaster::~Raycaster(void)
	{
		_scene = nullptr;
		_devicePtr = nullptr;
		if (g_device && g_device.use_count() == 1)
			g_device = nullptr; // if nobody use it, free it
	}

	bool	Raycaster::init(RTCSceneFlags sceneType)
	{
		if (!g_device)
		{
			// The two following macros set flagbits on the control register
			// used by SSE (see http://softpixel.com/~cwright/programming/simd/sse.php)
			_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);			// Enable 'Flush Zero' bit
			_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);	// Enable 'Denormals Zero'bit

			SIBR_LOG << "Initializing Raycaster" << std::endl;

			g_device = std::make_shared<RTCDevice>(rtcNewDevice(NULL));

			if (g_device == nullptr) {
				SIBR_LOG << "Cannot create an embree device : " << rtcGetDeviceError(*g_device.get()) << std::endl;
			}

			rtcSetDeviceErrorFunction(*g_device.get(), &Raycaster::rtcErrorCallback, nullptr); // Set callback error function
			_devicePtr = g_device; //Moved in the init
		}

		if (_scene)
			return true;



		_scene = std::make_shared<RTCScene>( // define a new scene
			/// \todo create a new static scene optimized for primary rays (TODO: test perf with RTC_SCENE_ROBUST)
			rtcNewScene(*g_device.get())
			); // set a custom deleter

		if (_scene == nullptr)
			SIBR_LOG << "Cannot create an embree scene" << std::endl;
		else {
			//SIBR_LOG << "Embree device and scene created" << std::endl;
			//SIBR_LOG << "Warning Backface culling state : "<< rtcGetDeviceProperty(*g_device, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED) << std::endl;
			return true; // Success
		}
		return false; // Fail
	}

	Raycaster::geomId	Raycaster::addMesh(const sibr::Mesh& mesh)
	{
		return addGenericMesh(mesh, RTC_BUILD_QUALITY_HIGH);
	}

	Raycaster::geomId	Raycaster::addDynamicMesh(const sibr::Mesh& mesh)
	{
		return addGenericMesh(mesh, RTC_BUILD_QUALITY_LOW);
	}

	Raycaster::geomId	Raycaster::addGenericMesh(const sibr::Mesh& mesh, RTCBuildQuality type)
	{
		if (init() == false)
			return Raycaster::InvalidGeomId;

		const sibr::Mesh::Vertices& vertices = mesh.vertices();
		const sibr::Mesh::Triangles& triangles = mesh.triangles();

		RTCGeometry geom_0 = rtcNewGeometry(*g_device.get(), RTC_GEOMETRY_TYPE_TRIANGLE); // EMBREE_FIXME: check if geometry gets properly committed
		rtcSetGeometryBuildQuality(geom_0, type);
		rtcSetGeometryTimeStepCount(geom_0, 1);
		geomId id = rtcAttachGeometry(*_scene.get(), geom_0);
		//rtcReleaseGeometry(geom_0);

		if (id == Raycaster::InvalidGeomId) {
			rtcCommitGeometry(geom_0);
			return Raycaster::InvalidGeomId;
		}

		struct Vertex { float x, y, z, a; };
		struct Triangle { int v0, v1, v2; };

		{ // Fill vertices of the geometry
			Vertex* vert = (Vertex*)rtcSetNewGeometryBuffer(geom_0, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 4 * sizeof(float), vertices.size());
			for (uint i = 0; i < mesh.vertices().size(); ++i)
			{
				vert[i].x = vertices[i][0];
				vert[i].y = vertices[i][1];
				vert[i].z = vertices[i][2];
				vert[i].a = 1.f;
			}

		}

		{ // Fill triangle indices of the geometry
			Triangle* tri = (Triangle*)rtcSetNewGeometryBuffer(geom_0, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(int), triangles.size());
			for (uint i = 0; i < triangles.size(); ++i)
			{
				tri[i].v0 = triangles[i][0];
				tri[i].v1 = triangles[i][1];
				tri[i].v2 = triangles[i][2];
			}

		}

		rtcCommitGeometry(geom_0);

		// Commit all changes on the scene
		rtcCommitScene(*_scene.get());

		return id;
	}

	// xform a mesh by transformation matrix "mat". Note that the original positions
	// are always stored in mesh.vertices -- we only xform the vertices in the embree buffer
	void	Raycaster::xformRtcMeshOnly(sibr::Mesh& mesh, geomId mesh_id, sibr::Matrix4f& mat, sibr::Vector3f& centerPt, float& maxlen)
	{
		struct Vertex { float x, y, z, a; };
		Vertex* vert = (Vertex*)rtcGetGeometryBufferData(rtcGetGeometry(*_scene.get(), mesh_id), RTC_BUFFER_TYPE_VERTEX, 0) /* EMBREE_FIXME: check if this should be rtcSetNewGeometryBuffer */;
		sibr::Vector4f averagePt = sibr::Vector4f(0, 0, 0, 1);
		maxlen = 0;

		const sibr::Mesh::Vertices& vertices = mesh.vertices();
		//const sibr::Mesh::Normals& normals = mesh.normals();
		for (uint i = 0; i < mesh.vertices().size(); ++i)
		{
			sibr::Vector4f v;

			// reset to original position
			v[0] = vert[i].x = vertices[i][0];
			v[1] = vert[i].y = vertices[i][1];
			v[2] = vert[i].z = vertices[i][2];
			v[3] = vert[i].a = 1.f;

			v = mat * v;
			vert[i].x = v[0], vert[i].y = v[1], vert[i].z = v[2];
			averagePt += v;
			float d = sibr::Vector3f(sibr::Vector4f(averagePt / (float)((i == 0) ? 1 : i)).xyz() - v.xyz()).norm();
			if (d > maxlen)
				maxlen = d;
		}

		sibr::Vector4f cp = averagePt / (float)mesh.vertices().size();
		centerPt = sibr::Vector3f(cp[0], cp[1], cp[2]);

		// Update mesh
		rtcCommitGeometry(rtcGetGeometry(*_scene.get(), mesh_id));
		// Commit changes to scene
		rtcCommitScene(*_scene.get());
	}

	bool	Raycaster::hitSomething(const Ray& inray, float minDist)
	{
		assert(minDist >= 0.f);

		RTCRay ray;
		ray.flags = 0;
		ray.org_x = inray.orig()[0];
		ray.org_y = inray.orig()[1];
		ray.org_z = inray.orig()[2];
		ray.dir_x = inray.dir()[0];
		ray.dir_y = inray.dir()[1];
		ray.dir_z = inray.dir()[2];

		ray.tnear = minDist;
		ray.tfar = RayHit::InfinityDist;

		if (init() == false)
			SIBR_ERR << "cannot initialize embree, failed cast rays." << std::endl;
		else
		{
			RTCIntersectContext context;
			rtcInitIntersectContext(&context);
			rtcOccluded1(*_scene.get(), &context, &ray);
		}
		return ray.tfar < 0.0f;
	}

	std::array<bool, 8>	Raycaster::hitSomething8(const std::array<Ray, 8> & inray, float minDist)
	{
		assert(minDist >= 0.f);

		RTCRay8 ray;
		for (int r = 0; r < 8; r++) {
			ray.org_x[r] = inray[r].orig()[0];
			ray.org_y[r] = inray[r].orig()[1];
			ray.org_z[r] = inray[r].orig()[2];
			ray.dir_x[r] = inray[r].dir()[0];
			ray.dir_y[r] = inray[r].dir()[1];
			ray.dir_z[r] = inray[r].dir()[2];

			ray.tnear[r] = minDist;
			ray.tfar[r] = RayHit::InfinityDist;
		}

		int valid8[8] = { -1,-1,-1,-1, -1, -1, -1, -1 };
		if (init() == false)
			SIBR_ERR << "cannot initialize embree, failed cast rays." << std::endl;
		else
		{
			RTCIntersectContext context;
			rtcInitIntersectContext(&context);
			rtcOccluded8(valid8, *_scene.get(), &context, &ray);
		}

		std::array<bool, 8> res;
		for (int r = 0; r < 8; r++) {
			bool hit = (ray.tfar[r] < 0.0f );
			res[r] = hit;
		}

		return res;
	}

	RayHit	Raycaster::intersect(const Ray& inray, float minDist)
	{
		assert(minDist >= 0.f);

		RTCRayHit rh;
		rh.ray.flags = 0;
		rh.ray.org_x = inray.orig()[0];
		rh.ray.org_y = inray.orig()[1];
		rh.ray.org_z = inray.orig()[2];
		rh.ray.dir_x = inray.dir()[0];
		rh.ray.dir_y = inray.dir()[1];
		rh.ray.dir_z = inray.dir()[2];

		rh.ray.tnear = minDist;
		rh.ray.tfar = RayHit::InfinityDist;
		rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;

		if (init() == false)
			SIBR_ERR << "cannot initialize embree, failed cast rays." << std::endl;
		else
		{
			RTCIntersectContext context;
			rtcInitIntersectContext(&context);
			rtcIntersect1(*_scene.get(), &context, &rh);
			rh.hit.Ng_x = -rh.hit.Ng_x; // EMBREE_FIXME: only correct for triangles,quads, and subdivision surfaces
			rh.hit.Ng_y = -rh.hit.Ng_y;
			rh.hit.Ng_z = -rh.hit.Ng_z;
		}

		// Convert to the RayHit struct (used for abstract embree)

		RayHit::Primitive prim;
		prim.geomID = rh.hit.geomID;
		prim.instID = rh.hit.instID[0];
		prim.triID = rh.hit.primID;

		RayHit::BCCoord coord;
		coord.u = rh.hit.u;
		coord.v = rh.hit.v;

		sibr::Vector3f normal = sibr::Vector3f(rh.hit.Ng_x, rh.hit.Ng_y, rh.hit.Ng_z);

		// Return the result.
		return RayHit(inray, rh.ray.tfar, coord, normal, prim);
	}

	std::array<RayHit, 8>	Raycaster::intersect8(const std::array<Ray, 8> & inray, const std::vector<int> & valid8, float minDist)
	{
		assert(minDist >= 0.f);

		RTCRayHit8 rh;
		for (int r = 0; r < 8; r++) {
			rh.ray.org_x[r] = inray[r].orig()[0];
			rh.ray.org_y[r] = inray[r].orig()[1];
			rh.ray.org_z[r] = inray[r].orig()[2];
			rh.ray.dir_x[r] = inray[r].dir()[0];
			rh.ray.dir_y[r] = inray[r].dir()[1];
			rh.ray.dir_z[r] = inray[r].dir()[2];

			rh.ray.tnear[r] = minDist;
			rh.ray.tfar[r] = RayHit::InfinityDist;
			rh.hit.geomID[r] = RTC_INVALID_GEOMETRY_ID;
		}

		if (init() == false)
			SIBR_ERR << "cannot initialize embree, failed cast rays." << std::endl;
		else
		{
			RTCIntersectContext context;
			rtcInitIntersectContext(&context);
			rtcIntersect8(valid8.data(), *_scene.get(), &context, &rh);
		}

		std::array<RayHit, 8> res;
		for (int r = 0; r < 8; r++) {
			if (valid8[r])
				res[r] = {
					inray[r],
					rh.ray.tfar[r],
					RayHit::BCCoord{
						rh.hit.u[r],rh.hit.v[r]
					},
					sibr::Vector3f(rh.hit.Ng_x[r], rh.hit.Ng_y[r], rh.hit.Ng_z[r]),
					RayHit::Primitive{
#ifdef SIBR_OS_WINDOWS
						(uint)rh.hit.primID[r] ,(uint)rh.hit.geomID[r],(uint)rh.hit.instID[r]
#else
						// Considering RTC_MAX_INSTANCE_LEVEL_COUNT to be 1 (Single-level instancing); see https://www.embree.org/api.html#rtchit
						(uint)rh.hit.primID[r] ,(uint)rh.hit.geomID[r],(uint)rh.hit.instID[0][r]
#endif
					
					}
				};
		}
		return res;
	}

	void Raycaster::clearGeometry()
	{
		_scene.reset();
	}

	sibr::Vector3f Raycaster::smoothNormal(const sibr::Mesh& mesh, const RayHit& hit)
	{
		if (!mesh.hasNormals()) {
			SIBR_ERR << " cannot compute smoothed normals if the mesh does not have normals " << std::endl;
		}
		const sibr::Mesh::Normals& normals = mesh.normals();
		const sibr::Vector3u& tri = mesh.triangles()[hit.primitive().triID];

		const float ucoord = hit.barycentricCoord().u;
		const float vcoord = hit.barycentricCoord().v;
		float wcoord = 1.f - ucoord - vcoord;
		wcoord = (wcoord >= 0.0f ? (wcoord <= 1.0f ? wcoord : 1.0f) : 0.0f);

		return (wcoord * normals[tri[0]] + ucoord * normals[tri[1]] + vcoord * normals[tri[2]]).normalized();
	}

	sibr::Vector3f Raycaster::smoothColor(const sibr::Mesh& mesh, const RayHit& hit)
	{
		if (!mesh.hasColors()) {
			SIBR_ERR << " cannot compute smoothed color if the mesh does not have colors " << std::endl;
		}
		const sibr::Mesh::Colors& colors = mesh.colors();
		const sibr::Vector3u& tri = mesh.triangles()[hit.primitive().triID];

		const float ucoord = hit.barycentricCoord().u;
		const float vcoord = hit.barycentricCoord().v;
		float wcoord = 1.f - ucoord - vcoord;
		wcoord = (wcoord >= 0.0f ? (wcoord <= 1.0f ? wcoord : 1.0f) : 0.0f);

		return wcoord * colors[tri[0]] + ucoord * colors[tri[1]] + vcoord * colors[tri[2]];
	}

	sibr::Vector2f Raycaster::smoothUV(const sibr::Mesh& mesh, const RayHit& hit)
	{
		if (!mesh.hasTexCoords()) {
			SIBR_ERR << " cannot compute UV if the mesh does not have texcoords " << std::endl;
		}
		const sibr::Mesh::UVs& uvs = mesh.texCoords();
		const sibr::Vector3u& tri = mesh.triangles()[hit.primitive().triID];

		const float ucoord = hit.barycentricCoord().u;
		const float vcoord = hit.barycentricCoord().v;
		float wcoord = 1.f - ucoord - vcoord;
		wcoord = (wcoord >= 0.0f ? (wcoord <= 1.0f ? wcoord : 1.0f) : 0.0f);

		return wcoord * uvs[tri[0]] + ucoord * uvs[tri[1]] + vcoord * uvs[tri[2]];
	}

} // namespace sibr
