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


#include "UVUnwrapper.hpp"
#include <core/system/SimpleTimer.hpp>
#include <core/graphics/Utils.hpp>
#include "xatlas.h"

int printCallback(const char * format, ...) {
	va_list args;
	va_start(args, format);
	std::cout << "\r";
	const int res = vprintf(format, args);
	va_end(args);
	return res;
}

bool progressCallback(xatlas::ProgressCategory category, int progress, void *userData){
	std::cout << "\r\t" << xatlas::StringForEnum(category) << "[" << std::flush;
	for (int i = 0; i < 10; i++)
		std::cout << (progress / ((i + 1) * 10) ? "*" : " ");
	std::cout << "] " << progress << "%" << std::flush;
	if(progress == 100) {
		std::cout << std::endl;
	}
	return true;
}

void setPixel(uint8_t *dest, int destWidth, int x, int y, const sibr::Vector3ub & color){
	uint8_t *pixel = &dest[x * 3 + y * (destWidth * 3)];
	pixel[0] = color[0];
	pixel[1] = color[1];
	pixel[2] = color[2];
}

// https://github.com/miloyip/line/blob/master/line_bresenham.c
// License: public domain.
static void rasterizeLine(uint8_t *dest, int destWidth, const int *p1, const int *p2, const sibr::Vector3ub & color)
{
	const int dx = abs(p2[0] - p1[0]), sx = p1[0] < p2[0] ? 1 : -1;
	const int dy = abs(p2[1] - p1[1]), sy = p1[1] < p2[1] ? 1 : -1;
	int err = (dx > dy ? dx : -dy) / 2;
	int current[2];
	current[0] = p1[0];
	current[1] = p1[1];
	while(setPixel(dest, destWidth, current[0], current[1], color), current[0] != p2[0] || current[1] != p2[1]){
		const int e2 = err;
		if (e2 > -dx) { err -= dy; current[0] += sx; }
		if (e2 < dy) { err += dx; current[1] += sy; }
	}
}

/*
https://github.com/ssloy/tinyrenderer/wiki/Lesson-2:-Triangle-rasterization-and-back-face-culling
Copyright Dmitry V. Sokolov

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
void rasterizeTriangle(uint8_t *dest, int destWidth, const int *t0, const int *t1, const int *t2, const sibr::Vector3ub & color)
{
	if (t0[1] > t1[1]) std::swap(t0, t1);
	if (t0[1] > t2[1]) std::swap(t0, t2);
	if (t1[1] > t2[1]) std::swap(t1, t2);
	const int total_height = t2[1] - t0[1];
	for (int i = 0; i < total_height; i++) {
		const bool second_half = i > t1[1] - t0[1] || t1[1] == t0[1];
		const int segment_height = second_half ? t2[1] - t1[1] : t1[1] - t0[1];
		const float alpha = (float)i / total_height;
		const float beta = (float)(i - (second_half ? t1[1] - t0[1] : 0)) / float(segment_height);
		int A[2], B[2];
		for (int j = 0; j < 2; j++) {
			A[j] = int(t0[j] + (t2[j] - t0[j]) * alpha);
			B[j] = int(second_half ? t1[j] + (t2[j] - t1[j]) * beta : t0[j] + (t1[j] - t0[j]) * beta);
		}
		if (A[0] > B[0]) std::swap(A, B);
		for (int j = A[0]; j <= B[0]; j++) {
			setPixel(dest, destWidth, j, t0[1] + i, color);
		}
	}
}

using namespace sibr;

UVUnwrapper::UVUnwrapper(const sibr::Mesh& mesh, unsigned int res) : _mesh(mesh) {
	_size = res;
	// Create empty atlas.
	xatlas::SetPrint(printCallback, false);
	_atlas = xatlas::Create();
	xatlas::SetProgressCallback(_atlas, progressCallback, nullptr);

	// Add the mesh to the atlas.
	SIBR_LOG << "[UVMapper] Adding one mesh with " << mesh.vertices().size() << " vertices and " << mesh.triangles().size() << " triangles." << std::endl;
	// For now consider everything as one mesh. Splitting in components *might* help.
	xatlas::MeshDecl meshDecl;
	meshDecl.vertexCount = uint32_t(mesh.vertices().size());
	meshDecl.vertexPositionData = mesh.vertexArray();
	meshDecl.vertexPositionStride = sizeof(sibr::Vector3f);
	if (mesh.hasNormals()) {
		meshDecl.vertexNormalData = mesh.normalArray();
		meshDecl.vertexNormalStride = sizeof(sibr::Vector3f);
	}
	// UV can be used as a hint.
	if (mesh.hasTexCoords()) {
		meshDecl.vertexUvData = mesh.texCoordArray();
		meshDecl.vertexUvStride = sizeof(sibr::Vector2f);
	}
	meshDecl.indexCount = uint32_t(mesh.triangles().size() * 3);
	meshDecl.indexData = mesh.triangleArray();
	meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
	const xatlas::AddMeshError error = xatlas::AddMesh(_atlas, meshDecl, 1);
	if (error != xatlas::AddMeshError::Success) {
		xatlas::Destroy(_atlas);
		SIBR_ERR << "\r[UVMapper] Error adding mesh: " << xatlas::StringForEnum(error) << std::endl;
	}
	// Not necessary. Only called here so geometry totals are printed after the AddMesh progress indicator
	xatlas::AddMeshJoin(_atlas);
}

	
sibr::Mesh::Ptr UVUnwrapper::unwrap() {

	// Generate atlas.
	SIBR_LOG << "[UVMapper] Generating atlas.." << std::endl;

	xatlas::ChartOptions chartOptions = xatlas::ChartOptions();
	xatlas::PackOptions packOptions = xatlas::PackOptions();
	packOptions.bruteForce = false;
	packOptions.resolution = uint32_t(_size);
	Timer timer;
	timer.tic();
	xatlas::Generate(_atlas, chartOptions, packOptions);

	SIBR_LOG << "[UVMapper] Generation took: " << timer.deltaTimeFromLastTic<Timer::s>() << "s." << std::endl;
	SIBR_LOG << "[UVMapper] Output resolution: " << _atlas->width << "x" << _atlas->height << std::endl;
	SIBR_LOG << "[UVMapper] Generated " << _atlas->chartCount << " charts, " << _atlas->atlasCount << " atlases." << std::endl;
	for (uint32_t i = 0; i < _atlas->atlasCount; i++) {
		SIBR_LOG << "[UVMapper] \tAtlas " << i << ": utilisation: " << _atlas->utilization[i] * 100.0f << "%" << std::endl;
	}


	uint32_t totalVertices = 0;
	uint32_t totalFaces = 0;
	for (uint32_t i = 0; i < _atlas->meshCount; i++) {
		const xatlas::Mesh& xmesh = _atlas->meshes[i];
		totalVertices += xmesh.vertexCount;
		totalFaces += xmesh.indexCount / 3;
	}
	SIBR_LOG << "[UVMapper] Output geometry data: " << totalVertices << " vertices, " << totalFaces << " triangles." << std::endl;
	// Write meshes.
	uint32_t firstVertex = 0;
	std::vector<sibr::Vector3f> positions;
	std::vector<sibr::Vector3f> normals;
	std::vector<sibr::Vector2f> texcoords;
	std::vector<sibr::Vector3f> colors;
	std::vector<sibr::Vector3u> triangles;
	
	// We could preallocate and paraellize if needed.
	for (uint32_t i = 0; i < _atlas->meshCount; i++) {
		const xatlas::Mesh& xmesh = _atlas->meshes[i];
		for (uint32_t v = 0; v < xmesh.vertexCount; v++) {
			const xatlas::Vertex& vertex = xmesh.vertexArray[v];
			const sibr::Vector3f& pos = _mesh.vertices()[vertex.xref];
			positions.emplace_back(pos);
			if (_mesh.hasNormals()) {
				const sibr::Vector3f& n = _mesh.normals()[vertex.xref];
				normals.emplace_back(n);
			}
			if (_mesh.hasColors()) {
				const sibr::Vector3f& c = _mesh.colors()[vertex.xref];
				colors.emplace_back(c);
			}
			
			_mapping.emplace_back(vertex.xref);
			texcoords.emplace_back(vertex.uv[0] / float(_atlas->width), vertex.uv[1] / float(_atlas->height));
		}
		for (uint32_t f = 0; f < xmesh.indexCount; f += 3) {
			const uint32_t i0 = firstVertex + xmesh.indexArray[f + 0];
			const uint32_t i1 = firstVertex + xmesh.indexArray[f + 1];
			const uint32_t i2 = firstVertex + xmesh.indexArray[f + 2];
			triangles.emplace_back(i0, i1, i2);
		}
		firstVertex += xmesh.vertexCount;
	}
	Mesh::Ptr finalMesh(new Mesh(false));
	finalMesh->vertices(positions);
	finalMesh->normals(normals);
	finalMesh->texCoords(texcoords);
	finalMesh->colors(colors);
	finalMesh->triangles(triangles);

	SIBR_LOG << "[UVMapper] Done." << std::endl;
	return finalMesh;
}

const std::vector<uint>& UVUnwrapper::mapping() const
{
	return _mapping;
}


std::vector<ImageRGB::Ptr> UVUnwrapper::atlasVisualization() const {
	if(!_atlas || _atlas->width <= 0 || _atlas->height <= 0) {
		SIBR_WRG << "[UVMapper] Atlas has not been created/processed." << std::endl;
		return {};
	}

	SIBR_LOG << "[UVMapper] Rasterizing result maps..." << std::endl;
	
	// Rasterize unwrapped meshes.
	// \todo port to SIBR image API.
	std::vector<uint8_t> outputChartsImage;
	const uint32_t imageDataSize = _atlas->width * _atlas->height * 3;
	outputChartsImage.resize(_atlas->atlasCount * imageDataSize);
	for (uint32_t i = 0; i < _atlas->meshCount; i++) {
		const xatlas::Mesh &xmesh = _atlas->meshes[i];
		const sibr::Vector3ub white = { 255, 255, 255 };
		// Rasterize mesh charts.
		for (uint32_t j = 0; j < xmesh.chartCount; j++) {
			const xatlas::Chart *chart = &xmesh.chartArray[j];
			const sibr::Vector3ub color = sibr::randomColor<unsigned char>();
			for (uint32_t k = 0; k < chart->faceCount; k++) {
				int verts[3][2];
				for (int l = 0; l < 3; l++) {
					const xatlas::Vertex &v = xmesh.vertexArray[xmesh.indexArray[chart->faceArray[k] * 3 + l]];
					verts[l][0] = int(v.uv[0]);
					verts[l][1] = int(v.uv[1]);
				}
				uint8_t *imageData = &outputChartsImage[chart->atlasIndex * imageDataSize];
				rasterizeTriangle(imageData, _atlas->width, verts[0], verts[1], verts[2], color);
				rasterizeLine(imageData, _atlas->width, verts[0], verts[1], white);
				rasterizeLine(imageData, _atlas->width, verts[1], verts[2], white);
				rasterizeLine(imageData, _atlas->width, verts[2], verts[0], white);
			}
		}
	}
	
	// Convert raw vectors to images.
	std::vector<ImageRGB::Ptr> views(_atlas->meshCount);
	for (uint32_t i = 0; i < _atlas->meshCount; i++) {
		views[i].reset(new ImageRGB(_atlas->width, _atlas->height));
		uint8_t *imageData = &outputChartsImage[i * imageDataSize];
#pragma omp parallel for
		for(int y = 0; y < int(_atlas->height); ++y) {
			for(int x = 0; x < int(_atlas->width); ++x) {
				const size_t baseId = (y * _atlas->width + x)*3;
				for(int j = 0; j < 3; ++j) {
					views[i](x, y)[j] = imageData[baseId + j];
				}		
			}
		}
		views[i]->flipH();
	}
	return views;
}

UVUnwrapper::~UVUnwrapper() {
	xatlas::Destroy(_atlas);
}

