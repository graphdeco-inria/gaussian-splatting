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


#include "VoxelGrid.hpp"

namespace sibr {

	VoxelGridBase::VoxelGridBase(const Box & boundingBox, int n, bool forceCube)
		: VoxelGridBase(boundingBox, sibr::Vector3i(n, n, n), forceCube)
	{
	}

	VoxelGridBase::VoxelGridBase(const Box & boundingBox, const sibr::Vector3i & numsPerDim, bool forceCube)
		: box(boundingBox), dims(numsPerDim), _generator(0), _distribution(-1.0, 1.0)
	{
		if (forceCube) {
			float maxDimSize = box.sizes().cwiseQuotient(dims.cast<float>()).maxCoeff();
			for (int c = 0; c < 3; ++c) {
				dims[c] = (int)std::round(box.sizes()[c] / maxDimSize);
			}
		}
		cellSize = box.sizes().cwiseQuotient(dims.cast<float>());
		cellSizeNorm = cellSize.norm();

		//std::cout << dims << std::endl;
		//std::cout << getCellSize() << std::endl;

		static const sibr::Mesh::Triangles trianglesBorders =
		{
			{ 0,4, 4 },
		{ 5,1, 1 },
		{ 4,5, 5 },
		{ 0,1, 1 },
		{ 2,6, 6 },
		{ 7,3, 3 },
		{ 6,7, 7 },
		{ 2,3, 3 },
		{ 0,2, 2 },
		{ 1,3, 3 },
		{ 4,6, 6 },
		{ 5,7, 7 }
		};


		Box baseCell;
		baseCell.extend(box.min());
		baseCell.extend(box.min() + getCellSize());


		sibr::Mesh::Vertices vs(8);
		for (int i = 0; i < 8; ++i) {
			vs[i] = baseCell.corner((Box::CornerType)i);
		}

		baseCellMesh.reset(new sibr::Mesh(false));
		baseCellMesh->vertices(vs);
		baseCellMesh->triangles(trianglesBorders);

		static const sibr::Mesh::Triangles trianglesFilled =
		{
			{ 0,1,5 },
		{ 0,5,4 },
		{ 1,3,7 },
		{ 1,7,5 },
		{ 3,2,6 },
		{ 3,6,7 },
		{ 2,0,4 },
		{ 2,4,6 },
		{ 0,2,3 },
		{ 0,3,1 },
		{ 4,5,7 },
		{ 4,7,6 }
		};


		baseCellMeshFilled.reset(new sibr::Mesh(false));
		baseCellMeshFilled->vertices(vs);
		baseCellMeshFilled->triangles(trianglesFilled);
	}

	bool VoxelGridBase::isInside(const sibr::Vector3f & worldPos) const
	{
		return box.contains(worldPos);
	}

	bool VoxelGridBase::outOfBounds(const sibr::Vector3i & v) const
	{
		return (v.array() < 0).any() || (v.array() >= dims.array()).any();
		//return v[0] < 0 || v[0] >= dims[0] || v[1] < 0 || v[1] >= dims[1] || v[2] < 0 || v[2] >= dims[2];
	}

	size_t VoxelGridBase::getNumCells() const
	{
		return (size_t)dims.prod();
	}

	const sibr::Vector3i & VoxelGridBase::getDims() const
	{
		return dims;
	}

	sibr::Vector3i VoxelGridBase::getCell(size_t cellId) const
	{
		if (cellId >= getNumCells()) {
			SIBR_ERR;
		}

		sibr::Vector3i cell;
	
		std::div_t div;
		for (int i = 0; i < 2; ++i) {
			div = std::div((int)cellId, dims[i]);
			cell[i] = div.rem;
			cellId = div.quot;
		}
		cell[2] = (int)cellId;

		if (outOfBounds(cell)) {
			SIBR_ERR << cell << " " << dims;
		}

		//if ((cell.array() < 0).any() || (cell.array() >= dims.array()).any()) {
		//	SIBR_ERR;
		//}

		return cell;
	}

	sibr::Vector3i VoxelGridBase::getCell(const sibr::Vector3f & worldPos) const
	{
		sibr::Vector3f posUV = (worldPos - box.min()).cwiseQuotient(box.sizes());
		sibr::Vector3i cellCoord = (dims.cast<float>().cwiseProduct(posUV)).unaryExpr([](float f) { return std::floor(f); }).cast<int>();

		if ((cellCoord.array() < 0).any() || (cellCoord.array() >= dims.array()).any()) {
			SIBR_ERR;
		}

		return cellCoord;
	}

	sibr::Vector3i VoxelGridBase::getCellInclusive(const sibr::Vector3f & worldPos) const
	{
		sibr::Vector3f posUV = (worldPos - box.min()).cwiseQuotient(box.sizes());
		sibr::Vector3i cellCoord = (dims.cast<float>().cwiseProduct(posUV).unaryExpr([](float f) { return std::floor(f); })).cast<int>();

		//because of the floor function, a pixel exactly at the boundary would be outside
		for (int c = 0; c < 3; c++) {
			if (cellCoord[c] == -1){
				++cellCoord[c];
			}
			if (cellCoord[c] == dims[c]) {
				--cellCoord[c];
			}
		}

		if ((cellCoord.array() < 0).any() || (cellCoord.array() >= dims.array()).any()) {
			SIBR_ERR << worldPos << " " << box.min() << " " << box.max() << " " << cellCoord;
		}

		return cellCoord;
	}

	std::vector<size_t> VoxelGridBase::rayMarch(const Ray & ray) const
	{
		sibr::Vector3f start = ray.orig();

		if (!isInside(start)) {
			sibr::Vector3f intersection;
			if (intersectionWithBox(ray, intersection)) {
				start = intersection;
			} else {
				return {};
			}
		}
		
		start = start.cwiseMax(box.min()).cwiseMin(box.max() - 0.01f*getCellSize());

		sibr::Vector3i currentVoxel = getCell(start);
	
		sibr::Vector3i steps = ray.dir().unaryExpr([](float f) { return f >= 0 ? 1 : -1; }).cast<int>();

		const sibr::Vector3f deltas = getCellSize().cwiseQuotient(ray.dir().cwiseAbs());
		const sibr::Vector3f frac = (start - box.min()).cwiseQuotient(getCellSize()).unaryExpr([](float f) { return f - std::floor(f); });
		sibr::Vector3i finalVoxels; 
		sibr::Vector3f ts;
		for (int c = 0; c < 3; c++) {
			ts[c] = deltas[c] * (ray.dir()[c] >= 0 ? 1.0f - frac[c] : frac[c]);
			finalVoxels[c] = (ray.dir()[c] >= 0 ? dims[c] : -1);
		}

		std::vector<size_t> visitedCellsIds;
		while (true) {
			visitedCellsIds.push_back(getCellId(currentVoxel));
		
			int c = getMinIndex(ts);
			currentVoxel[c] += steps[c];
			if (currentVoxel[c] == finalVoxels[c]) {
				break;
			}
			ts[c] += deltas[c];
		}

		return visitedCellsIds;
	}

	sibr::Mesh::Ptr VoxelGridBase::getCellMesh(const sibr::Vector3i & cell) const
	{
		return getCellMeshInternal(cell, false);
	}

	sibr::Mesh::Ptr VoxelGridBase::getAllCellMesh() const
	{
		return getAllCellMeshInternal(false);
	}

	sibr::Mesh::Ptr VoxelGridBase::getCellMeshFilled(const sibr::Vector3i & cell) const
	{
		return getCellMeshInternal(cell, true);
	}

	sibr::Mesh::Ptr VoxelGridBase::getAllCellMeshFilled() const
	{
		return getAllCellMeshInternal(true);
	}

	Eigen::AlignedBox3f VoxelGridBase::getCellBox(size_t cellId) const
	{
		sibr::Vector3i cell = getCell(cellId);
		sibr::Vector3f center = getCellCenter(cell);
		sibr::Vector3f half_diagonal = 0.5f*getCellSize();

		Eigen::AlignedBox3f out;
		out.extend(center - half_diagonal);
		out.extend(center + half_diagonal);
		return out;
	}

	std::vector<size_t> VoxelGridBase::getNeighbors(size_t cellId) const
	{
		static const sibr::Vector3f shifts[6] = {
			{-1, 0, 0}, { +1 ,0, 0 },
			{0, -1, 0},{0, +1, 0},
			{0, 0, -1},{0, 0, +1}
		};

		sibr::Vector3f pos = getCellCenter(cellId);

		std::vector<size_t> n_ids;
		for (int i = 0; i < 6; ++i) {
			sibr::Vector3f n_pos = pos + shifts[i].cwiseProduct(getCellSize());
			if (getBBox().contains(n_pos)) {
				n_ids.push_back(getCellId(n_pos));
			}
		}
		return n_ids;
	}

	VoxelGridBase VoxelGridBase::extend(int numCells) const
	{
		sibr::Vector3f additionalSize = ((float)numCells)*getCellSize();
		//sibr::Vector3f half_diagonal = 0.5*box.diagonal() + additionalSize;

		Box extendedBox;
		extendedBox.extend(box.max() + additionalSize);
		extendedBox.extend(box.min() - additionalSize);

		//extendedBox.extend(box.center() + (additionalSize.norm() + 0.5f*box.diagonal().norm()) *box.diagonal().normalized());
		//extendedBox.extend(box.center() - (additionalSize.norm() + 0.5f*box.diagonal().norm()) *box.diagonal().normalized());
		
		VoxelGridBase extendedGrid = VoxelGridBase(extendedBox, dims.array() + 2*numCells);
		return extendedGrid;
	}

	bool VoxelGridBase::intersectionWithBox(const Ray & ray, sibr::Vector3f & intersection) const
	{
		//adpated from https://github.com/papaboo/smalldacrt/

		sibr::Vector3f minTs = (box.min() - ray.orig()).cwiseQuotient(ray.dir());
		sibr::Vector3f maxTs = (box.max() - ray.orig()).cwiseQuotient(ray.dir());

		float nearT = (minTs.cwiseMin(maxTs)).maxCoeff();
		float farT = (minTs.cwiseMax(maxTs)).minCoeff();

		if (nearT <= farT && 0 <= nearT) {
			intersection = ray.orig() + nearT*ray.dir();
			return true;
		} 
		return false;	
	}

	const sibr::Vector3f & VoxelGridBase::getCellSize() const
	{
		return cellSize;
	}

	float VoxelGridBase::getCellSizeNorm() const
	{
		return cellSizeNorm;
	}

	sibr::Vector3f VoxelGridBase::sampleCell(size_t cellId)
	{
		sibr::Vector3f out;
		out[0] = float(_distribution(_generator));
		out[1] = float(_distribution(_generator));
		out[2] = float(_distribution(_generator));
		return getCellCenter(getCell(cellId)) + out.cwiseProduct(getCellSize());
	}

	sibr::Mesh::Ptr VoxelGridBase::getCellMeshInternal(const sibr::Vector3i & cell, bool filled) const
	{
		sibr::Mesh::Ptr baseMesh = filled ? baseCellMeshFilled : baseCellMesh;

		const sibr::Vector3f offset = cell.cast<float>().array()*getCellSize().array();

		auto out = std::make_shared<sibr::Mesh>(true);
		out->triangles(baseMesh->triangles());
		sibr::Mesh::Vertices vs(8);
		for (int i = 0; i < 8; ++i) {
			vs[i] = baseMesh->vertices()[i] + offset;
		}
		out->vertices(vs);
		return out;
	}

	sibr::Mesh::Ptr VoxelGridBase::getAllCellMeshInternal(bool filled) const
	{
		auto out = std::make_shared<sibr::Mesh>();

		sibr::Mesh::Ptr baseMesh = filled ? baseCellMeshFilled : baseCellMesh;

		const int numT = int(baseMesh->triangles().size());
		const int numTtotal = int(getNumCells())*numT;
		const int numV = int(baseMesh->vertices().size());
		const int numVtotal = int(getNumCells())*numV;
		const sibr::Vector3u offsetT = sibr::Vector3u(numV, numV, numV);

		sibr::Mesh::Vertices vs(numVtotal);
		sibr::Mesh::Triangles ts(numTtotal);
		for (int i = 0; i < getNumCells(); ++i) {
			const auto cell = getCell(i);
			const sibr::Vector3f offsetV = cell.cast<float>().array()*getCellSize().array();

			for (int v = 0; v < numV; ++v) {
				vs[i*numV + v] = baseMesh->vertices()[v] + offsetV;
			}
			for (int t = 0; t < numT; ++t) {
				ts[i*numT + t] = baseMesh->triangles()[t] + i * offsetT;
			}
		}

		out->vertices(vs);
		out->triangles(ts);
		return out;
	}

	size_t VoxelGridBase::getCellId(const sibr::Vector3i & v) const
	{
		if (outOfBounds(v)) {
			SIBR_ERR << v << " " << dims;
		}
		return v[0] + dims[0] * (v[1] + dims[1] * v[2]); //v[2] + dims[2] * (v[1] + dims[1] * v[0]);
	}

	size_t VoxelGridBase::getCellId(const sibr::Vector3f & world_pos) const
	{
		return getCellId(getCell(world_pos));
	}

	sibr::Vector3f VoxelGridBase::getCellCenter(const sibr::Vector3i & cell) const
	{
		return box.min() + (0.5f*sibr::Vector3f(1, 1, 1) + cell.cast<float>()).cwiseProduct(getCellSize());
	}

	sibr::Vector3f VoxelGridBase::getCellCenter(size_t cellId) const
	{
		return getCellCenter(getCell(cellId));
	}

	int VoxelGridBase::getMinIndex(const sibr::Vector3f & v)
	{
		if (v.x() < v.y()) {
			return v.x() < v.z() ? 0 : 2;
		} else {
			return v.y() < v.z() ? 1 : 2;
		}
	}

	sibr::Vector3f orthoVector(const sibr::Vector3f & v)
	{
		return std::abs(v[2]) < std::abs(v[0]) ? sibr::Vector3f(v[1], -v[0], 0) : sibr::Vector3f(0, -v[2], v[1]);
	}

}