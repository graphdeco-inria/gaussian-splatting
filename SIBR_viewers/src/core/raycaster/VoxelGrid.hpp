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


# include <vector>
#include <random>

# include <core/raycaster/Config.hpp>

#include <core/raycaster/Ray.hpp>
#include <core/graphics/Mesh.hpp>

namespace sibr
{
	/**
	\addtogroup sibr_raycaster
	@{
	*/

	/** Generate a vector orthogonal to the input one.
	\param v the input vector
	\return the ortogonal vector
	\warning The output vector is not necessarily of unit length.
	*/
	sibr::Vector3f orthoVector(const sibr::Vector3f & v);

	/** Default voxel type, storing binary occupancy. */
	struct BasicVoxelType {

		/** \return true if filled. */
		operator bool() const { return used; }
		
		bool used = true; ///< The voxel status.
	};

	/** Basic voxel grid interface provinding cell manipulation and query helpers. 
	It doesn't store any voxel data. */
	class SIBR_RAYCASTER_EXPORT VoxelGridBase {
		SIBR_CLASS_PTR(VoxelGridBase);

	public:

		typedef Eigen::AlignedBox<float, 3> Box;

		/** Constructor.
		\param boundingBox bounding box delimiting the voxellized region
		\param numPerDim number of voxels along each dimension
		\param forceCube if true, the largest dimension will be split in numPerDim voxels and the other such that the voxels are cubes in world space
		*/
		VoxelGridBase(const Box & boundingBox, int numPerDim, bool forceCube = true);


		/** Constructor.
		\param boundingBox bounding box delimiting the voxellized region
		\param numsPerDim number of voxels along each dimension
		\param forceCube if true, the largest dimension will be split in numPerDims voxels and the other such that the voxels are cubes in world space
		*/
		VoxelGridBase(const Box & boundingBox, const sibr::Vector3i & numsPerDim, bool forceCube = true);

		/** Check if a position is in the voxel grid.
		\param worldPos the world 3D position
		\return true if the position is in the grid bounding box
		*/
		bool isInside(const sibr::Vector3f & worldPos) const;

		/** Check if a set of indices correspond to a reachable voxel.
		\param cell the voxel integer coordinates
		\return true if the voxel exists in the grid
		*/
		bool outOfBounds(const sibr::Vector3i & cell) const;

		/** \return the number of voxels. */
		size_t getNumCells() const;

		/** \return the number of voxels along each axis. */
		const sibr::Vector3i & getDims() const;

		/** Convert a linear cell ID to a set of 3D indices.
		\param cellId linear ID
		\return the indices of the voxel along each axis
		*/
		sibr::Vector3i getCell(size_t cellId) const;

		/** Convert a voxel 3D indices to a linear ID.
		\param cell the voxel integer coordinates
		\return the linear ID
		*/
		size_t getCellId(const sibr::Vector3i & cell) const;

		/** Convert a 3D position to the linear ID of the voxel containing it.
		\param world_pos the position
		\return the linear ID of the voxel
		*/
		size_t getCellId(const sibr::Vector3f & world_pos) const;

		/** Get the position of a voxel center in world space.
		\param cell the voxel integer coordinates
		\return the center 3D position
		*/
		sibr::Vector3f getCellCenter(const sibr::Vector3i & cell) const;

		/** Get the position of a voxel center in world space.
		\param cellId linear voxel ID
		\return the center 3D position
		*/
		sibr::Vector3f getCellCenter(size_t cellId) const;

		/** Intersect a ray with the voxel grid, listing all intersected voxels.
		\param ray the ray to cast
		\return linear IDs of the intersected voxels
		*/
		std::vector<size_t> rayMarch(const Ray & ray) const;

		/** Generate a wireframe mesh representing a voxel.
		\param cell the voxel integer coordinates
		\return the generated wireframe cube mesh
		*/
		sibr::Mesh::Ptr getCellMesh(const sibr::Vector3i & cell) const;

		/** Generate a wireframe mesh representing all voxels.
		\return the generated wireframe cube mesh
		*/
		sibr::Mesh::Ptr getAllCellMesh() const;

		/** Generate a triangle mesh representing a voxel.
		\param cell the voxel integer coordinates
		\return the generated filled cube mesh
		*/
		sibr::Mesh::Ptr getCellMeshFilled(const sibr::Vector3i & cell) const;

		/** Generate a triangle mesh representing all voxels.
		\return the generated filled cube mesh
		*/
		sibr::Mesh::Ptr getAllCellMeshFilled() const;

		/** Get a voxel bounding box.
		\param cellId the voxel linear index
		\return the bounding box.
		*/
		Eigen::AlignedBox3f getCellBox(size_t cellId) const;

		/** Get a voxel neighbors linear IDs.
		\param cellId linear voxel ID
		\return the linear IDs of the neigbors.
		*/
		std::vector<size_t> getNeighbors(size_t cellId) const;

		/** Extend the voxel grid along all dimensions. 
		This means that if the initial count along a given axis was N, the new is N+2*numCells.
		\param numCells the number of cells to add
		\return the extended voxel grid
		*/
		VoxelGridBase extend(int numCells) const;

		/** \return the voxel grid bounding box. */
		const Box & getBBox() const { return box; }

		/** Return the index of the smallest coefficient of the input vector.
		\param v the vector
		\return the location of the minimum
		*/
		static int getMinIndex(const sibr::Vector3f & v);
		
		/** Get the integer coordinates of the cell containing a position.
		\param worldPos the position
		\return the cell integer coordinates
		*/
		sibr::Vector3i getCell(const sibr::Vector3f & worldPos) const;

		/** Get the integer coordinates of the cell containing a position.
		Positions along the boundaries of the voxel grid are considered as belonging to the closest cell.
		\param worldPos the position
		\return the cell integer coordinates
		*/
		sibr::Vector3i getCellInclusive(const sibr::Vector3f & worldPos) const;

		/** Check if a ray intersect the voxel grid.
		\param ray the ray to cast
		\param intersection will contain the intersection position if it exists
		\return true if there is an intersection
		*/
		bool intersectionWithBox(const Ray & ray, sibr::Vector3f & intersection) const;

		/** \return the size of a voxel. */
		const sibr::Vector3f & getCellSize() const;

		/** \return the length of a voxel diagonal. */
		float getCellSizeNorm() const;

		/** Sample a random position in a given voxel.
		\param cellId the voxel to sample from
		\return the sampled position
		\note The random generator is seeded at 0 when creating the grid.
		\warning The current implementation is sampling in center+(random(-1,1)^3)*cellSize.
		*/
		sibr::Vector3f sampleCell(size_t cellId);

	protected:

		/** Helper to generate a voxel mesh.
		\param cell the coordinates of the voxel to generate
		\param filled should the mesh be wireframe (false) or faceted (true)
		\return the generated mesh
		*/
		sibr::Mesh::Ptr getCellMeshInternal(const sibr::Vector3i & cell, bool filled) const;

		/** Helper to generate the voxel grid mesh.
		\param filled should the mesh be wireframe (false) or faceted (true)
		\return the generated mesh
		*/
		sibr::Mesh::Ptr getAllCellMeshInternal(bool filled) const;

		sibr::Vector3i dims; ///< Integer grid dimensions.
		sibr::Vector3f cellSize; ///< World space voxel size.
		float cellSizeNorm; ///< World space voxel diagonal length.
		Box box; ///< Grid bounding box.
		sibr::Mesh::Ptr baseCellMesh, baseCellMeshFilled; ///< Base meshes for visualisation.

		std::mt19937 _generator; ///< Generator for sampling, seeded at 0.
		std::uniform_real_distribution<double> _distribution; ///< (-1,1) distribution.

	};


	/** Voxel grid with custom data storage. */
	template<typename CellType = BasicVoxelType> class VoxelGrid : public VoxelGridBase {

		SIBR_CLASS_PTR(VoxelGrid);
	public:
		using VoxelType = CellType;

	public:

		/** Constructor.
		\param boundingBox bounding box delimiting the voxellized region
		\param numPerDim number of voxels along each dimension
		\param forceCube if true, the largest dimension will be split in numPerDim voxels and the other such that the voxels are cubes in world space
		*/
		VoxelGrid(const Box & boundingBox, int numPerDim, bool forceCube = true)
			: VoxelGrid(boundingBox, sibr::Vector3i(numPerDim, numPerDim, numPerDim) , forceCube)
		{
		}

		/** Constructor.
		\param boundingBox bounding box delimiting the voxellized region
		\param numsPerDim number of voxels along each dimension
		\param forceCube if true, the largest dimension will be split in numPerDim voxels and the other such that the voxels are cubes in world space
		*/
		VoxelGrid(const Box & boundingBox, const sibr::Vector3i & numsPerDim, bool forceCube = true)
		: VoxelGridBase(boundingBox, numsPerDim, forceCube) {
			data.resize(getNumCells());
		}

		/** Get voxel at a given linear index.
		\param cell_id the linear index
		\return a reference to the voxel
		*/
		CellType & operator[](size_t cell_id) {
			return data[cell_id];
		}

		/** Get voxel at a given linear index.
		\param cell_id the linear index
		\return a reference to the voxel
		*/
		const CellType & operator[](size_t cell_id) const {
			return data[cell_id];
		}

		/** Get voxel at given integer 3D coordinates.
		\param x x integer coordinate
		\param y y integer coordinate
		\param z z integer coordinate
		\return a reference to the voxel
		*/
		CellType & operator()(int x, int y, int z) {
			sibr::Vector3i v(x,y,z);
			return data[getCellId(v)];
		}

		/** Get voxel at given integer 3D coordinates.
		\param x x integer coordinate
		\param y y integer coordinate
		\param z z integer coordinate
		\return a reference to the voxel
		*/
		const CellType & operator()(int x, int y, int z) const {
			sibr::Vector3i v(x,y,z);
			return data[getCellId(v)];
		}

		/** Get voxel at given integer 3D coordinates.
		\param v integer coordinates
		\return a reference to the voxel
		*/
		CellType & operator[](const sibr::Vector3i & v) {
			return data[getCellId(v)];
		}

		/** Get voxel at given integer 3D coordinates.
		\param v integer coordinates
		\return a reference to the voxel
		*/
		const CellType & operator[](const sibr::Vector3i & v) const {
			return data[getCellId(v)];
		}

		/** Generate a mesh from all voxels satisfying a condition.
		\param filled should the mesh be wireframe (false) or faceted (true)
		\param func the predicate to evaluate, will receive as unique argument a voxel (CellType).
		\return the generated mesh
		*/
		template<typename FuncType>
		sibr::Mesh::Ptr getAllCellMeshWithCond(bool filled, const FuncType & func) const;

		/** Get cell meshes from their ids.
		\param filled should the mesh be wireframe (false) or faceted (true)
		\param cell_ids ids of cell meshes.
		\return the generated mesh
		*/
		sibr::Mesh::Ptr getAllCellMeshWithIds(bool filled, std::vector<std::size_t> cell_ids) const;

		/** List the voxels that statisfy a condition (for instance fullness)
		\param func the predicate to evaluate, will receive as unique argument a voxel (CellType).
		\return a list of linear indices of all voxels such that func(voxel) is true.
		*/
		template<typename FuncType>
		std::vector<std::size_t> detect_non_empty_cells(const FuncType & func) const;

		/** \return the voxel grid data. */
		const std::vector<CellType> & getData() const {
			return data;
		}

	protected:

		std::vector<CellType> data; ///< Voxels storage.
	};



	template<typename CellType> template<typename FuncType>
	inline std::vector<std::size_t> VoxelGrid<CellType>::detect_non_empty_cells(const FuncType & func) const {
		std::vector<std::size_t> out_ids;
		for (size_t i = 0; i < data.size(); ++i) {
			if (func(data[i])) {
				out_ids.push_back(i);
			}
		}
		return out_ids;
	}

	template<typename CellType> template<typename FuncType>
	inline sibr::Mesh::Ptr VoxelGrid<CellType>::getAllCellMeshWithCond(bool filled, const FuncType & f) const
	{
		std::vector<std::size_t> cell_ids = detect_non_empty_cells(f);
		return getAllCellMeshWithIds(filled, cell_ids);
	}

	/** }@ */

	template<typename CellType>
	inline sibr::Mesh::Ptr VoxelGrid<CellType>::getAllCellMeshWithIds(bool filled, std::vector<std::size_t> cell_ids) const
	{
		int numNonZero = (int)cell_ids.size();

		auto out = std::make_shared<sibr::Mesh>();

		sibr::Mesh::Ptr baseMesh = filled ? baseCellMeshFilled : baseCellMesh;

		const int numT = (int)baseMesh->triangles().size();
		const int numTtotal = numNonZero * numT;
		const int numV = (int)baseMesh->vertices().size();
		const int numVtotal = numNonZero * numV;
		const sibr::Vector3u offsetT = sibr::Vector3u(numV, numV, numV);

		sibr::Mesh::Vertices vs(numVtotal);
		sibr::Mesh::Triangles ts(numTtotal);
		for (int i = 0; i < numNonZero; ++i) {
			const auto cell = getCell(cell_ids[i]);
			const sibr::Vector3f offsetV = cell.cast<float>().array() * getCellSize().array();

			for (int v = 0; v < numV; ++v) {
				vs[i * numV + v] = baseMesh->vertices()[v] + offsetV;
			}
			for (int t = 0; t < numT; ++t) {
				ts[i * numT + t] = baseMesh->triangles()[t] + i * offsetT;
			}
		}

		out->vertices(vs);
		out->triangles(ts);
		return out;
	}

} // namespace sibr

