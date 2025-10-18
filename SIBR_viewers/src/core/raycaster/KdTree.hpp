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

#include "Config.hpp"

#include "core/system/Vector.hpp"
#include "nanoflann/nanoflann.hpp"

namespace sibr { 

	/**
	 * \class KdTree
	 * \brief Represent a 3D hierachical query structure baked by a nanoflann KdTree.
	 * \note With the default L2 distance, all distances and radii are expected to be 
	 * the squared values (this is a nanoflann constraint). For other metrics, use the distance directly.
	 * \ingroup sibr_raycaster
	 */
	template <typename num_t = double, class Distance = nanoflann::metric_L2>
	class  KdTree
	{
		SIBR_CLASS_PTR(KdTree);
	
	public:

		typedef	Eigen::Matrix<num_t, 3, 1, Eigen::DontAlign> Vector3X;
		typedef KdTree<num_t, Distance> self_t;
		typedef typename Distance::template traits<num_t, self_t>::distance_t metric_t;
		typedef nanoflann::KDTreeSingleIndexAdaptor< metric_t, self_t, 3, size_t>  index_t;
		typedef std::vector<std::pair<size_t, num_t>> Results;

		/**
		 * Constructor.
		 * The KdTree will do a copy of the positions vector.
		 * \param positions a list of 3D points
		 * \param leafMaxSize maximum number of points per leaf
		 */
		KdTree(const std::vector<Vector3X> & positions, size_t leafMaxSize = 10);

		/** Destructor. */
		~KdTree();

		/** Get the closest point stored in the KdTree for the specified distance
		* \param pos the reference point
		* \param distanceSq will contain the squared distance from pos to the closest point in the tree.
		* \return the index of the closest point in the tree.
		*/
		size_t getClosest(const Vector3X & pos, num_t & distanceSq) const;

		/** Get the closest point stored in the KdTree for the specified distance
		* \param pos the reference point
		* \param count the number of neighbours to query
		* \param idDistSqs will contain the indices of the closest points and their squared distances to the reference point
		*/
		void getClosest(const Vector3X & pos, size_t count, Results & idDistSqs) const;

		/** Get all points in a sphere of a given radius around a reference point.
		 *\param pos the reference point
		 *\param maxDistanceSq the squared sphere radius
		 *\param sorted should the points be sorted in ascending distance order
		 *\param idDistSqs will contain the indices of the points in the sphere and their squared distances to the reference point
		 */
		void getNeighbors(const Vector3X & pos, double maxDistanceSq, bool sorted, Results & idDistSqs) const;

		/// Interface expected by nanoflann for an adapter.
		const self_t & derived() const {
			return *this;
		}

		/// Interface expected by nanoflann for an adapter.
		self_t & derived() {
			return *this;
		}

		/// Interface: Must return the number of data points
		inline size_t kdtree_get_point_count() const {
			return _points.size();
		}

		/// Interface: Returns the dim'th component of the idx'th point in the class:
		inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const {
			return _points[idx][dim];
		}

		/// Interface: Optional bounding-box computation: \return false to default to a standard bbox computation loop.
		template <class BBOX>
		bool kdtree_get_bbox(BBOX & /*bb*/) const {
			return false;
		}

	private:

		const std::vector<Vector3X> _points;
		index_t * _index;
	};

	template <typename num_t, class Distance>
	KdTree<num_t, Distance>::KdTree(const std::vector<Vector3X>& positions, size_t leafMaxSize) : _points(positions) {
		if(positions.empty()) {
			SIBR_ERR << "[KdTree] Trying to build a Kd-Tree from an empty list of points." << std::endl;
		}
		_index = new index_t(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leafMaxSize));
		_index->buildIndex();
	}

	template<typename num_t, class Distance>
	KdTree<num_t, Distance>::~KdTree()
	{
		delete _index;
	}

	template <typename num_t, class Distance>
	inline size_t KdTree<num_t, Distance>::getClosest(const Vector3X& pos, num_t & distanceSq) const {
		size_t index = 0;
		_index->knnSearch(&pos[0], 1, &index, &distanceSq);
		return index;
	}

	template <typename num_t, class Distance>
	inline void KdTree<num_t, Distance>::getClosest(const Vector3X & pos, size_t count, Results & idDistSqs) const {
		std::vector<size_t> outIds(count);
		std::vector<num_t> outDists(count);
		const size_t foundCount = _index->knnSearch(&pos[0], count, &outIds[0], &outDists[0]);
		idDistSqs.resize(foundCount);
		for(size_t i = 0; i < foundCount; ++i) {
			idDistSqs[i] = std::make_pair(outIds[i], outDists[i]);
		}
	}

	template <typename num_t, class Distance>
	inline void KdTree<num_t, Distance>::getNeighbors(const Vector3X & pos, double maxDistanceSq, bool sorted, Results & idDistSqs) const {
		_index->radiusSearch(&pos[0], float(maxDistanceSq), idDistSqs, nanoflann::SearchParams(32, 0.0f, sorted));
	}


} /*namespace sibr*/ 

