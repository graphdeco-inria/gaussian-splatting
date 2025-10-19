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

#include <core/raycaster/Config.hpp>

#include <core/system/Utils.hpp>
#include <core/system/Vector.hpp>
#include <core/graphics/Image.hpp>
#include <core/system/Array2d.hpp>
#include <core/graphics/Mesh.hpp>
#include <core/graphics/Window.hpp>


/**
	Fit a plane to a point cloud using an improved RANSAC approach.
	\ingroup sibr_raycaster
*/
class SIBR_RAYCASTER_EXPORT PlaneEstimator {

public:

	/// Default constructor.
	PlaneEstimator();

	/** Constructor.
	\param vertices the point cloud
	\param excludeBB if true, reject points that are close to the vertices bounding box
	*/
	PlaneEstimator(const std::vector<sibr::Vector3f> & vertices, bool excludeBB=false);

	/** Compute one or more planes fitting the data using RANSAC. Points that are well fitted by a plan will bre moved from the set.
	\param numPlane number of planes to fit
	\param delta fit validity threshold
	\param numTry number of attempts to perform for each plane
	*/
	void computePlanes(const int numPlane,const float delta,const int numTry);
	
	/** Estimate the best plane in the remaining points set using RANSAC.
	\param delta fit validity threshold
	\param numTry number of attempts to perform for each plane
	\param bestMask for each point, will be set to 1 if the plane explains the point well
	\param vote will contain the number of points that fit
	\param bestCovMean unused
	\return the plane parameters
	*/
	sibr::Vector4f estimatePlane(const float delta,const int numTry, Eigen::MatrixXi & bestMask, int & vote, std::pair<Eigen::MatrixXf,sibr::Vector3f> & bestCovMean);
	
	/** Choose randomly 3 points among the vertices and compute the corresponding plane.
	\return the plane parameters
	*/
	sibr::Vector4f plane3Pts(); 

	/** Given a plane and a threshold, this function return the num of point that fit the plane in the remaining points and also the associated mask.
	\param plane the plane parameters
	\param delta validity threshold
	\param mask for each point, will be set to 1 if the plane explains the point well
	\param normalDot normal validity threshold
	\return number of points that fit and overall weighted score (based on normal similarity)
	*/
	std::pair<int, float> votePlane(const sibr::Vector4f plane, const float delta, Eigen::MatrixXi & mask, float normalDot=0.98);
	
	/** For visualization, display the point cloud and fitted plane in a window.
	\param window the windo to use for display
	\deprecated Empty, won't do anything.
	*/
	void displayPCAndPlane(sibr::Window::Ptr window);

	/** Estimate a fitting plane that is as orthogonal to the given up vector as possible.
	\param roughUp an estimation of the scene up vector
	\return the plane parameters.
	*/
	sibr::Vector4f estimateGroundPlane(sibr::Vector3f roughUp);
	
	/** Estimate the scene zenith from a set of camera up vectors (assuming photogrametric capture).
	\param ups a set of up vector
	\return the estimated median zenith vector
	*/
	static sibr::Vector3f estimateMedianVec(const std::vector<sibr::Vector3f> & ups);

	/** Generate a mesh representing a plane.
	\param plane the parameters of the plane to represent
	\param center center of the plane mesh
	\param radius extent of the plane mesh
	\return the generated plane mesh
	*/
	static sibr::Mesh getMeshPlane(sibr::Vector4f plane , sibr::Vector3f center, float radius);

	std::vector<sibr::Vector3f> _Points; ///< All initial points.
	int _numPoints3D; ///< Number of initial points.

	std::vector<sibr::Vector4f> _planes; ///< Planes are represented as Vector4f(n.x,n.y,n.z,d)
	std::vector<std::vector<sibr::Vector3f>> _points; ///< For each plane, list of fitting points.
	std::vector<sibr::Vector3f> _centers; ///< Plane centers.
	std::vector<int> _votes; ///< Number of votes per plane.
	std::vector<std::pair<Eigen::MatrixXf,sibr::Vector3f>> _covMeans; ///< Unused.
	
	/// Destructor.
	~PlaneEstimator(void);

protected:

	Eigen::MatrixXf _remainPoints3D; ///< Points to consider.
	Eigen::MatrixXf _remainNormals3D; ///< Associated normals to consider.
	std::vector<sibr::Vector3u> _Triangles; ///< Triangle list.
	bool _planeComputed; ///< Has the plane been computed.
};

