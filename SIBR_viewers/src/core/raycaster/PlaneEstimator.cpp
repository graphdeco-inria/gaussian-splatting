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


#include "PlaneEstimator.hpp"
#include <random>

typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb;

PlaneEstimator::PlaneEstimator() {}

PlaneEstimator::PlaneEstimator(const std::vector<sibr::Vector3f> & vertices, bool excludeBB)
{

	Eigen::AlignedBox<float, 3> boxScaled;
	if (excludeBB) {
		Eigen::AlignedBox<float, 3> box;
		for (const auto & vertex : vertices) {
			box.extend(vertex);
		}
		for (const auto & vertex : vertices) {
			boxScaled.extend(box.center()+0.99f*(vertex- box.center()));
		}

	}
	int bboxReject = 0;
	if (vertices.size() > 200000) {
		std::cout << "Found more than 200000 points reducing point cloud size ..." << std::endl;

		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		for (const auto & v : vertices) {
			double random = dist(mt);
			if (random < 200000.0 / double(vertices.size())) {
				if (!excludeBB || (boxScaled.exteriorDistance(v)==0) )
					_Points.push_back(v);
				else if (excludeBB && boxScaled.exteriorDistance(v) > 0) {
					bboxReject++;
				}
			}
		}

		if (excludeBB)
			std::cout << bboxReject << " points where rejected becaused considered on the bounding box" << std::endl;
	}
	else {
		_Points = vertices;
	}
	std::cout << "Point Cloud size: " << _Points.size() << std::endl;
	_numPoints3D = (int)_Points.size();
	_remainPoints3D.resize(_Points.size(), 3);
	_remainNormals3D.resize(_Points.size(), 3);

	for (int i = 0; i < _Points.size(); i++) {
		_remainPoints3D.row(i) = _Points[i];
		_remainNormals3D.row(i) = sibr::Vector3f(0, 0, 0);

	}

	_planeComputed = false;
}

void PlaneEstimator::computePlanes(const int numPlane, const  float delta, const int numTry) {


	_planeComputed = true; // we know that the planes were computed

	std::cout << "Original number of points " << _remainPoints3D.rows() << std::endl;
	for (int i = 0; i < numPlane; i++) {

		if (_remainPoints3D.rows() < _numPoints3D * 5 / 100)
		{
			std::cout << "Not enough points remaining, stop searching. " << i << " planes found." << std::endl;
			break;
		}


		sibr::Vector3f color(((float)rand() / (RAND_MAX)), ((float)rand() / (RAND_MAX)), ((float)rand() / (RAND_MAX)));

		Eigen::MatrixXi mask;
		//Eigen::MatrixXf maskNormals;
		std::pair<Eigen::MatrixXf, sibr::Vector3f> covMean;
		int vote = -1;
		sibr::Vector4f plane = estimatePlane(delta, numTry, mask, vote, covMean);

		if (vote < _numPoints3D * 2 / 100 && i >= 12) {
			std::cout << "Not enough points in candidate plane, stop searching. " << i << " planes found." << std::endl;
			break;
		}
		//

		Eigen::MatrixXf remainPoints3DTemp(_remainPoints3D.rows() - vote, 3);
		Eigen::MatrixXf remainNormals3DTemp(_remainNormals3D.rows() - vote, 3);
		//std::vector<sibr::Vector3i> remainImPosTemp;

		int notSel = 0;
		std::vector<sibr::Vector3f> pointsPlane;
		for (int rIt = 0; rIt < _remainPoints3D.rows(); rIt++) {
			if (mask.row(rIt)(0) == 0) { // not selected
				remainPoints3DTemp.row(notSel) = _remainPoints3D.row(rIt);
				remainNormals3DTemp.row(notSel) = _remainNormals3D.row(rIt);
				//remainImPosTemp.push_back(_remainImPos[rIt]);
				notSel++;
			}

			else { // In the plane
				pointsPlane.push_back(_remainPoints3D.row(rIt));
			}
		}

		std::cout << "vote :" << vote << " notSel " << notSel << " supposed total " << _remainPoints3D.rows() << std::endl;
		_remainPoints3D = remainPoints3DTemp;
		_remainNormals3D = remainNormals3DTemp;
		//_remainImPos=remainImPosTemp;
		std::cout << "Remaining number of points " << _remainPoints3D.rows() << std::endl;

		sibr::Vector3f center = plane.w()*plane.xyz();

		sibr::Vector4f finalPlane = plane;

		_planes.push_back(finalPlane);
		_points.push_back(pointsPlane);
		// centers and basis
		_centers.push_back(center);

		//plane statistical informations
		_covMeans.push_back(covMean);
		_votes.push_back(vote);
	}

}

sibr::Vector4f PlaneEstimator::estimatePlane(const float delta, const int numTry, Eigen::MatrixXi & bestMask, int & bestVote, std::pair<Eigen::MatrixXf, sibr::Vector3f> & bestCovMean) {

	sibr::Vector4f bestPlane;

	float bestWVote = 0;
#pragma omp parallel for
	for (int i = 0; i < numTry; i++) {

		Eigen::MatrixXi mask;
		sibr::Vector4f plane = plane3Pts();
		if (plane.xyz().norm() > 0) {
			std::pair<int, float> votePair = votePlane(plane, delta, mask);

#pragma omp critical
			{
				//std::cout << i << " ";
				if (votePair.second > bestWVote) {
					bestWVote = votePair.second;
					bestVote = votePair.first;
					bestPlane = plane;
					bestMask = std::move(mask); // move to avoid copy
				}
			}
		}

	}

	std::cout << "Best vote " << bestVote << " Best plane " << bestPlane << std::endl;
	/*
	std::cout << "Plane refinement ..." << std::endl;

	////////////////// Plane fitting

	Eigen::MatrixXf data(3, bestVote);
	int sel = 0;

	for (int rIt = 0; rIt < _remainPoints3D.rows(); rIt++) {
		if (bestMask.row(rIt)(0) == 1) {
			data.col(sel) = _remainPoints3D.row(rIt);
			sel++;
		}
	}


	std::cout << "Sel " << sel << std::endl;

	sibr::Vector3f center = data.rowwise().mean();
	Eigen::MatrixXf dataCentered = data.colwise() - center;

	bestCovMean.first = (dataCentered*dataCentered.adjoint()) / float(dataCentered.cols() - 1);
	bestCovMean.second = center;

	std::cout << "Cov Matrix : " << bestCovMean.first << " Mean : " << bestCovMean.second << std::endl;
	std::cout << "Cov Determinant : " << bestCovMean.first.determinant() << std::endl;

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(dataCentered, Eigen::ComputeFullU | Eigen::ComputeThinV);

	std::cout << "old normal" << " " << bestPlane.xyz();
	//the normal to the fitting plane is the eigenvector associated to the smallest eigenvalue (i.e. the direction in which the variance of all points is the smallest) 
	sibr::Vector3f normal = svd.matrixU().col(2);
	normal.normalize();

	float d = center.dot(normal);

	bestPlane = sibr::Vector4f(normal.x(), normal.y(), normal.z(), d);

	bestVote = votePlane(bestPlane, 10.0*delta, bestMask, 0.8f).first;*/

	std::cout << " new normal" << " " << bestPlane.xyz() << std::endl;
	// normal coherency
	//Eigen::ArrayXf dotWithOriNormal=(_remainNormals3D * bestPlane.xyz()).array().cwiseAbs();
	//bestMaskNormals = (bestMask.array().cast<float>()*dotWithOriNormal);

	std::cout << "Vote refined " << bestVote << " Plane refined " << bestPlane << /*" Normals coherency " << bestMaskNormals.sum()/bestVote  <<*/ std::endl;

	return bestPlane;

}


sibr::Vector4f PlaneEstimator::plane3Pts() {

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> dis(0, int(_remainPoints3D.rows() - 1));

	sibr::Vector3f pointA = _remainPoints3D.row(dis(gen));
	sibr::Vector3f pointB = _remainPoints3D.row(dis(gen));
	sibr::Vector3f pointC = _remainPoints3D.row(dis(gen));

	sibr::Vector3f normal = (pointB - pointA).cross(pointC - pointA);
	normal.normalize();

	float d = normal.dot(pointA);

	return sibr::Vector4f(normal.x(), normal.y(), normal.z(), d);

}

std::pair<int, float> PlaneEstimator::votePlane(const sibr::Vector4f plane, const float delta, Eigen::MatrixXi & mask, float normalDot) {

	sibr::Vector3f normal = plane.xyz();
	float d = plane.w();

	//std::cout << "size " << _points3D.size() << " " << normal.size() << " d " << d << std::endl;

	Eigen::ArrayXf distances = (_remainPoints3D * normal).array();

	Eigen::ArrayXf dotWithOriNormal = (_remainNormals3D * normal).array();

	/*for(int i=0; i< 10; i++){
	std::cout << distances.row(i) << " ";
	}
	std::cout << std::endl;*/

	distances = (distances - d * Eigen::ArrayXf::Ones(distances.rows())).cwiseAbs();
	dotWithOriNormal = dotWithOriNormal.cwiseAbs();

	/*for(int i=0; i< 10; i++){
	std::cout << distances.row(i) << " ";
	}
	std::cout << std::endl;*/
	mask = (distances < delta && (dotWithOriNormal > normalDot || dotWithOriNormal == 0)).cast<int>();
	
	Eigen::ArrayXf voteW = (distances+ 0.1f*delta* Eigen::ArrayXf::Ones(distances.rows()));
	voteW = mask.array().cast<float>().cwiseQuotient(voteW);
	//std::cout << "SUM " << mask.sum() << std::endl;

	return std::make_pair(mask.sum(), voteW.sum());

}

sibr::Vector4f PlaneEstimator::estimateGroundPlane(sibr::Vector3f roughUp)
{
	if (_planeComputed) {

		//find the floor plane
		int bestId = -1;
		int bestVote = 0;

		for (int p = 0; p < _planes.size(); p++) {
			if (abs(_planes[p].xyz().dot(roughUp)) > 0.87) {
				if (_votes[p] > bestVote) {
					bestId = p;
					bestVote = _votes[p];
				}
			}
		}
		return _planes[bestId];
	}
	else {
		std::cout << "Error : Plane not computed, you should call computePlanes first" << std::endl;
		SIBR_ERR;
		return { 0.0f, 0.0f, 0.0f, 0.0f };
	}
}


sibr::Vector3f PlaneEstimator::estimateMedianVec(const std::vector<sibr::Vector3f> & ups)
{

	std::vector<float> medUpX;
	std::vector<float> medUpY;
	std::vector<float> medUpZ;

	for (const auto & up : ups) {

		medUpX.push_back(up.x());
		medUpY.push_back(up.y());
		medUpZ.push_back(up.z());

	}
	std::sort(medUpX.begin(), medUpX.end());
	std::sort(medUpY.begin(), medUpY.end());
	std::sort(medUpZ.begin(), medUpZ.end());

	const size_t medPos = medUpX.size() / 2;

	sibr::Vector3f upMed(medUpX[medPos], medUpY[medPos], medUpZ[medPos]);
	upMed.normalize();

	return upMed;
}

sibr::Mesh PlaneEstimator::getMeshPlane(sibr::Vector4f plane, sibr::Vector3f center, float radius)
{
	sibr::Mesh planeMesh;

	sibr::Vector3f projCenter = center - (center - plane.w()*plane.xyz()).dot(plane.xyz())*plane.xyz();

	sibr::Mesh::Vertices vert;
	sibr::Mesh::Triangles tri;
	sibr::Mesh::Normals nml;
	sibr::Mesh::UVs tex;

	sibr::Vector3f u = (projCenter - plane.w()*plane.xyz()).normalized();
	sibr::Vector3f v = plane.xyz().cross(u).normalized();

	int numP = 50;
	for (int i = 0; i < numP; i++) {
		vert.push_back(projCenter + radius * cos(2 * M_PI*i / numP)*u + radius * sin(2 * M_PI*i / numP)*v);
		nml.push_back(plane.xyz().normalized());
		tri.push_back(sibr::Vector3u(numP, i, (i + 1) % numP));
	}

	vert.push_back(projCenter);

	planeMesh.vertices(vert);
	planeMesh.normals(nml);
	planeMesh.triangles(tri);
	return planeMesh;
}

void PlaneEstimator::displayPCAndPlane(sibr::Window::Ptr window)
{
}

PlaneEstimator::~PlaneEstimator(void)
{
}
