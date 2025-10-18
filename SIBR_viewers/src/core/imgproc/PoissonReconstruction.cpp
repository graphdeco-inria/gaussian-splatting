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


#include "PoissonReconstruction.hpp"
#include <queue>   
#include <Eigen/Sparse>


namespace sibr {



PoissonReconstruction::PoissonReconstruction(
	const cv::Mat3f & gradientsX,
	const cv::Mat3f & gradientsY,
	const cv::Mat1f & mask,
	const cv::Mat3f & img_target)
{
	// Make a copy of the target as we are going to modify it.
	_img_target = img_target.clone();
	_gradientsX = gradientsX;
	_gradientsY = gradientsY;
	_mask = mask;
	
}

void PoissonReconstruction::solve(void)
{
	parseMask();

	//solve Ai X=bi , Ai = A : coefs , bi : b_terms , i for each RGB
	std::vector< Eigen::Triplet<double> >  coefs;
	std::vector<Eigen::VectorXd> b_terms;

	for (int k = 0; k < 3; ++k) {
		b_terms.push_back(Eigen::VectorXd::Zero(_pixels.size()));
	}

	for ( int p=0; p<(int)_pixels.size(); p++ ) { 
		sibr::Vector2i pos(_pixels[p]);
		std::vector< sibr::Vector2i >  nPos ( getNeighbors(pos, _img_target.cols, _img_target.rows ));
		int num_neighbors = 0;
		cv::Vec3f new_term(0, 0, 0);

		for( int n_id = 0; n_id<nPos.size(); n_id++){ 
			sibr::Vector2i npos(nPos[n_id]);

			int nId = _pixelsId[npos.x() + _mask.cols * npos.y()];
			if( nId < -1 ) { continue; }
			++num_neighbors;	

			if( isInMask(npos) ) { //pair inside mask
				if(nId < 0 ) { std::cerr << "#"; }
				coefs.push_back(Eigen::Triplet<double>(p,nId,-1));
									
				// Four possibilities:
				if(npos.x() > pos.x()){ // right pixel
					new_term -= _gradientsY.at<cv::Vec3f>(pos.y(), pos.x());
				} else if (npos.x() < pos.x()){ // left pixel
					new_term += _gradientsY.at<cv::Vec3f>(npos.y(), npos.x());
				} else if (npos.y() > pos.y()){ // bottom pixel
					new_term -= _gradientsX.at<cv::Vec3f>(pos.y(), pos.x());
				} else if(npos.y() < pos.y()){ // top pixel
					new_term += _gradientsX.at<cv::Vec3f>(npos.y(), npos.x());
				} 

			} else if(!isIgnored(npos)) { //boundary
				new_term += _img_target.at<cv::Vec3f>(npos.y(),npos.x()); // color of target
				
			}
		}

		coefs.push_back(Eigen::Triplet<double>(p,p,(double)num_neighbors)); 

		for (int k = 0; k < 3; ++k) {
			b_terms[k](p) = new_term(k);
		}
			
	}

	Eigen::SparseMatrix<double> A((int)_pixels.size(),(int)_pixels.size());
	A.setFromTriplets(coefs.begin(),coefs.end());
	
	std::vector<Eigen::VectorXd> solutions;
	Eigen::SimplicialLDLT< Eigen::SparseMatrix<double> > eigenSolver;

	eigenSolver.compute(A);

	if(eigenSolver.info()!=Eigen::Success) {
		std::cerr << "decomp = failure" <<std::endl;
		return;
	} 

	for (int k = 0; k < 3; ++k) {
		solutions.push_back(eigenSolver.solve(b_terms[k]));
		if (eigenSolver.info() != Eigen::Success) {
			std::cerr << "decomp = failure" << std::endl;
		}

		float error = (float)(A*solutions[k] - b_terms[k]).squaredNorm();
		if (error > 1) {
			std::cerr << "distance to solution: " << error << std::endl;
		}
	}

	for (int p = 0; p<(int)_pixels.size(); p++) {
		sibr::Vector2i pos(_pixels[p]);
		cv::Vec3f color;
		for (int k = 0; k < 3; ++k) {
			color(k) = std::min(1.0f, std::max((float)solutions[k][p], 0.0f));
		}
		_img_target.at<cv::Vec3f>(pos.y(), pos.x()) = color;
	}

	postProcessing();
	postProcessing();
	
}

void PoissonReconstruction::parseMask( void )
{
	_pixels.resize(0);
	_boundaryPixels.resize(0);
	_pixelsId.resize(_mask.rows*_mask.cols,-2);

	//std::cerr << "size : " <<  _mask.cols << " x " << _mask.rows << std::endl;
	
	//first find boundaries
	for( int j=0; j<(int)_mask.rows; j++ ) {
		for( int i=0; i<(int)_mask.cols; i++) { 
			sibr::Vector2i pos(i,j);

			if(isIgnored(pos)) {
				continue;
			}
			if( !isInMask(pos) ) { 
				std::vector< sibr::Vector2i > neighbors = getNeighbors(pos, _img_target.cols, _img_target.rows);
				for( int n_id = 0; n_id<neighbors.size(); n_id++){ //if at least one neighbor is in mask, considered as boundary
					sibr::Vector2i npos(neighbors[n_id]);
					if( isInMask(npos) && !isIgnored(npos)) {
						_pixelsId[i+_mask.cols*j] = -1;
						_boundaryPixels.push_back(pos);
						break;
					}
				}
			} else {
				_pixelsId[i+_mask.cols*j] = 0;
			}

		}
	}

	checkConnectivity();

	//then find all valid pixels to edit
	for( int i=0; i<(int)_mask.cols; i++) { 
		for( int j=0; j<(int)_mask.rows; j++ ) {
			sibr::Vector2i pos(i, j);
			if( isIgnored(pos) ||  _pixelsId[i+_mask.cols*j] < 0 ) { 
				continue;
			} else {	
				_pixelsId[i+_mask.cols*j] = (int)_pixels.size();
				_pixels.push_back(sibr::Vector2i(i,j));				
			}

		}
	}

}

std::vector< sibr::Vector2i > PoissonReconstruction::getNeighbors( sibr::Vector2i pos, int width, int height )
{
	std::vector< sibr::Vector2i > output;
	int offset_list_4[4][2] = {	{0,1},{0,-1},{1,0},{-1,0} };

	for( int i=0; i<4; i++){
		sibr::Vector2i n_pos( pos[0]+offset_list_4[i][0], pos[1]+offset_list_4[i][1]);
		int x = n_pos.x();
		int y = n_pos.y();
		if( x>=0 && x< width && y>=0 && y< height) {
			output.push_back(n_pos);
		}
	}
	return output;
}

void PoissonReconstruction::computeGradients(const cv::Mat3f& src, cv::Mat3f& gradX, cv::Mat3f& gradY) {
	gradX = cv::Mat3f(src.size());
	gradY = cv::Mat3f(src.size());
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			// Compute forward differences.
			const int ip = std::min(i + 1, src.rows - 1);
			const int jp = std::min(j + 1, src.cols - 1);
			
			const cv::Vec3f c = src.at<cv::Vec3f>(i, j);
			const cv::Vec3f d = src.at<cv::Vec3f>(ip, j);
			const cv::Vec3f r = src.at<cv::Vec3f>(i, jp);
			const cv::Vec3f dX = d - c;
			const cv::Vec3f dY = r - c;
			gradX.at<cv::Vec3f>(i, j) = dX;
			gradY.at<cv::Vec3f>(i, j) = dY;
		}
	}
}

void PoissonReconstruction::checkConnectivity( void )
{
	// R(x) : 0 -> not connected to boundary, 1 -> connected, G(y) : 0 -> not checked, 1 -> checked
	sibr::ImageRGB connectivity( _mask.cols , _mask.rows);
	
	std::queue<sibr::Vector2i> pixelsToCheck;
	for(int p=0; p<(int)_boundaryPixels.size(); p++){
		sibr::Vector2i pos(_boundaryPixels[p]);
		pixelsToCheck.push(pos);
		connectivity(pos.x(),pos.y()).x() = 1; //boundaries are connected to boundaries
	}

	//propagate connectivity
	while(pixelsToCheck.size()>0){
		sibr::Vector2i pos(pixelsToCheck.front()); 
		pixelsToCheck.pop();
		connectivity(pos.x(),pos.y()).y() = 1;

		std::vector<sibr::Vector2i> neighbors( getNeighbors(pos, _img_target.cols, _img_target.rows) );
		for(int n_id=0; n_id<(int)neighbors.size(); n_id++){
			sibr::Vector2i npos(neighbors[n_id]); 
			if( connectivity(npos.x(),npos.y()).y()==1 || isIgnored(npos)) { 
				continue;
			} else {
				connectivity(npos.x(),npos.y()).x() = 1;
				connectivity(npos.x(),npos.y()).y() = 1;
				pixelsToCheck.push( npos );
			}
		}
	}

	//discard non connected pixel
	for(int i=0; i<(int)_mask.cols; i++){
		for(int j=0; j<(int)_mask.rows; j++){
			if( connectivity(i,j).x() == 0 ) {
				_pixelsId[i + _mask.cols * j] = -2;
				sibr::Vector2i coords(i,j);
				if( isInMask(coords) && !isIgnored(coords) ) {
					_img_target.at<cv::Vec3f>(j, i) = cv::Vec3f(0.0f,0.0f,0.0f);
				}
			}
		}
	}

}

void PoissonReconstruction::postProcessing(void)
{
	//std::cerr << "[PoissonRecons] Post Processing" << std::endl;
	/*
	 * for( int j=0; j<(int)_mask.rows; j++ ) {
		for( int i=0; i<(int)_mask.cols; i++) { 
			sibr::Vector2i pos(i,j);
			if( !isInMask(pos) ) { 
	 */
#pragma omp parallel for
	for (int j = 0; j < (int)_mask.rows; j++) {
		for (int i = 0; i < (int)_mask.cols; i++) {
			// mask: 0 -> in reconstruction, 1 -> keep fixed.
			//return (_mask.at<float>(pos.y(), pos.x())<0.5);
			if (std::abs(_mask.at<float>(j,i)) < 0.5f && cv::norm(_img_target.at<cv::Vec3f>(j,i)) == 0.0f) {
				std::vector<sibr::Vector2i> neighbors(getNeighbors(sibr::Vector2i(i, j), _mask.cols, _mask.rows));
				std::vector<bool> neighIsBlack(neighbors.size(), false);
				int black_neighbor = false;
				for (uint n_id = 0; n_id < neighbors.size() && !black_neighbor; n_id++) {
					sibr::Vector2i npos(neighbors[n_id]);
					if (cv::norm(_img_target.at<cv::Vec3f>(npos.y(), npos.x())) == 0.0f) {
						neighIsBlack[n_id] = true;
					}
				}
				if (!black_neighbor && neighbors.size() > 0) {
					cv::Vec3f new_color(0, 0, 0);
					int count = 0;
					for (uint n_id = 0; n_id < neighbors.size(); n_id++) {
						if(neighIsBlack[n_id]) {
							continue;
						}
						sibr::Vector2i npos(neighbors[n_id]);
						new_color += _img_target.at<cv::Vec3f>(npos.y(), npos.x());
						++count;
					}
					_img_target.at<cv::Vec3f>(j,i) = ((1.0f / (float)count)*new_color);
				}
			}
		}
	}
}

bool PoissonReconstruction::isInMask( sibr::Vector2i & pos)
{
	const float maskVal = _mask.at<float>(pos.y(), pos.x());
	return (std::abs(maskVal) < 0.5f);
}

bool PoissonReconstruction::isIgnored(sibr::Vector2i & pos)
{
	return (_mask.at<float>(pos.y(), pos.x()) <= -0.5f);
}


}