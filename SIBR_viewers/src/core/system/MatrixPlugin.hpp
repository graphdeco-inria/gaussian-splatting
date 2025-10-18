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



// This file is used to extend Eigen's MatrixBase class using
// the following tricks:
// https://eigen.tuxfamily.org/dox-3.2/TopicCustomizingEigen.html

public:

typedef Scalar Type;
enum { NumComp = RowsAtCompileTime };

//Matrix( const Scalar* data ) { for(int i=0; i<NumComp; i++) this->operator [] (i) = data[i]; }

/**
Matrix( float x, float y=0.f, float z=0.f, float w=0.f ) {
	float data[] = {x, y, z, w};
	for(int i=0; i<NumComp; i++) this->operator [] (i) = data[i];
}
**/

