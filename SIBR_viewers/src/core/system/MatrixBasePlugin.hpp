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

/** Helper to evaluate a transposed matrix without overwriting risks.
	\return a copy of the matrix, transposed
*/
inline MatrixBase transposed( void ) { return this->transpose().eval(); }

/** Get the first two components, filling with a default value if some are missing.
	\param fill the default value to use
	\return the selected components.
*/
Matrix<Scalar, 2, 1,Eigen::DontAlign>	xy( float fill=0.f ) const {
	return Matrix<Scalar, 2, 1,Eigen::DontAlign>( this->operator[](0), size()<2? fill:this->operator[](1));
}

/** Get the first two components swapped, filling with a default value if some are missing.
	\param fill the default value to use
	\return the selected components.
*/
Matrix<Scalar, 2, 1, Eigen::DontAlign>	yx(float fill = 0.f) const {
	return Matrix<Scalar, 2, 1, Eigen::DontAlign>(this->operator[](1), size()<2 ? fill : this->operator[](0));
}

/** Get the last two components swapped, filling with a default value if some are missing.
	\param fill the default value to use
	\return the selected components.
*/
Matrix<Scalar, 2, 1, Eigen::DontAlign>	wz(float fill = 0.f) const {
	return Matrix<Scalar, 2, 1, Eigen::DontAlign>(size()<4 ? fill : this->operator[](3), size()<3 ? fill : this->operator[](2));
}

/** Get the first three components, filling with a default value if some are missing.
	\param fill the default value to use
	\return the selected components.
*/
Matrix<Scalar, 3, 1,Eigen::DontAlign>	xyz( float fill=0.f ) const {
	return Matrix<Scalar, 3, 1,Eigen::DontAlign>( this->operator[](0), size()<2? fill:this->operator[](1), size()<3? fill:this->operator[](2));
}

/** Get the first four components, filling with a default value if some are missing.
	\param fill the default value to use
	\return the selected components.
*/
Matrix<Scalar, 4, 1,Eigen::DontAlign>	xyzw( float fill=0.f ) const {
	return Matrix<Scalar, 4, 1,Eigen::DontAlign>( this->operator[](0), size()<2? fill:this->operator[](1), size()<3? fill:this->operator[](2), size()<4? fill:this->operator[](3));
}

/** Get the first three components swapped (YXZ), filling with a default value if some are missing.
	\param fill the default value to use
	\return the selected components.
*/
Matrix<Scalar, 3, 1, Eigen::DontAlign>	yxz(float fill = 0.f) const {
	return Matrix<Scalar, 3, 1, Eigen::DontAlign>(size()<2 ? fill : this->operator[](1), this->operator[](0), size()<3 ? fill : this->operator[](2));
}

/** Get the first three components swapped (YZX), filling with a default value if some are missing.
	\param fill the default value to use
	\return the selected components.
*/
Matrix<Scalar, 3, 1, Eigen::DontAlign>	yzx(float fill = 0.f) const {
	return Matrix<Scalar, 3, 1, Eigen::DontAlign>(size()<2 ? fill : this->operator[](1), size()<3 ? fill : this->operator[](2), this->operator[](0));
}

/** Check if a vector is exactly zero for all components
\return true if all components are exactly zero.
*/
bool	isNull( void ) const { 
	return (array() == 0).all();
}

typedef Scalar Type;

//enum { NumComp = Derived::RowsAtCompileTime };

