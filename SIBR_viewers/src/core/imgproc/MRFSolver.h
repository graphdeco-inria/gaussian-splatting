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
#include "mrf/graph.h"
#include <vector>
#include <iostream>
#include <functional>
#include <memory>

namespace sibr {

	
	/** Object wrapper around Kolmogorov & Boykov MRF solver.
	 *Solve labelling problems on regular grids using alpha expension.
	\ingroup sibr_imgproc
	*/
	class SIBR_IMGPROC_EXPORT MRFSolver
	{

	public:

		typedef std::shared_ptr<std::function<double(int, int)> > UnaryFuncPtr; ///< Unary cost function that depend on node attributes and its label.
		typedef std::shared_ptr<std::function<double(int, int, int, int)> > PairwiseFuncPtr; ///< Pairwise cost function that depend on node attributes and their labels.
		
		typedef std::shared_ptr<std::function<double(int)> > UnaryLabelOnlyFuncPtr; ///< Unary cost function that only depend on the label.
		typedef std::shared_ptr<std::function<double(int, int)> > PairwiseLabelOnlyFuncPtr; ///< Pairwise cost function that only depend on the labels.
		
		/// Default constructor.
		MRFSolver(void);

		/** Initialize from a set of labels, connections and node/edge weights.
		 *\param labels list of available labels
		 *\param neighborMap connectivity map, for each node, list of neighboring nodes linear indices
		 *\param numIterations number of expansion iterations to perform
		 *\param unaryLabelOnly optional unary cost that only depends on the label: f(lab0), else provide nullptr
		 *\param unaryFull unary (per node) cost function evaluator, receiving the node linear index and label: f(ind0, lab0)
		 *\param pairwiseLabelsOnly optional pairwise cost that only depends on the labels: f(lab0, lab1), else provide nullptr
		 *\param pairwiseFull pairwise (per pair of nodes) cost function evaluator, receiving the nodes linear indices and their labels: f(ind0, ind1, lab0, lab1)
		 *\note the "*LabelsOnly" functions are optional and are precomputed and cached for optimized resolution.
		 */
		MRFSolver(std::vector<int> labels, std::vector<std::vector<int> >* neighborMap, int numIterations,
			UnaryLabelOnlyFuncPtr unaryLabelOnly,
			UnaryFuncPtr unaryFull,
			PairwiseLabelOnlyFuncPtr pairwiseLabelsOnly,
			PairwiseFuncPtr pairwiseFull
		);

		/// Solve using alpha expansion. When you have only two labels, use solveBinaryLabels instead
		void solveLabels(void);

		/// Solve for binary labels: if you only more than two labels, call solveLabels instead. 
		void solveBinaryLabels(void);

		/** For each pixel, get the estimated label (call either solveLabels or solveBinaryLabels before).
		 \return a list of labels, one per pixel */
		std::vector<int> getLabels(void);

		/** \return the total energy of the current labeling. */
		double getTotalEnergy(void);

		/** \return the unary energy of the current labeling. */
		double computeEnergyU(void);

		/** \return the pairwise energy of the current labeling. */
		double computeEnergyW(void);

		/** \return per label unary energy. */
		std::vector<double> getUnariesEnergies(void);

		/// Destructor.
		~MRFSolver(void);

	private:

		/** Build graph for the general case.
		 *\param label_iteration_id
		 **/
		void buildGraphAlphaExp(int label_iteration_id);

		/** Build graph for the binary labeling case. */
		void buildGraphBinaryLabels(void);

		/** Compute the unary cost of a node.
		 *\param p the node linear index
		 *\param lp_id the node label to consider
		 *\return the total unary cost
		 **/
		double unaryTotal(int p, int lp_id);

		/** Compute the pairwise cost of a pair of nodes.
		 *\param p the first node linear index
		 *\param q the second node linear index
		 *\param lp_id the first node label to consider
		 *\param lq_id the scond node label to consider
		 *\return the total pairwise cost
		 **/
		double pairwiseTotal(int p, int q, int lp_id, int lq_id);

		std::vector<int> _labList; ///< Map the label_id to the actual labels.
		std::vector<int> _labels; ///< Assign each node its current best label_id.
		std::vector<std::vector<int> >* _neighborMap; ///< For each variable, gives the list of its neighbor variables
		int _numIterations; ///< Number of iterations in alpha expansion.
		
		std::vector<double> _UnaryLabelOnly; ///< Unaries only requiring label.
		std::shared_ptr<std::function<double(int, int)> > _unaryFull; ///< Unaries requiring label and variable.
		std::vector<std::vector< double > >  _PairwiseLabelsOnly; ///< Pairwises only requiring labels.
		std::shared_ptr<std::function<double(int, int, int, int)> > _pairwiseFull; ///< Pairwises requiring labels and variables.

		typedef Graph<double, double, double> GraphType;
		double _energy; ///< Total energy.
		GraphType* _graph; ///< Graph.
		bool ignoreIsolatedNode; ///< Ignore nodes with no connections.
	};

}
