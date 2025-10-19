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


#include "MRFSolver.h"


namespace sibr {

	MRFSolver::MRFSolver(void)
	{
	}

	MRFSolver::MRFSolver(std::vector<int> labels_list, std::vector<std::vector<int> >* neighborMap, int numIterations,
		UnaryLabelOnlyFuncPtr unaryLabelOnly,
		UnaryFuncPtr unaryFull,
		PairwiseLabelOnlyFuncPtr pairwiseLabelsOnly,
		PairwiseFuncPtr pairwiseFull)
		: ignoreIsolatedNode(false)
	{
		SIBR_LOG << "[MRFSolver] Initialization ... ";

		_labList = labels_list;
		_neighborMap = neighborMap;
		_numIterations = numIterations;
		_unaryFull = unaryFull;
		_pairwiseFull = pairwiseFull;

		SIBR_LOG << "[MRFSolver] Labels : ";
		for (int l_id = 0; l_id < _labList.size(); l_id++) {
			std::cout << _labList[l_id] << ",";
		}
		std::cout << std::endl;

		//storing values for the unary part only requiring label
		if (unaryLabelOnly.get()) {
			SIBR_LOG << "[MRFSolver] unaryLabelOnly exists, precomputing." << std::endl;
			_UnaryLabelOnly.resize(_labList.size());
			for (int l_id = 0; l_id < _labList.size(); l_id++) {
				_UnaryLabelOnly[l_id] = (*unaryLabelOnly)(_labList[l_id]);
			}
		}
		else {
			SIBR_LOG << "[MRFSolver] unaryLabelOnly does not exist, skipping." << std::endl;
		}

		//storing values for the pairwise part only requiring labels
		if (pairwiseLabelsOnly.get()) {
			SIBR_LOG << "[MRFSolver] pairwiseLabelsOnly exists, precomputing." << std::endl;
			_PairwiseLabelsOnly.resize(_labList.size());

			for (int l_id1 = 0; l_id1 < _labList.size(); l_id1++) {
				_PairwiseLabelsOnly[l_id1].resize(_labList.size(), -1);

				for (int l_id2 = 0; l_id2 < _labList.size(); l_id2++) {
					_PairwiseLabelsOnly[l_id1][l_id2] = (*pairwiseLabelsOnly)(_labList[l_id1], _labList[l_id2]);
				}
			}
		}
		else {
			SIBR_LOG << "[MRFSolver] pairwiseLabelsOnly does not exist, skipping." << std::endl;
		}

		SIBR_LOG << "[MRFSolver] Setup complete." << std::endl;
	}

	void MRFSolver::solveLabels(void)
	{
		SIBR_LOG << "[MRFSolver] Running mincut... " << std::endl;

		double infty = (double)1e20;
		double min_unary, temp_unary;
		int num_nodes = (int)_neighborMap->size();
		SIBR_LOG << "[MRFSolver] Number of nodes = " << num_nodes;
		_labels.resize(num_nodes);

		int numLinks = 0;
		for (auto & links : (*_neighborMap)) {
			numLinks += (int)links.size();
		}
		SIBR_LOG << ", number of links = " << numLinks / 2 << std::endl;
		
		SIBR_LOG << "[MRFSolver] Initialization : minimizing unaries..." << std::flush;
		for (int p = 0; p < num_nodes; p++) {

			int label_id;
			min_unary = infty;
			int num_cand = 0;
			for (int lp_id = 0; lp_id < (int)_labList.size(); lp_id++) {
				temp_unary = unaryTotal(p, lp_id);
				if (temp_unary < (1 << 10)) { ++num_cand; }
				if (temp_unary < min_unary) {
					min_unary = temp_unary;
					label_id = lp_id;
				}
			}
			_labels[p] = label_id;
		}
		std::cout << " Done." << std::endl;

		SIBR_LOG << "[MRFSolver] Energies: U: " << computeEnergyU() << ", W: " << computeEnergyW() << std::endl;

		// Alpha-expansion algorithm
		SIBR_LOG << "[MRFSolver] Alpha-expansion [label,flow]..." << std::endl;
		for (int it = 0; it < _numIterations; it++) {
			SIBR_LOG << "[MRFSolver] Iteration " << (it+1)  << "/" << (_numIterations) << ": " << std::endl;
			
			for (int label_id = 0; label_id < (int)_labList.size(); label_id++) {
				int label = _labList.at(label_id);
				
				buildGraphAlphaExp(label_id);
				// Solve mincut
				_energy = _graph->maxflow();


				int num_change = 0;
				//assign new labels
				for (int p = 0; p < num_nodes; p++) {
					if (_graph->what_segment(p) == GraphType::SINK) {
						if (_labels[p] != label_id) { ++num_change; }
						_labels[p] = label_id;
					}
				}
				SIBR_LOG << "[MRFSolver]\t\tLabel " << label << ": modifications = " <<  num_change << ", energy = " << _energy << " ]" << std::endl;

				delete _graph;
			}
		}
		SIBR_LOG << "[MRFSolver] Done." << std::endl;
	}

	void MRFSolver::buildGraphAlphaExp(int label_iteration_id)
	{
		double infty = 1 << 25;
		int num_nodes = (int)_neighborMap->size();
		int n_nodes_estimation = num_nodes;
		int n_edges_estimation = num_nodes * 4;

		_graph = new GraphType(n_nodes_estimation, n_edges_estimation);

		//add nodes associated to pixels
		int node_id = 0;
		for (int p = 0; p < num_nodes; p++) {
			_graph->add_node();

			if (_labels[p] == label_iteration_id) {
				_graph->add_tweights(node_id, unaryTotal(p, label_iteration_id), infty);
			} else {
				_graph->add_tweights(node_id, unaryTotal(p, label_iteration_id), unaryTotal(p, _labels[p]));
			}
			++node_id;
		}
		
		//add nodes associated to connexions between pixels
		for (int p = 0; p < num_nodes; p++) {

			std::vector<int> & neighors = (*_neighborMap)[p];
			for (int q_id = 0; q_id < (int)neighors.size(); q_id++) {

				int q = neighors[q_id];

				if (p == q) { std::cerr << "!"; }
				if (q < p) { continue; }

				if (_labels[p] != _labels[q]) {
					//extra node associated to edge {p,q}
					_graph->add_node();

					_graph->add_tweights(node_id, 0, pairwiseTotal(q, p, _labels[q], _labels[p]));

					double pairwise_q_a = pairwiseTotal(q, p, _labels[q], label_iteration_id);
					_graph->add_edge(q, node_id, pairwise_q_a, pairwise_q_a);

					double pairwise_p_a = pairwiseTotal(q, p, label_iteration_id, _labels[p]);
					_graph->add_edge(p, node_id, pairwise_p_a, pairwise_p_a);

					++node_id;
				}
				else
				{
					double pairwise_p_q = pairwiseTotal(q, p, _labels[q], label_iteration_id);
					_graph->add_edge(q, p, pairwise_p_q, pairwise_p_q);
				}
			}
		}

	}

	void MRFSolver::solveBinaryLabels(void)
	{
		int numLabels = (int)_labList.size();
		if (numLabels < 2) {
			SIBR_ERR << "[MRFSolver] solveBinaryLabels, expected 2 labels, only " << numLabels << " labels " << std::endl;
		}
		else if (numLabels > 2) {
			SIBR_WRG << "[MRFSolver] solveBinaryLabels, found " << numLabels << " labels, only the first two will be used." << std::endl;
		}

		buildGraphBinaryLabels();

		_graph->maxflow();

		int num_nodes = (int)_neighborMap->size();
		_labels.resize(num_nodes);

		//assign new labels
		for (int p = 0; p < num_nodes; p++) {

			//TODO check this is not the opposite
			if (_graph->what_segment(p) == GraphType::SINK) {
				_labels[p] = 0;
			}
			else {
				_labels[p] = 1;
			}
		}

		delete _graph;
	}

	void MRFSolver::buildGraphBinaryLabels(void)
	{
		int num_nodes = (int)_neighborMap->size();
		int n_nodes_estimation = num_nodes;
		int n_edges_estimation = num_nodes * 4;

		_graph = new GraphType(n_nodes_estimation, n_edges_estimation);

		for (int p = 0; p < num_nodes; p++) {
			_graph->add_node();
			_graph->add_tweights(p, unaryTotal(p, 0), unaryTotal(p, 1));
		}

		for (int p = 0; p < num_nodes; p++) {
			std::vector<int> & neighors = (*_neighborMap)[p];
			for (int q_id = 0; q_id < (int)neighors.size(); q_id++) {

				int q = neighors[q_id];

				if (q < p) { continue; }

				double weight = pairwiseTotal(q, p, 0, 1);

				_graph->add_edge(q, p, weight, weight);
			}
		}
	}

	double MRFSolver::unaryTotal(int p, int lp_id)
	{
		double u = 0;
		if (!_UnaryLabelOnly.empty()) {
			u += _UnaryLabelOnly[lp_id];
		}
		if (_unaryFull) {
			u += (*_unaryFull)(p, _labList[lp_id]);
		}
		if (u < 0) { std::cerr << "!"; }
		return u;
	}

	double MRFSolver::pairwiseTotal(int p, int q, int lp_id, int lq_id)
	{
		double w = 0;
		if (!_PairwiseLabelsOnly.empty()) {
			w += _PairwiseLabelsOnly[lp_id][lq_id];
		}

		if (_pairwiseFull) {
			w += (*_pairwiseFull)(p, q, _labList[lp_id], _labList[lq_id]);
		}
		if (w < 0) { std::cerr << "?" << w << "?"; }
		return w;
	}

	std::vector<int> MRFSolver::getLabels(void)
	{
		std::vector<int> labels(_labels.size());

		//switch from labels id to actual labels 
		for (int p = 0; p < (int)labels.size(); p++) {
			labels[p] = _labList[_labels[p]];
		}
		return labels;
	}

	std::vector<double> MRFSolver::getUnariesEnergies(void)
	{
		if (_labels.size() == 0) { std::cerr << "[MRFSolver] warning getUnariesEnergies without nodes" << std::endl; return std::vector<double>(); }

		std::vector<double> energies(_labels.size());
		for (int p = 0; p < (int)_labels.size(); p++) {
			energies[p] = unaryTotal(p, _labels[p]);
		}

		return energies;
	}

	double MRFSolver::computeEnergyU(void)
	{
		double e = 0;
		for (int p = 0; p < (int)_labels.size(); p++) {
			e += unaryTotal(p, _labels[p]);
		}
		return e;
	}

	double MRFSolver::computeEnergyW(void)
	{
		double e = 0;
		for (int p = 0; p < (int)_labels.size(); p++) {
			for (int q_id = 0; q_id < (int)(*_neighborMap)[p].size(); q_id++) {
				int q = (*_neighborMap)[p][q_id];
				if (q < p) { continue; }
				e += pairwiseTotal(q, p, _labels[q], _labels[p]);
			}
		}
		return e;
	}

	double MRFSolver::getTotalEnergy(void)
	{
		return _energy;
	}

	MRFSolver::~MRFSolver(void)
	{
	}

}