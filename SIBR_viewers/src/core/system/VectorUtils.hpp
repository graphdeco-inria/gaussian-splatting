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
# include "core/system/Config.hpp"
#include <functional>

namespace sibr
{
	/**
	* \addtogroup sibr_system
	* @{
	*/

	/** Sum elements from a vector.
	 *\param vec vector
	 *\param f validity function (f(i) == true if the i-th element should be taken into account.
	 *\return the accumulated sum
	 **/
	template<typename T_in, typename T_out = T_in> T_out sum(
		const std::vector<T_in> & vec ,
		const std::function<bool(T_in)> & f = [](T_in val) { return true; }
	) {
		double sum = 0;
		for (T_in val : vec) {
			if( f(val) ){
				sum += (double)val;
			}
		}
		return (T_out)sum;
	}

	/** Weighted sum of elements in a vector.
	 *\param vec vector
	 *\param weights per-element weight
	 *\param f validity function (f(i) == true if the i-th element should be taken into account.
	 *\return the weighted sum
	 **/
	template<typename T_in, typename T_out = T_in> std::vector<T_out> weighted_normalization(
		const std::vector<T_in> & vec,
		const std::vector<T_in> & weights,
		const std::function<bool(T_in)> & f = [](T_in val) { return true; }
	) {
		double sum = 0;
		int size = (int)std::min(vec.size(), weights.size());
		for (int i = 0; i < size; ++i) {
			T_in val = vec[i];
			if (f(val)) {
				sum += (double)val*(double)weights[i];
			}
		}

		std::vector<T_out> out(size);
		for (int i = 0; i < size; ++i) {
			if ((sum == 0) || !f(vec[i])) {
				out[i] = (T_out)vec[i];
			} else {
				out[i] = (T_out)( ( (double)vec[i] * (double)weights[i] )/sum );
			}
		}
			
		return out;
	}

	/** Apply a function to each element of a vector (not in place)
	 *\param vec vector
	 *\param f function to apply
	 *\return vector containing the processed results
	 */
	template<typename T_in, typename T_out = T_in> std::vector<T_out> applyLambda(
		const std::vector<T_in> & vec,
		const std::function<T_out(T_in)> & f
	) {
		std::vector<T_out> out(vec.size());
		for (int i = 0; i < vec.size(); ++i) {
			out[i] = f(vec[i]);
		}
		return out;
	}

	/**  Apply a function to each pair of elements from two vectors of same size (not in place).
	 *\param vec1 first vector
	 *\param vec2 second vector
	 *\param f function to apply
	 *\return vector containing the processed results
	 */
	template<typename T_in, typename T_out> std::vector<T_out> applyLambda(
		const std::vector<T_in> & vec1,
		const std::vector<T_in> & vec2,
		const std::function<T_out(T_in,T_in)> & f
	) {
		int size = (int)std::min(vec1.size(), vec2.size());
		std::vector<T_out> out(size);
		for (int i = 0; i < size; ++i) {
			out[i] = f(vec1[i],vec2[i]);
		}
		return out;
	}

	/** Compute the variance of elements in a vector.
	 *\param vec vector
	 *\param f validity function (f(i) == true if the i-th element should be taken into account.
	 *\return the variance
	 **/
	template<typename T_in, typename T_out = T_in> T_out var(
		const std::vector<T_in> & vec,
		const std::function<bool(T_in)> & f = [](T_in val) { return true; } 
	) {
		double sum = 0;
		double sum2 = 0;
		int n = 0;

		for (T_in val : vec) {
			if ( f(val) ) {
				sum += (double)val;
				sum2 += (double)val*(double)val;
				++n;
			}
		}

		if (n < 2) {
			return (T_out)(-1);
		}
		else {
			return (T_out)((sum2 - sum*sum / (double)n) / double(n - 1));
		}
		
	}

	/** Normalize all elements in a vector based on the min and max values in it (not in place).
	 *\param vec vector
	 *\param f validity function (f(i) == true if the i-th element should be taken into account.
	 *\return a vector containing the normalized values
	 **/
	template<typename T_in, typename T_out = T_in> std::vector<T_out> normalizedMinMax(
		const std::vector<T_in> & vec,
		const std::function<bool(T_in)> & f = [](T_in val) { return true; }
	) {
		T_in min = 0, max = 0;
		bool first = true;
		for (T_in val : vec) {
			if (f(val)) {
				if (first || val > max) {
					max = val;
				}
				if (first || val < min) {
					min = val;
				}
				first = false;
			}
		}
		if (min == max) {
			return std::vector<T_out>();
		}
		
		std::vector<T_out> out(vec.size());
		const double normFactor = 1.0 / (double)(max - min);
		for (int i = 0; i < (int)vec.size(); ++i) {
			out[i] = f(vec[i]) ? (T_out)((double)(vec[i] - min)*normFactor) : (T_out)vec[i];
		}
		return out;
	}

	/** Apply a power-sum normalization.
	 *\param vec vector
	 *\param f validity function (f(i) == true if the i-th element should be taken into account.
	 *\return a vector containing the normalized values
	 */
	template<typename T_in, typename T_out = T_in, unsigned int Power = 2> std::vector<T_out> normalizedZeroOne(
		const std::vector<T_in> & vec,
		const std::function<bool(T_in)> & f = [](T_in val) { return true; }
	) {
		double sumP = 0;

		for (T_in val : vec) {
			if (f(val)) {
				sumP += std::pow((double)val, Power);
			}
			
		}

		if (sumP == 0) {
			return std::vector<T_out>();
		}
		
		std::vector<T_out> out(vec.size());
		for (int i = 0; i <(int)vec.size(); ++i) {
			out[i] = f(vec[i]) ? (T_out)(vec[i] / sumP) : (T_out)vec[i];
		}
		return out;
			
	}

	/*** @} */

	/**
	 * Multi dimensional vector.
	* \ingroup sibr_system
	*/
	template< typename T, unsigned int N >
	class MultiVector : public std::vector< MultiVector<T, N - 1> >
	{
		static_assert(N >= 1, " MultiVector<N> : the number of dimensions N must be >= 1 ");

		friend class MultiVector<T, N + 1>;

		typedef MultiVector<T, N - 1> SubVector;

	public:

		/// Constructor.
		MultiVector() {}

		/** Constructor.
		 *\param n number of elements on each axis
		 *\param t default value
		 */
		MultiVector(int n, const T & t = T() )
			: std::vector< SubVector >(n, SubVector(n, t)) { }

		/** Constructor.
		 *\param dims number of elements on each axis
		 *\param t default value
		 */
		MultiVector(const std::vector<int> & dims, const T & t = T() )
			: std::vector< SubVector >(dims.at(dims.size()-N), SubVector(dims, t)) { }

		/** Getter
		 *\param  ids N-d coordinates
		 *\return a reference to the corresponding value.
		 */
		T & multiAt(const std::vector<int> & ids) {
			return this.at(ids.at(ids.size() - N)).multiAt(ids);
		}

		/** Getter
		 *\param  ids N-d coordinates
		 *\return a const reference to the corresponding value.
		 */
		const T & multiAt(const std::vector<int> & ids) const {
			return this.at(ids.at(ids.size() - N)).multiAt(ids);
		}

		/** Get the size along each dimension.
		 *\return the N-d size
		 **/
		std::vector<int> dims() const
		{
			std::vector<int> v;
			dimsRecur(v);
			return v;
		}

		/**Print the size along each dimension.
		 */
		void dimsDisplay() const {
			std::vector<int> d(dims());
			std::cout << " [ ";
			for (int i = 0; i < N; ++i) {
				std::cout << d[i] << (i != N - 1 ? " x " : "");
			}
			std::cout << " ] " << std::endl;
		}

	protected:

		/** Helper to get the dimensions.
		 *\param v will contain the size along each axis
		 */
		void dimsRecur(std::vector<int> & v) const
		{
			v.push_back((int)this.size());
			this.at(0).dimsRecur(v);
		}
	};

	/** Base multi-dimensional vector class (a 1D vector). 
	\ingroup sibr_system
	*/
	template<typename T>
	class MultiVector<T, 1> : public std::vector<T>
	{
		friend class MultiVector<T, 2>;

	public:

		/// Constructor.
		MultiVector() {}

		/** Constructor.
		 *\param n number of elements
		 *\param t default value
		 */
		MultiVector(int n, const T & t = T() )
			: std::vector<T>(n, t) { }

		/** Constructor.
		 *\param dims number of elements on each axis (only the last one will be considered here).
		 *\param t default value
		 */
		MultiVector(const std::vector<int> & dims, const T & t = T())
			: std::vector<T>(dims.at(dims.size()-1), t) { }

		/** Getter
		 *\param  ids N-d coordinates
		 *\return a reference to the corresponding value.
		 */
		T & multiAt(const std::vector<int> & ids) {
			return this.at(ids.at(ids.size() - 1));
		}

		/** Getter
		 *\param  ids N-d coordinates
		 *\return a const reference to the corresponding value.
		 */
		const T & multiAt(const std::vector<int> & ids) const {
			return this.at(ids.at(ids.size() - 1));
		}

		/**Print the size along each dimension.
		 */
		void dimsDisplay() const {
			std::cout << " [ " << this.size() << " ] " << std::endl;
		}

	protected:

		/** Helper to get the dimensions.
		 *\param v will contain the size along each axis
		 */
		void dimsRecur(std::vector<int> & v) const
		{
			v.push_back((int)this.size());
		}

	};

	
} // namespace sibr
