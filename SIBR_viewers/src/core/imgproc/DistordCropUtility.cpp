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


#include "DistordCropUtility.hpp"

namespace sibr {
	

	bool DistordCropUtility::isBlack(const sibr::Vector3ub & pixelColor, Vector3i backgroundColor, int threshold_black_color) {
		sibr::Vector3i c = pixelColor.cast<int>() - backgroundColor;
		return c.squaredNorm() < threshold_black_color;
	}


	bool DistordCropUtility::is_number(const std::string& s)
	{
		return !s.empty() && std::find_if(s.begin(),
			s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
	}

	void DistordCropUtility::addPixelToQueue(const sibr::Vector2i & pixel, const sibr::ImageRGB & img, std::priority_queue<sibr::Vector2i> & queue, sibr::Array2d<bool> & arrayVisited, Vector3i backgroundColor, int threshold_black_color) {
		if (!arrayVisited(pixel.x(), pixel.y()) && isBlack(img(pixel.x(), pixel.y()), backgroundColor, threshold_black_color)) {
			queue.push(pixel);
			arrayVisited(pixel.x(), pixel.y()) = true;
		}
	}

	void DistordCropUtility::findBounds(sibr::Array2d<bool> & isBlack, DistordCropUtility::Bounds & bounds, int thinest_bounding_box_size)
	{
		bool wasUpdated = true;

		while (wasUpdated) {

			wasUpdated = false;

			for (int x = bounds.xMin; x <= bounds.xMax; ++x) {
				wasUpdated = wasUpdated || isBlack(x, bounds.yMax) || isBlack(x, bounds.yMin);
			}
			for (int y = bounds.yMin; y <= bounds.yMax; ++y) {
				wasUpdated = wasUpdated || isBlack(bounds.xMax, y) || isBlack(bounds.xMin, y);
			}

			if (wasUpdated) {
				--bounds.xMax;
				++bounds.xMin;
				--bounds.yMax;
				++bounds.yMin;
			}

			if (bounds.xMax - bounds.xMin < thinest_bounding_box_size || bounds.yMax - bounds.yMin < thinest_bounding_box_size) {
				break;
			}
		}
	}


	DistordCropUtility::Bounds DistordCropUtility::getBounds(const sibr::ImageRGB & img, Vector3i backgroundColor, int threshold_black_color, int thinest_bounding_box_size, float toleranceFactor) {
		int w = img.w() - 1;
		int h = img.h() - 1;

		sibr::Array2d<bool> wasVisited(img.w(), img.h(), false);
		sibr::Array2d<bool> isBlack(img.w(), img.h(), false);
		std::priority_queue<sibr::Vector2i> pixelsQueue;

		//init with boundary pixel (set initial pixelQueue)
		// add first row and last row of pixels to the pixelsQueue (if they are black) and marked them as visited
		for (int x = 0; x<w; ++x) {
			addPixelToQueue(sibr::Vector2i(x, 0), img, pixelsQueue, wasVisited, backgroundColor, threshold_black_color);
			addPixelToQueue(sibr::Vector2i(x, h - 1), img, pixelsQueue, wasVisited, backgroundColor, threshold_black_color);
		}

		// add left col and right col of pixels to the pixelsQueue (if they are black) and marked them as visited
		for (int y = 0; y<h; ++y) {
			addPixelToQueue(sibr::Vector2i(0, y), img, pixelsQueue, wasVisited, backgroundColor, threshold_black_color);
			addPixelToQueue(sibr::Vector2i(w - 1, y), img, pixelsQueue, wasVisited, backgroundColor, threshold_black_color);
		}

		//neighbors shifts
		sibr::Vector2i shiftsArray[4] = { sibr::Vector2i(1,0), sibr::Vector2i(-1,0), sibr::Vector2i(0,-1), sibr::Vector2i(0,1) };
		std::vector<sibr::Vector2i> shifts(shiftsArray, shiftsArray + sizeof(shiftsArray) / sizeof(sibr::Vector2i));

		//find all black pixels linked to the boundaries
		while (pixelsQueue.size() > 0) {
			sibr::Vector2i currentPix = pixelsQueue.top();
			pixelsQueue.pop();
			// if it was in the queue, then it was black
			isBlack(currentPix.x(), currentPix.y()) = true;

			for (auto & shift : shifts) {
				sibr::Vector2i newPos = currentPix + shift;
				if (img.isInRange(newPos.x(), newPos.y())) {
					addPixelToQueue(newPos, img, pixelsQueue, wasVisited, backgroundColor, threshold_black_color);
				}
			}

		}

		//find maximal bounding box not containing black pixels
		DistordCropUtility::Bounds bounds(img);
		findBounds(isBlack, bounds, thinest_bounding_box_size);

		bounds.xRatio = bounds.xMax / (float)img.w() - 0.5f;
		bounds.yRatio = bounds.yMax / (float)img.h() - 0.5f;

		int proposedWidth = bounds.xMax - bounds.xMin;
		int proposedHeight = bounds.yMax - bounds.yMin;

		bounds.width = int(float(int(img.w()) - proposedWidth) * toleranceFactor + float(proposedWidth));
		bounds.height = int(float(int(img.h()) - proposedHeight) * toleranceFactor + float(proposedHeight));

		return bounds;
	}


	sibr::Vector2i DistordCropUtility::calculateAvgResolution(const std::vector<Path>& imagePaths, std::vector<sibr::Vector2i> & resolutions, const int batch_size)
	{
		const int nrBatches = static_cast<int>(ceil((float)(imagePaths.size()) / batch_size));
		resolutions.resize(imagePaths.size());
		std::vector<std::pair<std::pair<long, long>, unsigned>> sumAndNrItems(nrBatches);

		for (int batchId = 0; batchId < nrBatches; batchId++) {

			const int nrItems = (batchId != nrBatches - 1) ? batch_size : ((nrBatches * batch_size != int(imagePaths.size())) ? (int(imagePaths.size()) - (batch_size * batchId)) : batch_size);
			long sumOfWidths = 0;
			long sumOfHeights = 0;

			std::vector<sibr::ImageRGB> chunkOfInputImages(nrItems);

#pragma omp parallel for
			for (int localImgIndex = 0; localImgIndex < nrItems; localImgIndex++) {
				unsigned globalImgIndex = (batchId * batch_size) + localImgIndex;
				chunkOfInputImages.at(localImgIndex).load(imagePaths.at(globalImgIndex).string(), false);

#pragma omp critical
				{
					sumOfWidths += long(chunkOfInputImages[localImgIndex].w());
					sumOfHeights += long(chunkOfInputImages[localImgIndex].h());
					resolutions[localImgIndex].x() = chunkOfInputImages[localImgIndex].w();
					resolutions[localImgIndex].y() = chunkOfInputImages[localImgIndex].h();
				}
			}
			std::pair<long, long> sums(sumOfWidths, sumOfHeights);
			std::pair<std::pair<long, long>, unsigned> batch(sums, nrItems);
			sumAndNrItems[batchId] = batch;
		}

		long sumOfWidth = 0;
		long sumOfHeight = 0;
		for (unsigned i = 0; i < sumAndNrItems.size(); i++) {
			sumOfWidth += sumAndNrItems[i].first.first;
			sumOfHeight += sumAndNrItems[i].first.second;
		}

		const long globalAvgWidth = sumOfWidth / long(imagePaths.size());
		const long globalAvgHeight = sumOfHeight / long(imagePaths.size());

		return sibr::Vector2i(int(globalAvgWidth), int(globalAvgHeight));
	}

	sibr::Vector2i DistordCropUtility::findBiggestImageCenteredBox(const Path & root,
		const std::vector<Path>& imagePaths, 
		std::vector<sibr::Vector2i>& resolutions, 
		int avgWidth, int avgHeight, 
		const int batch_size, 
		float resolutionThreshold, 
		float threshold_ratio_bounding_box_size, 
		Vector3i backgroundColor, 
		int threshold_black_color, 
		int thinest_bounding_box_size, 
		float toleranceFactor)
	{
		// check if avg resolution needs to be calculated
		if (avgWidth == 0 || avgHeight == 0) {
			std::cout << "about to calculate avg resolution. use python get_image_size script if dataset has too many images\n";
			sibr::Vector2i avgResolution = calculateAvgResolution(imagePaths, resolutions, batch_size);
			avgWidth = avgResolution.x();
			avgHeight = avgResolution.y();
		}

		std::cout << "[distordCrop] average resolution " << avgWidth << "x" << avgHeight << " and nr resolutions given: " << resolutions.size() << "\n";

		// discard images with different resolution
		std::vector<uint> preExcludedCams;
		for (unsigned i = 0; i < resolutions.size(); i++) {
			bool shrinkHorizontally = ((resolutions[i].x() < avgWidth) && ((avgWidth - resolutions[i].x()) > avgWidth * resolutionThreshold)) ? true : false;
			bool shrinkVertically = ((resolutions[i].y() < avgHeight) && ((avgHeight - resolutions[i].y()) > avgHeight * resolutionThreshold)) ? true : false;
			if (shrinkHorizontally || shrinkVertically) {
				preExcludedCams.push_back(i);
				std::cout << "[distordCrop] excluding input image " << i << " resolution=" << resolutions[i].x() << "x" << resolutions[i].y() << "\n";
			}
		}

		std::cout << "[distordCrop] nr pre excluded images " << preExcludedCams.size() << "\n";

		// compute bounding boxes for all non-discarded images
		std::vector<Bounds> allBounds(imagePaths.size());

		const int nrBatches = static_cast<int>(ceil((float)(imagePaths.size()) / batch_size));

		// processs batches sequentially (we don't want to run out of memory)
		for (int batchId = 0; batchId < nrBatches; batchId++) {

			const int nrItems = (batchId != nrBatches - 1) ? batch_size : ((nrBatches * batch_size != int(imagePaths.size())) ? (int(imagePaths.size()) - (batch_size * batchId)) : batch_size);

			std::vector<sibr::ImageRGB> chunkOfInputImages(nrItems);

			// load images in parallel (OpenMP 2.0 doesn't allow unsigned int as index. must be signed integral type)
#pragma omp parallel for
			for (int localImgIndex = 0; localImgIndex < nrItems; localImgIndex++) {
				const uint globalImgIndex = uint((batchId * batch_size) + localImgIndex);
				// if cam was discarded, do nothing
				if (std::find(preExcludedCams.begin(), preExcludedCams.end(), globalImgIndex) == preExcludedCams.end()) {
					// only now load the img
					chunkOfInputImages.at(localImgIndex).load(imagePaths.at(globalImgIndex).string(), false);
					allBounds.at(globalImgIndex) = getBounds(chunkOfInputImages.at(localImgIndex), backgroundColor, threshold_black_color, thinest_bounding_box_size, toleranceFactor);

				}
			}
		}

		Bounds finalBounds(resolutions.at(0));

		int im_id = 0;

		// generate exclude file based on x and y ratios
		std::string excludeFilePath = root.string() + "/exclude_images.txt";
		std::ofstream excludeFile(excludeFilePath, std::ios::trunc);

		int minWidth = -1;
		int minHeight = -1;

		for (auto & bounds : allBounds) {
			bool wasPreExcluded = std::find(preExcludedCams.begin(), preExcludedCams.end(), im_id) != preExcludedCams.end();

			if (!wasPreExcluded && bounds.xRatio > threshold_ratio_bounding_box_size && bounds.yRatio > threshold_ratio_bounding_box_size) {
				// get global x and y ratios
				bool check = false;
				if (bounds.xRatio < finalBounds.xRatio) {
					finalBounds.xRatio = bounds.xRatio;
					check = true;
				}
				if (bounds.yRatio < finalBounds.yRatio) {
					finalBounds.yRatio = bounds.yRatio;
					check = true;
				}

				minWidth = (minWidth < 0 || bounds.width < minWidth) ? bounds.width : minWidth;
				minHeight = (minHeight < 0 || bounds.height < minHeight) ? bounds.height : minHeight;
			}
			else {
				std::cerr << im_id << " ";
				excludeFile << im_id << " ";

				std::cout << wasPreExcluded << " " << bounds.xRatio << " " << threshold_ratio_bounding_box_size << " " << bounds.yRatio << " " << threshold_ratio_bounding_box_size << std::endl;
			}

			++im_id;

		}
		excludeFile.close();
		std::cout << std::endl;

		return sibr::Vector2i(minWidth, minHeight);

	}

	sibr::Vector2i DistordCropUtility::findMinImageSize(const Path & root, const std::vector<Path>& imagePaths)
	{
		std::vector<sibr::ImageRGB> inputImgs(imagePaths.size());
		std::vector<sibr::Vector2i> imSizes(imagePaths.size());

		std::cout << "[distordCrop] loading input images : " << std::flush;

#pragma omp parallel for
		for (int id = 0; id < (int)inputImgs.size(); ++id) {
			inputImgs.at(id).load(imagePaths.at(id).string(), false);
			imSizes[id] = inputImgs[id].size().cast<int>();
		}

		sibr::Vector2i minSize = imSizes[0];
		for (const auto & size : imSizes) {
			minSize = minSize.cwiseMin(size);
		}

		// generate exclude file based on x and y ratios
		std::string excludeFilePath = root.string() + "/excludeImages.txt";
		std::ofstream excludeFile(excludeFilePath, std::ios::trunc);
		excludeFile.close();

		return minSize;
	}

}