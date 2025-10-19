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


#include "CropScaleImageUtility.hpp"

namespace sibr {

	std::vector<std::string> CropScaleImageUtility::getPathToImgs(const std::string & inputFileName)
	{
		std::ifstream inputFile(inputFileName);
		std::string line;
		std::vector<std::string> pathToImgs;
		while (getline(inputFile, line)) {
			std::stringstream ss(line);
			std::string path;
			unsigned width, height;
			ss >> path >> width >> height;
			pathToImgs.push_back(path);
		}
		inputFile.close();
		return pathToImgs;
	}

	void CropScaleImageUtility::logExecution(const sibr::Vector2i & originalResolution, unsigned nrImages, long long elapsedTime, bool wasTransformed, const char* log_file_name)
	{
		// check if file exists
		const bool isEmptyFile = !sibr::fileExists(log_file_name);
		std::ofstream outputFile(log_file_name, std::ios::app);

		if (isEmptyFile) {
			outputFile << "date\t\t\tresolution\tnrImgs\telapsedTime\twas transformed?\n";
		}

		time_t now = std::time(nullptr);
		
#ifdef SIBR_OS_WINDOWS
		tm ltm = {0,0,0,0,0,0,0,0,0};
		localtime_s(&ltm, &now);
#else
		tm ltm = *(std::localtime(&now));
#endif
		
		std::stringstream dateSS;
		dateSS << "[" << 1900 + ltm.tm_year << "/" << 1 + ltm.tm_mon << "/" << ltm.tm_mday << "] "
			<< ltm.tm_hour << ":" << ltm.tm_min << ":" << ltm.tm_sec;

		outputFile << dateSS.str() << "\t" << originalResolution[0] << "x" << originalResolution[1] << "\t\t" << nrImages << "\t" << elapsedTime << "\t" << wasTransformed << "\n";

		outputFile.close();
	}

	void CropScaleImageUtility::writeListImages(const std::string path_to_file, const std::vector<CropScaleImageUtility::Image> & listOfImages)
	{
		std::ofstream outputFile(path_to_file);

		for (unsigned i = 0; i < listOfImages.size(); i++) {
			outputFile << listOfImages[i].filename << " " << listOfImages[i].width << " " << listOfImages[i].height << "\n";
		}

		outputFile.close();
	}

	sibr::Vector2i CropScaleImageUtility::parseResolution(const std::string & param)
	{
		size_t delimiterPos = param.find('x');
		std::string widthStr = param.substr(0, delimiterPos);
		std::string heightStr = param.substr(delimiterPos + 1);
		return sibr::Vector2i(std::stoi(widthStr), std::stoi(heightStr));
	}

	void CropScaleImageUtility::writeScaleFactor(const std::string path_to_file, float scaleFactor)
	{
		std::ofstream outputFile(path_to_file);

		outputFile << scaleFactor << "\n";

		outputFile.close();
	}

	void CropScaleImageUtility::writeTargetResolution(const std::string path_to_file, const sibr::Vector2i & targetResolution)
	{
		std::ofstream outputFile(path_to_file);

		outputFile << targetResolution[0] << " " << targetResolution[1] << "\n";

		outputFile.close();
	}
}