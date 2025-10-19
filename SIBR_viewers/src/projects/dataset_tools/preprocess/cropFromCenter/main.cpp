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


#include <core/imgproc/CropScaleImageUtility.hpp>
#include <core/system/CommandLineArgs.hpp>



/*
Crop input images from center so they end up with resolution <crop_width> x <crop_height>
if scale down factor is also passed, after the image has been cropped, it will be scaled down by that value
*/
const char* USAGE = "Usage: cropFromCenter --inputFile <path_to_input_file> --outputPath <path_to_output_folder> --avgResolution <width x height> --cropResolution <width x height> [--scaleDownFactor <alpha> --targetResolution <width x height>] \n";
//const char* USAGE						= "Usage: cropFromCenter --inputFile <path_to_input_file> --outputPath <path_to_output_folder> --avgResolution <width x height> --cropResolution <widht x height> [--scaleDownFactor <alpha> --targetResolution <width x height>] \n";
const char* TAG = "[cropFromCenter]";
const unsigned PROCESSING_BATCH_SIZE = 150;
const char* LOG_FILE_NAME = "cropFromCenter.log";
const char* SCALED_DOWN_SUBFOLDER = "scaled";
const char* SCALED_DOWN_FILENAME = "scale_factor.txt";

struct CropAppArgs :
	virtual sibr::BasicIBRAppArgs {
	sibr::Arg<std::string> inputFileArg = { "inputFile", "" };
	sibr::Arg<std::string> outputFolderArg = { "outputPath", "" };
	sibr::Arg<sibr::Vector2i> avgResolutionArg = { "avgResolution",{ 0, 0 } };
	sibr::Arg<sibr::Vector2i> cropResolutionArg = { "cropResolution",{ 0, 0 } };
	sibr::Arg<float> scaleDownFactorArg = { "scaleDownFactor", 0.0f };
	sibr::Arg<sibr::Vector2i> targetResolutionArg = { "targetResolution",{ 0, 0 } };
};

void printUsage()
{
	std::cout << USAGE << std::endl;
}


bool getParamas(int argc, const char ** argv,
	std::string & inputFile, boost::filesystem::path & outputPath,
	sibr::Vector2i & avgResolution, sibr::Vector2i & cropResolution, float & scaleDownFactor, sibr::Vector2i & targetResolution)
{

	sibr::CommandLineArgs::parseMainArgs(argc, argv);
	CropAppArgs myArgs;

	inputFile = myArgs.inputFileArg;

	std::string outputFolder = myArgs.outputFolderArg;
	outputPath = outputFolder;

	avgResolution = myArgs.avgResolutionArg;

	cropResolution = myArgs.cropResolutionArg;

	// optional parameters
	if (myArgs.scaleDownFactorArg != 0.0f) {
		scaleDownFactor = myArgs.scaleDownFactorArg;
	}

	if (myArgs.targetResolutionArg.get() != sibr::Vector2i(0, 0)) {
		targetResolution = myArgs.targetResolutionArg;
	}


	if (inputFile.empty() || outputFolder.empty() || avgResolution == sibr::Vector2i(0, 0) || cropResolution == sibr::Vector2i(0, 0)) {
		return false;
	}

	return true;
}


int main(const int argc, const char** argv)
{
	// process parameters
	std::string					inputFileName;
	boost::filesystem::path		outputFolder;
	boost::filesystem::path		scaledDownOutputFolder;
	sibr::Vector2i				avgInitialResolution;		// just for statistics and log file
	sibr::Vector2i				cropResolution;
	float						scaleDownFactor = 0.f;
	sibr::Vector2i				targetResolution;

	sibr::CropScaleImageUtility appUtility;

	if (!getParamas(argc, argv, inputFileName, outputFolder, avgInitialResolution, cropResolution, scaleDownFactor, targetResolution)) {
		std::cerr << TAG << " ERROR: wrong parameters.\n";
		printUsage();
		return -1;
	}

	scaledDownOutputFolder = (outputFolder / SCALED_DOWN_SUBFOLDER);

	bool scaleDown = (scaleDownFactor > 0);
	//cv::Size resizedSize (finalResolution[0], cropResolution[1] * ((float)(finalResolution[0]) / cropResolution[0]));
	cv::Size resizedSize(int(cropResolution[0] * scaleDownFactor), int(cropResolution[1] * scaleDownFactor));


	if (!boost::filesystem::exists(outputFolder))
	{
		boost::filesystem::create_directory(outputFolder);
	}

	if (scaleDown && !boost::filesystem::exists(scaledDownOutputFolder)) {
		boost::filesystem::create_directory(scaledDownOutputFolder);
	}

	// read input file
	std::vector<std::string> pathToImgs = appUtility.getPathToImgs(inputFileName);
	std::vector<sibr::CropScaleImageUtility::Image> listOfImages(pathToImgs.size());
	std::vector<sibr::CropScaleImageUtility::Image> listOfImagesScaledDown(scaleDown ? pathToImgs.size() : 0);

	// calculate nr batches
	const int nrBatches = static_cast<int>(ceil((float)(pathToImgs.size()) / PROCESSING_BATCH_SIZE));

	std::chrono::time_point <std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	const int batchSize = int(PROCESSING_BATCH_SIZE);
	// run batches sequentially
	for (int batchId = 0; batchId < nrBatches; batchId++) {

		const int nrItems = (batchId != nrBatches - 1) ? batchSize : ((nrBatches * batchSize != int(pathToImgs.size())) ? (int(pathToImgs.size()) - (batchSize * batchId)) : batchSize);

		#pragma omp parallel for
		for (int localImgIndex = 0; localImgIndex < nrItems; localImgIndex++) {

			const int globalImgIndex = (batchId * batchSize) + localImgIndex;

			// using next code will keep filename in output directory
			boost::filesystem::path boostPath(pathToImgs[globalImgIndex]);
			//std::string outputFileName = (outputFolder / boostPath.filename()).string();

			std::stringstream ss;
			ss << std::setfill('0') << std::setw(8) << globalImgIndex << boostPath.extension().string();
			std::string outputFileName = (outputFolder / ss.str()).string();
			std::string scaledDownOutputFileName = (scaledDownOutputFolder / ss.str()).string();

			cv::Mat img = cv::imread(pathToImgs[globalImgIndex], 1);

			cv::Rect areOfIntererst = cv::Rect((img.cols - cropResolution[0]) / 2, (img.rows - cropResolution[1]) / 2, cropResolution[0], cropResolution[1]);

			cv::Mat croppedImg = img(areOfIntererst);

			cv::imwrite(outputFileName, croppedImg);

			listOfImages[globalImgIndex].filename = ss.str();
			listOfImages[globalImgIndex].width = croppedImg.cols;
			listOfImages[globalImgIndex].height = croppedImg.rows;

			if (scaleDown) {
				cv::Mat resizedImg;
				cv::resize(croppedImg, resizedImg, resizedSize, 0, 0, cv::INTER_LINEAR);

				cv::imwrite(scaledDownOutputFileName, resizedImg);

				listOfImagesScaledDown[globalImgIndex].filename	= ss.str();
				listOfImagesScaledDown[globalImgIndex].width	= resizedImg.cols;
				listOfImagesScaledDown[globalImgIndex].height	= resizedImg.rows;
			}
		}
	}

	end = std::chrono::system_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

	std::cout << TAG << " elapsed time=" << elapsedTime << "s.\n";

	appUtility.logExecution(avgInitialResolution, int(pathToImgs.size()), elapsedTime, scaleDown, LOG_FILE_NAME);

	// write list_images.txt
	appUtility.writeListImages((outputFolder / "list_images.txt").string(), listOfImages);

	// write list_images.txt and scale_factor in scaled down directoy if needed
	if (scaleDown) {
		appUtility.writeListImages((scaledDownOutputFolder / "list_images.txt").string(), listOfImagesScaledDown);
		appUtility.writeScaleFactor((scaledDownOutputFolder / SCALED_DOWN_FILENAME).string(), scaleDownFactor);

		if (targetResolution != sibr::Vector2i(0, 0)) {
			appUtility.writeTargetResolution((scaledDownOutputFolder / "target_resolution.txt").string(), targetResolution);
		}
	}

	return 0;
}