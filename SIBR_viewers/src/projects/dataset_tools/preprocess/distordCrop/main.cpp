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


#include <core/imgproc/DistordCropUtility.hpp>
#include <core/system/CommandLineArgs.hpp>

typedef boost::filesystem::path Path;
using namespace boost::filesystem;

int threshold_black_color = 10; //10
int thinest_bounding_box_size = 5;
int threshold_bounding_box_size = 500;
float threshold_ratio_bounding_box_size = 0.2f;

const int PROCESSING_BATCH_SIZE = 150;	// process PROCESSING_BATCH_SIZE images together

sibr::Vector3i backgroundColor = sibr::Vector3i(0, 0, 0);

/*
	if input image resolution is too different from the avg, it will be discarded automatically at the beginning
*/
float resolutionThreshold = 0.15f;

/*
	tolerance factor is used to allow somehow some black borders in the final images.
	if tolerance factor is zero, then all black borders are remove.
	if tolerance factor is one, then the image keeps its original resolution
*/
float toleranceFactor = 0.0f;

bool debug_viz = false;

struct DistordCropAppArgs :
	virtual sibr::BasicIBRAppArgs {
	sibr::Arg<int> black_threshold = { "black", threshold_black_color };
	sibr::Arg<int> minSizeThresholdArg = { "min", threshold_bounding_box_size};
	sibr::Arg<float> minRatioThresholdArg = { "ratio", threshold_ratio_bounding_box_size };
	sibr::Arg<float> resThreshold = { "resolution_threshold", 0.15f };
	sibr::Arg<float> toleranceArg = { "tolerance", toleranceFactor };
	sibr::Arg<bool> vizArg = { "debug" };
	sibr::ArgSwitch modeArg = { "modesame", true };
	sibr::Arg<int> avgWidthArg = { "avg_width", 0 };
	sibr::Arg<int> avgHeightArg = { "avg_height", 0 };
	sibr::Arg<sibr::Vector3i> backgroundColor = { "backgroundColor", sibr::Vector3i(0, 0, 0) };
};


/*
utility program that determines a new resolution taking into account that some input images have black borders added by reality capture.
the second output of the program [optional] is a excludeImages.txt file containing the id of the images that didn't pass the threshold test
(they would have to be cropped to much). current pipeline (IBR_recons_RC.py) doesn't used that file properly.

we might need to call process_cam_selection manually passing as argument the excludeImages.txt in order to actually remove the cameras that
didn't pass the threshold test

update: reality capture (using the 'fit' option when exporting bundle) sometimes produces datasets that have images not only with black borders
but also with a completely different resolution. We need to take into account those datasets too.
*/


using namespace sibr;


int main(const int argc, const char* const* argv)
{
	// parameters stuff
	sibr::CommandLineArgs::parseMainArgs(argc, argv);
	DistordCropAppArgs myArgs;

	DistordCropUtility appUtils;

	std::string datasetPath = myArgs.dataset_path;

	threshold_black_color = myArgs.black_threshold;
	threshold_bounding_box_size = myArgs.minSizeThresholdArg;
	threshold_ratio_bounding_box_size = myArgs.minRatioThresholdArg;
	toleranceFactor = myArgs.toleranceArg;
	backgroundColor = myArgs.backgroundColor;	
	resolutionThreshold = myArgs.resThreshold;

	if( myArgs.vizArg.get()) {
		debug_viz = true;
	}
	
	int avgWidth = myArgs.avgWidthArg;
	int avgHeight = myArgs.avgHeightArg;
	
	bool sameSize = myArgs.modeArg;
	// end parameters stuff

	Path root(datasetPath);

	std::cout << "[distordCrop] looking for input images : " << std::endl;
	std::vector< Path > imagePaths;
	directory_iterator it(root), eod;
	std::vector<sibr::Vector2i> resolutions;

	BOOST_FOREACH(Path const &p, std::make_pair(it, eod)) {
		if (is_regular_file(p) && ( p.extension() == ".jpg" || p.extension() == ".JPG" || p.extension() == ".PNG" || p.extension() == ".png" ) && appUtils.is_number(p.stem().string())) {

			std::cout << "\t " << p.filename().string() << std::endl;
			imagePaths.push_back(p);
		}
		else if (is_regular_file(p) && p.extension() == ".txt" && p.stem().string() == "resolutions") {
			
			// read resolutions file
			ifstream inputFile(p.string());

			std::string line;
			while (getline(inputFile, line)) {
				std::stringstream iss(line);
				std::string pathToImg;
				std::string widthStr;
				std::string heightStr;

				getline(iss, pathToImg, '\t');
				getline(iss, widthStr, '\t');
				getline(iss, heightStr, '\n');

				sibr::Vector2i res(std::stoi(widthStr), std::stoi(heightStr));

				resolutions.push_back(res);

			}

			inputFile.close();
		}
	}

	if (resolutions.size() == 0) {
		std::cout << "[distordCrop] WARNING : no resolution.txt file found" << std::endl;
		return 0;
	}

	if (imagePaths.size() == 0) {
		std::cout << "[distordCrop] WARNING: no images found: need .jpg,.JPG,.png,.PNG " << std::endl;
		return 0;
	}

	if (resolutions.size() != imagePaths.size()) {
		std::cout << "[distordCrop] WARNING : different number of input images and resolutions written in resolutions.txt" << std::endl;
		return 0;
	}

	int minWidth, minHeight, new_half_w, new_half_h;
	
	if (sameSize) {
		std::cout << " ALL IMG SHOULD HAVE SAME SIZE " << std::endl;
		sibr::Vector2i minSize = appUtils.findBiggestImageCenteredBox(root, imagePaths, resolutions, avgWidth, avgHeight, 
			PROCESSING_BATCH_SIZE, resolutionThreshold, threshold_ratio_bounding_box_size, backgroundColor, 
			threshold_black_color, thinest_bounding_box_size, toleranceFactor);

		std::cout << "[distordCrop] minSize " << minSize[0] << "x" << minSize[1] << std::endl;
		minWidth = minSize[0];
		minHeight = minSize[1];
	} else {
		std::cout << " ALL IMG SHOULD NOT HAVE SAME SIZE " << std::endl;
		sibr::Vector2i minSize = appUtils.findMinImageSize(root, imagePaths);
		minWidth = minSize[0];
		minHeight = minSize[1];
	}

	new_half_w = (minWidth % 2 == 0) ? (minWidth / 2) : (--minWidth / 2);
	new_half_h = (minHeight % 2 == 0) ? (minHeight / 2) : (--minHeight / 2);

	while ((new_half_w % 4) != 0) { --new_half_w; }
	while ((new_half_h % 4) != 0) { --new_half_h; }

	std::string outputFilePath = root.string() + "/cropNewSize.txt";
	std::ofstream file(outputFilePath, std::ios::trunc);
	if (file) {
		file << 2 * new_half_w << " " << 2 * new_half_h;
		file.close(); 
	}
	else {
		std::cout << "[distordCrop]  ERROR cant open file : " << outputFilePath << std::endl;
		return 1;
	}

	std::cout << "[distordCrop] done, new size is " << 2 * new_half_w << " x " << 2 * new_half_h << std::endl;

	return 0;
}
