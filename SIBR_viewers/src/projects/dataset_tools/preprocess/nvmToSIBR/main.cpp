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


#include <fstream>
#include <iostream>
#include <core/system/CommandLineArgs.hpp>
#include <core/scene/BasicIBRScene.hpp>
#include <core/raycaster/CameraRaycaster.hpp>
#include <core/assets/ImageListFile.hpp>
#include <core/system/Utils.hpp>


#define PROGRAM_NAME "sibr_nvm_to_sibr"
using namespace sibr;

const char* usage = ""
"Usage: " PROGRAM_NAME " -path " "\n"
;

struct ColmapPreprocessArgs :
	virtual BasicIBRAppArgs {
};

int main(const int argc, const char** argv)
{

	CommandLineArgs::parseMainArgs(argc, argv);
	ColmapPreprocessArgs myArgs;

	std::string pathScene = myArgs.dataset_path;

	std::vector<std::string> dirs = { "cameras", "images", "meshes"};

	std::cout << "Generating SIBR scene." << std::endl;
	BasicIBRScene scene(myArgs, true);

	// load the cams
	std::vector<InputCamera::Ptr>	cams = scene.cameras()->inputCameras();
	const int maxCam = int(cams.size());
	const int minCam = 0;

	for (auto dir : dirs) {
		std::cout << dir << std::endl;
		if (!directoryExists(pathScene + "/" + dir.c_str())) {
			makeDirectory(pathScene + "/" + dir.c_str());
		}
	}

	std::ofstream outputBundleCam;
	std::ofstream outputListIm;
	std::ofstream outputSceneMetadata;

	outputBundleCam.open(pathScene + "/cameras/bundle.out");
	outputListIm.open(pathScene + "/images/list_images.txt");
	outputSceneMetadata.open(pathScene + "/scene_metadata.txt");
	outputBundleCam << "# Bundle file v0.3" << std::endl;
	outputBundleCam << maxCam << " " << 0 << std::endl;
	outputSceneMetadata << "Scene Metadata File\n" << std::endl;
	outputSceneMetadata << "[list_images]\n<filename> <image_width> <image_height> <near_clipping_plane> <far_clipping_plane>" << std::endl;

	std::sort(cams.begin(), cams.end(), [](const InputCamera::Ptr & a, const InputCamera::Ptr & b) {
		return a->id() < b->id();
	});

	for (int c = minCam; c < maxCam; c++) {
		InputCamera & camIm = *cams[c];

		std::string extensionFile = boost::filesystem::extension(camIm.name());
		std::ostringstream ssZeroPad;
		ssZeroPad << std::setw(8) << std::setfill('0') << camIm.id();
		std::string newFileName = ssZeroPad.str() + extensionFile;

		boost::filesystem::copy_file(pathScene + "/nvm/" + camIm.name(), pathScene + "/images/" + newFileName, boost::filesystem::copy_option::overwrite_if_exists);
		outputBundleCam << camIm.toBundleString();
		outputListIm << newFileName << " " << camIm.w() << " " << camIm.h() << std::endl;
		outputSceneMetadata << newFileName << " " << camIm.w() << " " << camIm.h() << " " << camIm.znear() << " " << camIm.zfar() << std::endl;
	}


	outputSceneMetadata << "\n// Always specify active/exclude images after list images\n\n[exclude_images]\n<image1_idx> <image2_idx> ... <image3_idx>" << std::endl;

	for (int i = 0; i < scene.data()->activeImages().size(); i++) {
		if (!scene.data()->activeImages()[i])
			outputSceneMetadata << i << " ";
	}

	outputSceneMetadata << "\n\n\n[other parameters]" << std::endl;

	outputBundleCam.close();
	outputListIm.close();
	outputSceneMetadata.close();

	const std::string meshPath = pathScene + "/capreal/mesh.ply";
	sibr::copyFile(meshPath, pathScene + "/meshes/recon.ply", true);

	return EXIT_SUCCESS;
}