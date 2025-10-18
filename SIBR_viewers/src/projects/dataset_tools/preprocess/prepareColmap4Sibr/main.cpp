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


#define PROGRAM_NAME "prepareColmap4Sibr"
using namespace sibr;

const char* usage = ""
"Usage: " PROGRAM_NAME " -path " "\n"
;

struct ColmapPreprocessArgs : public BasicIBRAppArgs {
	Arg<bool> fix_metadata = { "fix_metadata", "Fix scene_metadata after crop and distort " };
};

int main(const int argc, const char** argv)
{

	CommandLineArgs::parseMainArgs(argc, argv);
	ColmapPreprocessArgs myArgs;

	std::string pathScene = myArgs.dataset_path;

	std::vector<std::string> dirs = { "sfm_mvs_cm" , "sibr_cm" };

	std::ofstream outputSceneMetadata;

	if( myArgs.fix_metadata ) {
		std::string cm_path = myArgs.dataset_path.get() + "/sibr_cm";
		myArgs.dataset_path = cm_path;

		BasicIBRScene cm_scene(myArgs, true, true);

		std::vector<InputCamera::Ptr>	cams = cm_scene.cameras()->inputCameras();

		std::string tmpFileName = cm_path +  "/scene_metadata_tmp.txt"; 
		// done in a second pass, when everything has been created.
		outputSceneMetadata.open(tmpFileName);

		// overwrite previous version since image sizes have changed when running sibr preprocessing
		outputSceneMetadata << "Scene Metadata File\n" << std::endl;

		if (outputSceneMetadata.bad())
			SIBR_ERR << "Problem writing new metadata file" << std::endl;

		SIBR_LOG << "Writing new scene_metadata.txt file " << cm_path + "/scene_metadata.txt" << std::endl;

		outputSceneMetadata << "[list_images]\n<filename> <image_width> <image_height> <near_clipping_plane> <far_clipping_plane>" << std::endl;

		for (int c = 0; c < cams.size(); c++) {
			InputCamera & camIm = *cams[c];

			std::string extensionFile = boost::filesystem::extension(camIm.name());
			std::ostringstream ssZeroPad;
			ssZeroPad << std::setw(8) << std::setfill('0') << camIm.id();
			std::string newFileName = ssZeroPad.str() + extensionFile;
			// load image
			std::string imgpath = cm_path + "/images/" + camIm.name();
			sibr::ImageRGB im;
			if (!im.load(imgpath, false))
				SIBR_ERR << "Cant open image " << imgpath << std::endl;

			std::cerr << newFileName << " " << im.w() << " " << im.h() << " " << camIm.znear() << " " << camIm.zfar() << std::endl;
			outputSceneMetadata << newFileName << " " << im.w() << " " << im.h() << " " << camIm.znear() << " " << camIm.zfar() << std::endl;
		}

		outputSceneMetadata << "\n// Always specify active/exclude images after list images\n\n[exclude_images]\n<image1_idx> <image2_idx> ... <image3_idx>" << std::endl;

		for (int i = 0; i < cm_scene.data()->activeImages().size(); i++) {
			if (!cm_scene.data()->activeImages()[i])
				outputSceneMetadata << i << " ";
		}
		outputSceneMetadata << "\n\n\n[other parameters]" << std::endl;
		outputSceneMetadata.close();

		std::string SMName = cm_path +  "/scene_metadata.txt"; 

		SIBR_LOG << "Copying " << tmpFileName << " to " << SMName << std::endl;
		boost::filesystem::copy_file( tmpFileName, SMName, boost::filesystem::copy_option::overwrite_if_exists);
		boost::filesystem::remove(tmpFileName);

		exit(0);
	}
	std::cout << "Creating bundle file for SIBR scene." << std::endl;
	BasicIBRScene scene(myArgs, true, true);

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

	outputBundleCam.open(pathScene + "/sfm_mvs_cm/bundle.out");
	outputListIm.open(pathScene + "/sfm_mvs_cm/list_images.txt");
	outputBundleCam << "# Bundle file v0.3" << std::endl;
	outputBundleCam << maxCam << " " << 0 << std::endl;

	outputSceneMetadata.open(pathScene +  "/sibr_cm/scene_metadata.txt");
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

		boost::filesystem::copy_file(pathScene + "/colmap/stereo/images/" + camIm.name(), pathScene + "/sfm_mvs_cm/" + newFileName, boost::filesystem::copy_option::overwrite_if_exists);
		// keep focal
		outputBundleCam << camIm.toBundleString(false, true);
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

	std::vector<std::vector<std::string>> meshPathList = {
		{ "/capreal/mesh.ply", 			"/sfm_mvs_cm/recon.ply"},
		{ "/capreal/mesh.obj", 			"/sfm_mvs_cm/recon.ply"},
		{ "/capreal/mesh.mtl", 			"/sfm_mvs_cm/"},
		{ "/capreal/texture.png", 			"/sfm_mvs_cm/"},
		{ "/capreal/mesh_u1_v1.png", 	"/sfm_mvs_cm/"},
		{ "/colmap/stereo/meshed-delaunay.ply", 			"/sfm_mvs_cm/recon.ply"},
	};

	bool success = false;
	for(const std::vector<std::string> & meshPaths : meshPathList) {
		if(boost::filesystem::exists(pathScene + meshPaths[0])) {
			sibr::copyFile(pathScene + meshPaths[0], pathScene + meshPaths[1], true);
			success = true;
		}
	}
	if (!success) {
		std::cerr << "Couldnt file proxy geometry in any of the following places ";
		for (const std::vector<std::string>& meshPaths : meshPathList)
			std::cerr << pathScene + meshPaths[0] << std::endl;
		SIBR_ERR << "No proxy geometry, exiting" << std::endl;

	}

	return EXIT_SUCCESS;
}
