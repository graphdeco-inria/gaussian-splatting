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
#include <core/raycaster/CameraRaycaster.hpp>
#include <core/assets/ImageListFile.hpp>
#include <core/system/Utils.hpp>

/*
generate clipping_planes.txt file
*/
const char* USAGE						= "Usage: clippingPlanes <dataset-path> \n";
const char* TAG							= "[clippingPlanes]";

using namespace sibr;


int main(const int argc, const char** argv)
{

	if (argc < 1 + 1)
	{
		std::cout << USAGE << std::endl;
		return 1;
	}

	std::string		datasetPath = argv[1];

	if (directoryExists(datasetPath) == false) {
		SIBR_ERR << "Wrong program options, check the usage.";
		return 1;
	}

	// load rest of the things
	std::vector<InputCamera::Ptr>	inCams = InputCamera::load(datasetPath);
	ImageListFile				imageListFile;
	Mesh						proxy(false);

	// check needed things are there
	if (imageListFile.load(datasetPath + "/images/list_images.txt") == false && imageListFile.load(datasetPath + "/list_images.txt") == false)
		return 1;

	if ((proxy.load(datasetPath + "/meshes/pmvs_recon.ply") == false) && (proxy.load(datasetPath + "/meshes/mesh.ply") == false) && (proxy.load(datasetPath + "/pmvs_recon.ply") == false) && (proxy.load(datasetPath + "/recon.ply") == false) && (proxy.load(datasetPath + "/meshes/recon.ply") == false))
		return 1;

	const std::string clipping_planes_file_path = datasetPath + "/clipping_planes.txt";
	if (!sibr::fileExists(clipping_planes_file_path)) {

		std::vector<sibr::Vector2f> nearsFars;
		CameraRaycaster::computeClippingPlanes(proxy, inCams, nearsFars);

		std::ofstream file(clipping_planes_file_path, std::ios::trunc | std::ios::out);
		if (file) {
			for (const auto & nearFar : nearsFars) {
				if (nearFar[0] > 0 && nearFar[1] > 0) {
					file << nearFar[0] << ' ' << nearFar[1] << std::endl;
				}
				else {
					/** \todo [SP]Temporary fix. Ideally we should exclude these images. */
					file << "0.1 100.0" << std::endl;
				}
			}
			file.close();
		}
		else {
			SIBR_WRG << " Could not save file '" << clipping_planes_file_path << "'." << std::endl;
		}
	}

	std::cout << TAG << " done!\n";
	return 0;
}