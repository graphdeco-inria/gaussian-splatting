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


#include "core/system/CommandLineArgs.hpp"
#include "core/assets/InputCamera.hpp"

using namespace sibr;

/* Camera converter args. */
struct CameraConverterArgs : virtual AppArgs {
	RequiredArg<std::string> input = { "input",  "input camera file" };
	RequiredArg<std::string> output = { "output",   "output camera file" };
	RequiredArg<std::string> colmapPath = { "colmapPath",   "path to colmap recon for camera file" };
	Arg<std::string> transfo = { "transfo",  "", "matrix file" };
	Arg<sibr::Vector2u> inputRes = {"ires", {1920, 1080}, "input camera resolution (not required for all formats)"};
	Arg<sibr::Vector2u> outputRes = { "ores", {1920, 1080}, "output camera resolution (not required for all formats)" };
	Arg<bool> inverse = {"inverse",  "reverse the transformation"};
	Arg<bool> bundleImageList = { "images_list", "for a bundle output, output list_images.txt" };
	Arg<bool> bundleImageFiles = { "images_files",  "for a bundle output, output empty images in a 'visualize' subdirectory" };
	Arg<std::string> inImageFilePath = { "in_images_files", "", "for a bundle input images file directory (for list_images etc)" };
	Arg<float> scale = { "scale", 1.0, "scale images for cameras.txt file" };
};

/* SIBR binary path loader helper.
 * \param filename the .path binary file
 * \param cams will be populated with the loaded cameras
 * \param res the image resolution (for aspect ratio)
 * \return the loading status
 */
bool load(const std::string& filename, std::vector<InputCamera::Ptr> & cams, const sibr::Vector2u & res){
	sibr::ByteStream stream;
	if(!stream.load(filename)) {
		return false;
	}
	int32 num = 0;
	stream >> num;
	while (num > 0) {
		Camera cam;
		stream >> cam;
		cams.push_back(std::make_shared<InputCamera>(cam, res[0], res[1]));
		--num;
	}
	return stream;
}

/** SIBR binary path saver helper.
 * \param filename the .path output file
 * \param cams the cameras to save
 */
void save(const std::string& filename, const std::vector<InputCamera::Ptr> & cams){
	sibr::ByteStream stream;
	const int32 num = int32(cams.size());
	stream << num;
	for (const InputCamera::Ptr& cam : cams) {
		Camera subcam(*cam);
		stream << subcam;
	}
	stream.saveToFile(filename);
}

	

void colmapSave(const std::string& filename, const std::vector<InputCamera::Ptr> & xformPath, float scale, float focaly, float focalx) {
	// save as colmap images.txt file
	sibr::Matrix3f converter;
	converter << 1, 0, 0,
		     0, -1, 0,
		     0, 0, -1;
	
	std::ofstream outputColmapPath, outputColmapPathCams;
	std::string colmapPathCams = parentDirectory(filename) + std::string("/cameras.txt");

	std::cerr << std::endl;
	std::cerr << std::endl;
	std::cerr << "Writing colmap path to " << parentDirectory(filename) << std::endl;
	
	outputColmapPath.open(filename);
	if(!outputColmapPath.good())
		SIBR_ERR << "Cant open output file " <<  filename << std::endl;
	outputColmapPathCams.open(colmapPathCams);

	outputColmapPathCams << "# Camera list with one line of data per camera:" << std::endl;
	outputColmapPathCams << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
	outputColmapPathCams << "# Number of cameras: 1" << std::endl;
	if (focalx == -1) {
		focalx = xformPath[0]->focal() * xformPath[0]->aspect(); // use aspect ratio
		SIBR_WRG << "No focal x given making it equal to focaly * aspect ratio; use result at own risk. Should have a colmap dataset as input" << std::endl;
	}
	else
	{
		std::cerr << "FX " << focalx << std::endl;
		focalx = xformPath[0]->focal() * (focalx / focaly);
		SIBR_WRG << "Focal x set to f / (fx/fy); f of first image :" << focalx<<std::endl;
	}
	for(int i=0; i<xformPath.size(); i++)  {
		outputColmapPathCams << i+1 << " PINHOLE " << xformPath[0]->w()*scale << " " << xformPath[0]->h()*scale
			<< " " << xformPath[0]->focal()*scale << " " << focalx*scale 
			<< " " << xformPath[0]->w()*scale * 0.5 << " " << xformPath[0]->h()*scale * 0.5 << std::endl;
	}


	outputColmapPath<< "# Image list with two lines of data per image:" << std::endl;
	outputColmapPath<< "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME" << std::endl;
	outputColmapPath<< "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
	for(int i=0; i<xformPath.size(); i++)  {
		sibr::Matrix3f tmp = xformPath[i]->rotation().toRotationMatrix() * converter;
		sibr::Matrix3f Qinv = tmp.transpose();
		sibr::Quaternionf q = quatFromMatrix(Qinv);
		sibr::Vector3f t = -Qinv*xformPath[i]->position();

		outputColmapPath << i << " " <<  q.w() << " " <<  -q.x() << " " <<  -q.y() << " " <<  -q.z() << " " <<  
			t.x() << " " << t.y() << " " << t.z() << " " << 1 << " " << "pathImage"<<i << std::endl;
		outputColmapPath << std::endl; // empty line, no points
	}
	outputColmapPath.close();
	outputColmapPathCams.close();
}


/** The camera converter is a utility to convert a camera path from a file format to another, with additional options.
 * You can specify a 4x4 transformation matrix to apply, stored in a txt file (values separated by spaces/newlines, f.i. the output of CloudCompare alignment).
 * If you want to apply the inverse transformation, use the --inverse flag.
 * Supported inputs: path (*), lookat (*), bundle, nvm (*), colmap
 * Supported outputs: path, bundle(*), lookat, colmap (images.txt)
 * (*): requires an input/output resolution (often only for the aspect ratio).
 * */
int main(int ac, char** av) {

	// Parse Command-line Args
	sibr::CommandLineArgs::parseMainArgs(ac, av);
	const CameraConverterArgs args;

	// Load cameras.
	std::vector<InputCamera::Ptr> cams;
	const std::string ext = sibr::getExtension(args.input);
	if(ext == "path") {
		load(args.input, cams, args.inputRes);
	} else if(ext == "lookat") {
		cams = InputCamera::loadLookat(args.input, { args.inputRes });
	} else if (ext == "out") {
		if (std::string(args.inImageFilePath) == "")
			SIBR_ERR << "Please provide image file directory for bundler input (use option -in_images_files DIRECTORY_CONTAINING_LIST_IMAGES.txt )\nIf necessary use the generate_list_images.py script to generate list_images.txt " << std::endl;
		cams = InputCamera::loadBundle(args.input, 0.01, 1000, args.inImageFilePath, true);
	} else if (ext == "nvm") {
		cams = InputCamera::loadNVM(args.input, 0.01f, 1000.0f, {args.inputRes});
	} else if (sibr::directoryExists(args.input)) {
		// If we got a directory, assume colmap sparse.
		cams = InputCamera::loadColmap(args.input, 0.01, 1000, 1);
	} else {
		SIBR_ERR << "Unsupported path file extension: " << ext << "." << std::endl;
		return EXIT_FAILURE;
	}
	SIBR_LOG << "Loaded " << cams.size() << " cameras." << std::endl;

	float focaly = cams[0]->focal(); // y by default
	float focalx = cams[0]->focalx();

	// if a path is given try and get focalx
	std::vector<InputCamera::Ptr> camsFx;
	if (args.colmapPath != "") {
		std::cerr << "COLMAP " << args.colmapPath << std::endl;
		std::string cm_sparse_path = args.colmapPath.get() + "/stereo/sparse";
		if (directoryExists(cm_sparse_path)) {
			camsFx = InputCamera::loadColmap(cm_sparse_path, 0.01, 1000, 1);
			std::cerr << "Found " << camsFx.size() << " cameras fovx " << camsFx[0]->focalx() << std::endl;
			focalx = camsFx[0]->focalx();
		}
		else
			std::cerr << "Cant find " << cm_sparse_path << std::endl;
	}

	// Load the transformation.
	std::ifstream transFile(args.transfo.get());
	sibr::Matrix4f transf = sibr::Matrix4f::Identity();
	if (transFile.is_open()) {
		for (int i = 0; i < 16; ++i) {
			float f;
			transFile >> f;
			transf(i) = f;
		}
		transFile.close();
	}
	transf.transposeInPlace();
	if (args.inverse) {
		transf = transf.inverse().eval();
	}

	// Apply transformation to each camera keypoints, if it's not identity.
	if(!transf.isIdentity()) {
		SIBR_LOG << "Applying transformation: " << std::endl << transf << std::endl;
		for (auto & cam : cams) {
			sibr::Vector3f pos = cam->position();
			sibr::Vector3f center = cam->position() + cam->dir();
			sibr::Vector3f up = cam->position() + cam->up();
			pos = (transf * pos.homogeneous()).xyz();
			center = (transf * center.homogeneous()).xyz();
			up = (transf * up.homogeneous()).xyz();
			cam->setLookAt(pos, center, (up - pos).normalized());

		}
	}

	// Save cameras.
	const std::string outExt = sibr::getExtension(args.output);
	if (outExt == "path") {
		save(args.output, cams);
	} else if (outExt == "out") { // bundler
		std::vector<InputCamera::Ptr> outCams;
		for(const auto & cam : cams) {
			const int outH = int(args.outputRes.get()[1]);
			const int outW = int(std::round(cam->aspect() * float(outH)));
			InputCamera::Ptr oc;
			outCams.push_back(oc=std::make_shared<InputCamera>(*cam, outW, outH));
			// reset focal
			oc->setFocal(cam->focal());
		}
		sibr::InputCamera::saveAsBundle(outCams, args.output, args.bundleImageList, args.bundleImageFiles, false);
	} else if (outExt == "lookat") {
		std::vector<InputCamera::Ptr> outCams;
		for (const auto& cam : cams) {
			const int outH = int(args.outputRes.get()[1]);
			const int outW = int(std::round(cam->aspect() * float(outH)));
			outCams.push_back(std::make_shared<InputCamera>(*cam, outW, outH));
		}
		sibr::InputCamera::saveAsLookat(outCams, args.output);
	}
	else if (getFileName(args.output) == "images.txt" ) { // colmap
		colmapSave(args.output, cams, args.scale, focaly, focalx);
	} else {
		SIBR_ERR << "Unsupported output file extension: " << outExt << "." << std::endl;
		return EXIT_FAILURE;
	}
	SIBR_LOG << "Saved transformed cameras to \"" << args.output.get() << "\"." << std::endl;
		
	return EXIT_SUCCESS;
}


