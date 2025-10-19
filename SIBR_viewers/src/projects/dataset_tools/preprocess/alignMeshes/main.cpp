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
#include <boost/filesystem.hpp>
#include <core/system/Utils.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <core/renderer/DepthRenderer.hpp>
#include <core/graphics/Window.hpp>
#include <core/scene/BasicIBRScene.hpp>
#include <core/scene/ParseData.hpp>

#define PROGRAM_NAME "sibr_chunk2sibr"
using namespace sibr;

const char* usage = ""
"Usage: " PROGRAM_NAME " -path <reference scene path> -path2 <scene to align path> -outPath <mesh output path>" "\n"
;

double distPatch(sibr::ImageRGB& im1, sibr::Vector2i& tpos, sibr::ImageRGB& im2, sibr::Vector2i& spos, int size) {
	//only need to check boundaries for target


	double dist = 0;
	for (int i = -size; i <= size; i++) {
		for (int j = -size; j <= size; j++) {
			sibr::Vector2i debug(spos.x() + i, spos.y() + j);
			if (!im2.isInRange(debug.x(), debug.y()))
				std::cout << "Pos patch is : " << sibr::Vector2i(spos.x() + i, spos.y() + j) << std::endl;
			dist += (im1(tpos.x() + i, tpos.y() + j).cast<double>() - im2(spos.x() + i, spos.y() + j).cast<double>()).squaredNorm();
		}
	}
	return dist;
}


// Convienience function to find the Median Absolute Deviation from a vector of deviations.
// Note that it isn't a strict median( just a quick approximation. )
float findMAD(const Eigen::VectorXf& vec) {
	Eigen::VectorXf vec_ = vec;
	// Sort the data in increasing order.
	std::sort(vec_.data(), vec_.data() + vec_.size());
	// Return the 'middle' element.
	return vec_[((vec_.size() + 1) / 2)];
}

// Weight function.( Takes a list of standardized adjusted residuals and returns the square-root-weights for each one )
// Currently, the Bisquares estimator is used.
// Note that this function should return the square root of the actual weight value since both X and Y are multiplied by this vector.
Eigen::VectorXf weight(Eigen::VectorXf v) {
	Eigen::VectorXf vout = v;

	for (int i = 0; i < v.size(); i++) {
		float r = v[i];
		vout[i] = ((abs(r) < 1) ? (1 - (r * r)) : 0);
	}

	return vout;
}

#define MAX_ITERS 100
// Procedure for IRLS( Iterative Reweighted Least Squares ).
void irls(Eigen::MatrixX4f mX, Eigen::VectorXf vY, Eigen::Vector4f& mCoeffs, float tune) {

	Eigen::MatrixXf mX_ = mX;
	Eigen::VectorXf vY_ = vY;
	// Find the least squares coefficients.
	Eigen::Vector4f vC = mX_.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(vY);
	//Log(EInfo, "Finished solving for LS solution.");

	// Form the leverage value matrix as H = X . ( X_T . X ) . X_T
	// Form the leverage factor matrix as 1/sqrt(1 - diag(H))
	Eigen::VectorXf mH = (mX_ * (((mX_.transpose() * mX_).inverse()) * mX_.transpose())).diagonal();
	Eigen::MatrixXf mH_ = (Eigen::VectorXf::Constant(mH.rows(), 1, 1) - mH).cwiseSqrt().cwiseInverse().asDiagonal();

	std::cout << vC << std::endl;
	for (int i = 0; ; i++) {
		std::cout << "IRLS: Iteration " << i << ":";

		// Find residuals:
		Eigen::VectorXf resid = vY - mX * vC;

		float mad = findMAD(resid.cwiseAbs());

		// Calcualte Standardized Adjusted Residuals.
		Eigen::VectorXf r = (mH_ * resid * 0.6745) / (mad * tune);

		// Find the root weight of residuals.
		Eigen::VectorXf wt = weight(r);

		// Multiply X and Y with the root of the weights.
		mX_ = wt.asDiagonal() * mX;
		vY_ = wt.asDiagonal() * vY;

		std::cout << "MAD= " << mad << ", ";

		// Regress the weighted X and Y to find weighted least squares optimisation.
		Eigen::Vector4f vC_ = mX_.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(vY_);

		// Find mean deviation in coefficients.
		float meanDiff = (vC - vC_).cwiseAbs().mean();
		std::cout << "MD=" << meanDiff << ",";
		// Terminate if the deviation is too small or number of iterations has been exceeded.
		if (meanDiff < 0.01f || i > MAX_ITERS) {
			mCoeffs = vC_;
			std::cout << "\n";
			break;
		}

		vC = vC_;
		std::cout << "\n";
	}
	std::cout << vC << std::endl;
}

static bool isRawRC(std::string pathRC)
{
	// do we have bundle, mesh and list images ?
	sibr::Mesh mesh2Align;
	if (!mesh2Align.load(pathRC + "/recon.ply")) {
		SIBR_WRG << "***** No file " << pathRC + "/recon.ply ; make sure your mesh has the correct name !!";
		return false;
	}
	if (!fileExists(pathRC + "/bundle.out")) {
		SIBR_WRG << "***** No file " << pathRC + "/bundle.out ; make sure your bundle file has the correct name !!";
		return false;
	}
	if (!fileExists(pathRC + "/list_images.txt")) {
		SIBR_WRG << "***** No file " << pathRC + "/list_images.txt ; make sure you generate the list_images.txt file ";
		return false;
	}
	return true;
}

static void loadRawRC(std::string pathRC, std::vector<sibr::InputCamera::Ptr>& cams2Align,
	std::vector<sibr::ImageRGB::Ptr>& imgs2Align, sibr::Mesh& mesh2Align)
{
	cams2Align = sibr::InputCamera::loadBundle(pathRC + "/bundle.out", 0.01f, 1000.0f, pathRC + "/list_images.txt");
	mesh2Align.load(pathRC + "/recon.ply");
	imgs2Align.resize(cams2Align.size());
	for (int c = 0; c < cams2Align.size(); c++) {
		sibr::ImageRGB::Ptr imgPtr;
		sibr::ImageRGB img;
		if (!img.load(pathRC + "/" + cams2Align[c]->name()))
			if (!img.load(pathRC + "/" + cams2Align[c]->name() + ".png"))
				if (!img.load(pathRC + "/" + cams2Align[c]->name() + ".jpg")) {
					SIBR_ERR << "Error loading dataset to align from " << pathRC;
					SIBR_ERR << "Problem loading images from raw RC, exiting ";
				}

		imgs2Align[c] = img.clonePtr();
	}
}

static bool isRawSynthetic(std::string pathSynthetic)
{
	// do we have bundle, mesh and list images ?
	sibr::Mesh mesh2Align;
	if (!fileExists(pathSynthetic + "/scene.obj")) {
		SIBR_WRG << "***** No file " << pathSynthetic + "/scene.obj ; make sure your mesh has the correct name !!";
		return false;
	}
	if (!fileExists(pathSynthetic + "/cameras.lookat")) {
		SIBR_WRG << "***** No file " << pathSynthetic + "/cameras.lookat ; make sure your bundle file has the correct name !!";
		return false;
	}
	if (!directoryExists(pathSynthetic + "/images")) {
		SIBR_WRG << "***** No file " << pathSynthetic + "/images ; make sure you have images folder inside the scene ";
		return false;
	}
	return true;
}

static void loadRawSynthetic(std::string pathSynthetic, std::vector<sibr::InputCamera::Ptr>& cams2Align,
	std::vector<sibr::ImageRGB::Ptr>& imgs2Align, sibr::Mesh& mesh2Align)
{
	cams2Align = sibr::InputCamera::loadLookat(pathSynthetic + "/cameras.lookat", std::vector<sibr::Vector2u>{sibr::Vector2u(1920, 1080)}, 0.01f, 1000.0f);
	SIBR_WRG << "We assume a size of the synthetic images of 1920*1080. If it is not your case, this loading will not work properly";
	mesh2Align.load(pathSynthetic + "/scene.obj");
	imgs2Align.resize(cams2Align.size());
	for (int c = 0; c < cams2Align.size(); c++) {
		sibr::ImageRGB::Ptr imgPtr;
		sibr::ImageRGB img;
		if (!img.load(pathSynthetic + "/images/" + cams2Align[c]->name()))
			if (!img.load(pathSynthetic + "/images/" + cams2Align[c]->name() + ".png"))
				if (!img.load(pathSynthetic + "/images/" + cams2Align[c]->name() + ".jpg")) {
					SIBR_ERR << "Error loading dataset to align from " << pathSynthetic;
					SIBR_ERR << "Problem loading images from raw RC, exiting ";
				}

		imgs2Align[c] = img.clonePtr();
	}
}


int assignImages(
	std::vector<sibr::ImageRGB::Ptr>& imgs2Align, std::vector<sibr::ImageRGB>& imgs2AlignSmall,
	std::vector<sibr::ImageRGB::Ptr>& imgsRef, std::vector<sibr::ImageRGB>& imgsRefSmall,
	std::map<int, int>& alignCamToRef, std::vector<sibr::InputCamera::Ptr>& camsRef, std::vector<sibr::InputCamera::Ptr> cams2Align,
	int resizeW, std::set<int>& assignedCam, float threshold)
{
	int assignCnt = 0;
	std::cout << "Assigning " << imgs2Align.size() << " cameras from the set to align to the fixed one: ";
	//We then look for closest match and assign it only if the distance between the images is half the median distance
	//This prevent issues in the case were a camera is missing from one set
	for (int i = 0; i < imgs2Align.size(); i++) {

		std::cout << "Assigning camera " << i << ", ";

		sibr::ImageRGB& im2Align = imgs2AlignSmall[i];
		sibr::Vector2i pos2Align(resizeW / 2, im2Align.h() / 2);

		double minImDist = DBL_MAX;
		int bestIm = -1;

		std::vector<double> dists;
		std::cerr << "2 ALIGN TESTING " << std::endl;

		cv::Rect centerROI(im2Align.w() / 8, im2Align.h() / 8, 6 * im2Align.w() / 8, 6 * im2Align.h() / 8);

		for (int j = 0; j < imgsRef.size(); j++) {
			if (assignedCam.find(j) != assignedCam.end())
				continue;

			sibr::ImageRGB& imRef = imgsRefSmall[j];
			double minDist = DBL_MAX;
			int wIm = imRef.w();
			int hIm = imRef.h();

			for (int dx = -wIm / 8; dx <= wIm / 8; dx += 4) {
				for (int dy = -hIm / 8; dy <= hIm / 8; dy += 4) {
					cv::Rect shiftROI(dx + wIm / 8, dy + hIm / 8, 6 * wIm / 8, 6 * hIm / 8);
					double d = cv::norm(imRef.toOpenCV()(shiftROI), im2Align.toOpenCV()(centerROI));
					if (d < minDist)
					{
						minDist = d;
					}
				}
			}

			dists.push_back(minDist);

			if (minDist < minImDist) {
				minImDist = minDist;
				bestIm = j;

				//show(imRef);
				//show(im2Align);
			}
		}

		std::sort(dists.begin(), dists.end());
		std::cerr << " SIZe " << dists.size() << " min " << minImDist << " half " << threshold * dists[dists.size() / 2] << std::endl;
		if (dists.size() > 5 && minImDist < threshold * dists[dists.size() / 2]) {
			alignCamToRef[i] = bestIm;
			assignedCam.emplace(bestIm);
			std::wcout << i << " -> " << bestIm << " -- " << cams2Align[i]->name().c_str() << " -> " << camsRef[bestIm]->name().c_str() << std::endl;
			assignCnt++;
		}
		else {
			alignCamToRef[i] = -1;
			std::wcout << i << " -> " << "Not assigned " << std::endl;
			std::wcout << i << " BEST MATCH -> " << bestIm << " -- " << cams2Align[i]->name().c_str() << " -> " << camsRef[bestIm]->name().c_str() << std::endl;
		}

		//show(imgs2Align[i]);
		//show(imgsRef[bestIm]);

	}
	return assignCnt;
}

InputCamera rot90CC(InputCamera::Ptr& in)
{
	InputCamera rotCam = *in;
	rotCam.size(rotCam.h(), rotCam.w());
	rotCam.aspect(1.0f / rotCam.aspect());
	rotCam.fovy(2.0 * atan(0.5 * rotCam.h() / rotCam.focal()));
	rotCam.setLookAt(rotCam.position(), rotCam.position() + rotCam.dir(), rotCam.right());
	return rotCam;
}


struct AlignMeshesArgs :
	virtual BasicIBRAppArgs {
	RequiredArg<std::string> pathRef = { "pathRef", "Path to the fixed scene" };
	RequiredArg<std::string> pathToAlign = { "path2Align", "Path to the scene to align" };
	RequiredArg<std::string> outPath = { "out", "Path to the folder where to write the transformed mesh and the matrix" };
	Arg<bool> forceLandscape = { "forceLandscape", "Option to force all images to be in landscape orientation before image assignation and correspondances computation" };
	Arg<bool> saveScene = { "saveScene", "If true saves entire scene, else only save the transformed mesh and transform.txt file in out dir"
	};
};

int main(int ac, char** av)
{
	// Parse Commad-line Args
	CommandLineArgs::parseMainArgs(ac, av);
	AlignMeshesArgs myArgs;

	//sibr::Window window(100, 100, "Window");
	sibr::Window		window(PROGRAM_NAME, sibr::Vector2i(50, 50), myArgs);
	int wRender, hRender;

	std::cout << "This method relies on images, cameras and meshes of both scenes." << std::endl;

	//	Here is the data strctures that we will use for this program to make it as generic as possible
	std::vector<sibr::ImageRGB::Ptr> imgsRef;
	std::vector<sibr::ImageRGB::Ptr> imgs2AlignOriginal;
	std::vector<sibr::ImageRGB::Ptr> imgs2Align;

	std::vector<sibr::ImageRGB> imgsRefSmall;
	std::vector<sibr::ImageRGB> imgs2AlignSmall;

	std::vector<sibr::InputCamera::Ptr> camsRef;
	std::vector<sibr::InputCamera::Ptr> cams2Align;

	sibr::Mesh meshRef;
	sibr::Mesh mesh2Align;
	//Create the two scenes

	//Load the reference data
	IParseData::Type refSceneType;
	if (myArgs.pathRef.get() == "")
		SIBR_ERR << "Reference path empty";
	BasicIBRAppArgs argsRefScene;
	argsRefScene.dataset_path = myArgs.pathRef.get();

	try {
		BasicIBRScene::Ptr		sceneRef(new BasicIBRScene(argsRefScene, true));

		if ((refSceneType = sceneRef->data()->datasetType()) != IParseData::Type::EMPTY) {
			meshRef = sceneRef->proxies()->proxy();
			imgsRef = sceneRef->images()->inputImages();
			camsRef = sceneRef->cameras()->inputCameras();
		}
		else 
			SIBR_ERR << "Error loading reference dataset from " << myArgs.pathRef.get();
	}

   	catch(...) {
		std::cout << "Trying to load Raw RealityCapture or Synthetic data" << std::endl;
		if (isRawRC(argsRefScene.dataset_path)) {  // try "raw RC" option
			loadRawRC(argsRefScene.dataset_path, camsRef, imgsRef, meshRef);
		}
		else if (isRawSynthetic(argsRefScene.dataset_path)) {
			loadRawSynthetic(argsRefScene.dataset_path, camsRef, imgsRef, meshRef);
		}
		else 
			SIBR_ERR << "Error loading reference dataset from " << myArgs.pathRef.get();
   	}


	//Load the data for the scene to align	
	if (myArgs.pathToAlign.get() == "")
		SIBR_ERR << "Path to mesh to align empty";
	BasicIBRAppArgs argsAlignScene;
	argsAlignScene.dataset_path = myArgs.pathToAlign.get();
	// 
	try {
		BasicIBRScene::Ptr		sceneAlign(new BasicIBRScene(argsAlignScene, true));

		if (sceneAlign->data()->datasetType() != IParseData::Type::EMPTY) {
			mesh2Align = sceneAlign->proxies()->proxy();
			imgs2AlignOriginal = sceneAlign->images()->inputImages();
			cams2Align = sceneAlign->cameras()->inputCameras();
		}
		else {
			SIBR_ERR << "Error loading dataset to align from " << myArgs.pathToAlign.get();
		}
	}


   	catch(...) {
		std::cout << "Trying to load Raw RealityCapture or Synthetic data" << std::endl;
		if (isRawRC(argsAlignScene.dataset_path)){ // try "raw RC" option
			loadRawRC(argsAlignScene.dataset_path, cams2Align, imgs2AlignOriginal, mesh2Align);
		}
		else if (isRawSynthetic(argsAlignScene.dataset_path)) {
			loadRawSynthetic(argsAlignScene.dataset_path, cams2Align, imgs2AlignOriginal, mesh2Align);
		}
		else
			SIBR_ERR << "Error loading dataset to align from " << myArgs.pathToAlign.get();
   	}
	
	if (myArgs.forceLandscape) {
		for (int c = 0; c < camsRef.size(); c++) {
			if (imgsRef[c]->h() > imgsRef[c]->w()) {
				//rotate the image
				cv::rotate(imgsRef[c]->toOpenCV(), imgsRef[c]->toOpenCVnonConst(), cv::ROTATE_90_COUNTERCLOCKWISE);
				//rotate the camera
				*camsRef[c] = rot90CC(camsRef[c]);
			}

		}
		for (int c = 0; c < cams2Align.size(); c++) {
			if (imgs2AlignOriginal[c]->h() > imgs2AlignOriginal[c]->w()) {
				cv::rotate(imgs2AlignOriginal[c]->toOpenCV(), imgs2AlignOriginal[c]->toOpenCVnonConst(), cv::ROTATE_90_COUNTERCLOCKWISE);
				//rotate the camera
				*cams2Align[c] = rot90CC(cams2Align[c]);
			}
		}

	}


	//We resize the input images to the same width to account for possible rescale between the two scenes
#pragma omp parallel for
	for (int c = 0; c < camsRef.size(); c++) {
		*imgsRef[c] = imgsRef[c]->resized(1024, 1024.0f * imgsRef[c]->h() / imgsRef[c]->w(), cv::INTER_LINEAR);
	}

	for (int c = 0; c < cams2Align.size(); c++) {
		imgs2Align.push_back(sibr::ImageRGB::Ptr(new sibr::ImageRGB()));
	}
#pragma omp parallel for
	for (int c = 0; c < cams2Align.size(); c++) {
		*imgs2Align[c] = imgs2AlignOriginal[c]->resized(1024, 1024.0f * imgs2AlignOriginal[c]->h() / imgs2AlignOriginal[c]->w(), cv::INTER_LINEAR);
	}

	//Create output dir
	std::string outPath;
	outPath = myArgs.outPath.get();
	//First create all the needed directories
	sibr::makeDirectory(outPath);

	//We now match images between the two scenes as we cannot rely on correspondance between cameras
	//First we make all image 512*512
	int resizeW = 512;
	cv::Rect centerROI(resizeW / 4, resizeW / 4, resizeW / 2, resizeW / 2);
	std::map<int, int> alignCamToRef;
	std::set<int> assignedCam;

	std::cout << "Resizing images" << std::endl;

	imgs2AlignSmall.resize(imgs2Align.size());
#pragma omp parallel for
	for (int i = 0; i < imgs2Align.size(); i++) {
		imgs2AlignSmall[i] = imgs2Align[i]->resized(resizeW, resizeW, cv::INTER_AREA);
		imgs2AlignSmall[i].fromOpenCV(imgs2AlignSmall[i].toOpenCV()(centerROI));
	}
	imgsRefSmall.resize(imgsRef.size());
#pragma omp parallel for
	for (int i = 0; i < imgsRef.size(); i++) {
		imgsRefSmall[i] = imgsRef[i]->resized(resizeW, resizeW, cv::INTER_AREA);
		imgsRefSmall[i].fromOpenCV(imgsRefSmall[i].toOpenCV()(centerROI));
	}

	int cnt = assignImages(imgs2Align, imgs2AlignSmall, imgsRef, imgsRefSmall, alignCamToRef, camsRef, cams2Align, resizeW, assignedCam, 0.7);
	std::cout << "Assigned " << cnt << std::endl;

	/////////////////
	/////////////////
	// Now we will compute closely matched feature between pair of images
	/////////////////
	/////////////////

	std::vector<sibr::Vector3f> listFeatPRef;
	std::vector<sibr::Vector3f> listFeatP2Align;

	std::vector<float> dist2CamRef;
	std::vector<float> dist2Cam2Align;

	//Maximum shift for patch alignement
	int shiftMax = 16;
	const int patchRadius = 8;

	for (int im = 0; im < imgs2Align.size(); im++) {

		if (alignCamToRef[im] < 0)
			continue;

		std::cout << "IM " << im << std::endl;

		sibr::ImageRGB& imRef = *imgsRef[alignCamToRef[im]];
		sibr::ImageRGB& im2Align = *imgs2Align[im];

		// To see feature match in images
		sibr::ImageRGB imRefCopy = imRef.clone();
		sibr::ImageRGB im2AlignCopy = im2Align.clone();

		//Compute im center to transpose position from imref to im2Align. The center should stay aligned with the crop.
		//Small errors are absorbed by the shift estimation.
		sibr::Vector2i imRefCenter(imRef.w() / 2, imRef.h() / 2);
		sibr::Vector2i im2AlignCenter(im2Align.w() / 2, im2Align.h() / 2);

		//Get the two cameras
		sibr::InputCamera::Ptr camRef = camsRef[alignCamToRef[im]];
		sibr::InputCamera::Ptr cam2Align = cams2Align[im];

		//Depth map for 3D position estimation
		sibr::ImageL32F depthMapRef, depthMap2Align;

		//We render the two depth map of the two meshes to be able to recover 3D position from pixel position
		int wCamRef, hCamRef, wCam2Align, hCam2Align;
		wCamRef = camRef->w();
		hCamRef = camRef->h();
		wCam2Align = cam2Align->w();
		hCam2Align = cam2Align->h();

		std::cout << "Rendering reference DepthMap ..." << std::endl;
		sibr::DepthRenderer rendererDepthRef(wCamRef, hCamRef);
		glViewport(0, 0, wCamRef, hCamRef);
		rendererDepthRef.render(*camRef, meshRef);
		rendererDepthRef._depth_RT->readBack(depthMapRef);

		//showFloat(depthMapRef.resized(1024, 1024 * hCamRef / wCamRef));
		std::cout << "Rendering recon DepthMap ..." << std::endl;

		sibr::DepthRenderer rendererDepth2Align(wCam2Align, hCam2Align);
		glViewport(0, 0, wCam2Align, hCam2Align);
		rendererDepth2Align.render(*cam2Align, mesh2Align);
		rendererDepth2Align._depth_RT->readBack(depthMap2Align);

		//showFloat(depthMap2Align.resized(1024, 1024 * hCam2Align / wCam2Align));
		const int stride = std::max(16.0, sqrt(imRef.w() * imRef.h() * imgs2Align.size() / 50000.0));
		std::wcout << "   Stride:" << stride << std::endl;

		float ratioRefW = (float)depthMapRef.w() / imRefCopy.w();
		float ratioRefH = (float)depthMapRef.h() / imRefCopy.h();
		float ratio2AlignW = (float)depthMap2Align.w() / im2AlignCopy.w();
		float ratio2AlignH = (float)depthMap2Align.h() / im2AlignCopy.h();

#pragma omp parallel for
		for (int i = patchRadius; i < imRef.w(); i += stride) {
			for (int j = patchRadius; j < imRef.h(); j += stride) {

				sibr::Vector2i posRef(i, j);
				//Corresponding position is estimated from the center because the two center of the images should be aligned
				sibr::Vector2i pos2Align = im2AlignCenter + posRef - imRefCenter;


				if (imRef.isInRange(i, j)
					&& imRef(i, j) != Vector3ub(0, 0, 0)
					&& im2Align.isInRange(pos2Align.x(), pos2Align.y())
					&& im2Align.isInRange(pos2Align.x() - (shiftMax + patchRadius), pos2Align.y() - (shiftMax + patchRadius))
					&& im2Align.isInRange(pos2Align.x() + shiftMax + patchRadius + 1, pos2Align.y() + shiftMax + patchRadius + 1)) {

					// To visualize feature match in images
					imRefCopy(i, j) = sibr::Vector3ub(255, 0, 0);
					im2AlignCopy(pos2Align.x(), pos2Align.y()) = sibr::Vector3ub(255, 0, 0);

					double minDist = DBL_MAX;
					sibr::Vector2i bestShift(0, 0);

					//We find the best shift to refine our point matching
					for (int k = -shiftMax; k <= shiftMax; k++) {
						for (int l = -shiftMax; l <= shiftMax; l++) {

							sibr::Vector2i shift(k, l);
							sibr::Vector2i pos2AlignShifted = pos2Align + shift;

							double dist = distPatch(imRef, posRef, im2Align, pos2AlignShifted, patchRadius);

							if (dist < minDist || (dist == minDist && shift.norm() < bestShift.norm())) {
								bestShift = shift;
								minDist = dist;
							}
						}
					}

					sibr::Vector2i pos2AlignShifted = pos2Align + bestShift;
					sibr::Vector2i posImFullRef(posRef.x() * ratioRefW, posRef.y() * ratioRefH);
					sibr::Vector2i posImFull2Align(pos2AlignShifted.x() * ratio2AlignW, pos2AlignShifted.y() * ratio2AlignH);

					if (depthMapRef(posImFullRef.x(), posImFullRef.y()).x() != 1 &&
						depthMap2Align(posImFull2Align.x(), posImFull2Align.y()).x() != 1) {
						// To see feature match in images
						im2AlignCopy(pos2Align.x() + bestShift.x(), pos2Align.y() + bestShift.y()) = sibr::Vector3ub(0, 255, 0);

						float dRef = depthMapRef(posImFullRef.x(), posImFullRef.y()).x();
						sibr::Vector3f pos3DRef = camRef->unprojectImgSpaceInvertY(
							posImFullRef, dRef);

						float d2Align = depthMap2Align(posImFull2Align.x(), posImFull2Align.y()).x();
						sibr::Vector3f pos3D2Align = cam2Align->unprojectImgSpaceInvertY(
							posImFull2Align, d2Align);

						//Add those feature to the list
#pragma omp critical
						{
							if (pos3DRef == pos3DRef && pos3D2Align == pos3D2Align) {
								listFeatPRef.push_back(pos3DRef);
								listFeatP2Align.push_back(pos3D2Align);

								dist2CamRef.push_back((pos3DRef - camRef->position()).norm());
								dist2Cam2Align.push_back((pos3D2Align - cam2Align->position()).norm());
							}
							else {
								std::cout << "Skipping bad point" << std::endl;
							}
						}
					}

				}

			}
		}

		// To see feature match in images
		//show(imRefCopy);
		//show(im2AlignCopy);

	}

	//Now we will remove outliers that appears for several reason, one of them is obviously the part of the images where the content changed.
	//We compute the median scale factor using the distances to cameras, and we only keep matches that respect this median
	std::vector<float> scalesfromCam;
	for (int i = 0; i < dist2CamRef.size(); i++) {
		scalesfromCam.push_back(dist2CamRef[i] / dist2Cam2Align[i]);
	}
	std::sort(scalesfromCam.begin(), scalesfromCam.end());
	float medianScale = scalesfromCam[scalesfromCam.size() / 2];
	std::cout << std::endl << "Median is " << medianScale << std::endl;

	std::vector<sibr::Vector3f> listStrongFeatPRef;
	std::vector<sibr::Vector3f> listStrongFeatP2Align;

	//5% above and under the median
	for (int i = 0; i < dist2CamRef.size(); i++) {
		if (dist2CamRef[i] / dist2Cam2Align[i] > 0.95 * medianScale &&
			dist2CamRef[i] / dist2Cam2Align[i] < 1.05 * medianScale) {
			listStrongFeatPRef.push_back(listFeatPRef[i]);
			listStrongFeatP2Align.push_back(listFeatP2Align[i]);
		}
	}

	std::cout << "Cleaned matches: " << listFeatPRef.size() << " to " << listStrongFeatPRef.size() << std::endl;

	/////////
	/////////
	//Now we will estimate the transformation using irls.
	/////////
	/////////

	//Format the data
	std::vector<float> XData;
	std::vector<float> YData0;
	std::vector<float> YData1;
	std::vector<float> YData2;

	for (int i = 0; i < listStrongFeatPRef.size(); i++) {
		XData.push_back(listStrongFeatP2Align[i].x());
		XData.push_back(listStrongFeatP2Align[i].y());
		XData.push_back(listStrongFeatP2Align[i].z());

		YData0.push_back(listStrongFeatPRef[i].x());
		YData1.push_back(listStrongFeatPRef[i].y());
		YData2.push_back(listStrongFeatPRef[i].z());
	}

	///////////////
	//SOLVING WITH IRLS
	int numX = listStrongFeatP2Align.size();
	// Convert std::vector to Eigen::VectorXf
	Eigen::Map<Eigen::VectorXf> evY0(YData0.data(), numX);
	Eigen::Map<Eigen::VectorXf> evY1(YData1.data(), numX);
	Eigen::Map<Eigen::VectorXf> evY2(YData2.data(), numX);

	Eigen::Map<Eigen::Matrix<float, -1, 3, Eigen::RowMajor> > mX3(XData.data(), numX, 3);

	// Convert 3-column matrix into a 4-column one using all 1s for the last column.
	Eigen::MatrixX4f mX4(mX3.rows(), 4);
	mX4 << mX3, Eigen::ArrayXXf::Ones(mX3.rows(), 1);

#define TUNING_CONSTANT 4.685
	Eigen::Vector4f vCoeffs0;
	Eigen::Vector4f vCoeffs1;
	Eigen::Vector4f vCoeffs2;

	// Run IRLS considering each of the Y-matrix's calumns as the target Y-vector individually.
	// We get one row of the solution matrix at each step which is then put together to form the 
	// complete solution.
	//Log(EInfo, "Running IRLS on row 0");
	irls(mX4, evY0, vCoeffs0, TUNING_CONSTANT);
	//Log(EInfo, "Running IRLS on row 1");
	irls(mX4, evY1, vCoeffs1, TUNING_CONSTANT);
	//Log(EInfo, "Running IRLS on row 2");
	irls(mX4, evY2, vCoeffs2, TUNING_CONSTANT);
	//Log(EInfo, "Finished running IRLS");

	// Put all the rows together.
	Eigen::Matrix4f mFinal;
	mFinal << vCoeffs0.x(), vCoeffs0.y(), vCoeffs0.z(), vCoeffs0.w(),
		vCoeffs1.x(), vCoeffs1.y(), vCoeffs1.z(), vCoeffs1.w(),
		vCoeffs2.x(), vCoeffs2.y(), vCoeffs2.z(), vCoeffs2.w(),
		0, 0, 0, 1;

	std::cout << "Matrix is:" << std::endl;
	std::cout << vCoeffs0.x() << " " << vCoeffs0.y() << " " << vCoeffs0.z() << " " << vCoeffs0.w() << std::endl;
	std::cout << vCoeffs1.x() << " " << vCoeffs1.y() << " " << vCoeffs1.z() << " " << vCoeffs1.w() << std::endl;
	std::cout << vCoeffs2.x() << " " << vCoeffs2.y() << " " << vCoeffs2.z() << " " << vCoeffs2.w() << std::endl;
	std::cout << 0 << " " << 0 << " " << 0 << " " << 1 << std::endl;
	std::cout << medianScale << std::endl; // for xFormScene scale factor of 1

	std::ofstream myfile;
	myfile.open(outPath + "/transform.txt");
	myfile << vCoeffs0.x() << " " << vCoeffs0.y() << " " << vCoeffs0.z() << " " << vCoeffs0.w() << std::endl;
	myfile << vCoeffs1.x() << " " << vCoeffs1.y() << " " << vCoeffs1.z() << " " << vCoeffs1.w() << std::endl;
	myfile << vCoeffs2.x() << " " << vCoeffs2.y() << " " << vCoeffs2.z() << " " << vCoeffs2.w() << std::endl;
	myfile << 0 << " " << 0 << " " << 0 << " " << 1 << std::endl;
	myfile << 1 << std::endl; // for xFormScene scale factor of 1
	myfile.close();

	//create new mesh :
	Mesh alignedMesh = mesh2Align;
	std::cout << "Input vertices num : " << mesh2Align.vertices().size();
	Mesh::Vertices newVertices;
	for (sibr::Vector3f v : alignedMesh.vertices()) {
		sibr::Vector4f v4(v.x(), v.y(), v.z(), 1.0);
		newVertices.push_back((mFinal * v4).xyz());
	}
	alignedMesh.vertices(newVertices);
	std::cout << " Output vertices num : " << alignedMesh.vertices().size() << std::endl;

	if (myArgs.saveScene) {
		sibr::makeDirectory(outPath + "/meshes");
		sibr::makeDirectory(outPath + "/cameras");
		sibr::makeDirectory(outPath + "/images");

		//Save the meshes
		alignedMesh.save(outPath + "/meshes/recon.ply", true);
		alignedMesh.save(outPath + "/meshes/recon.obj", true);
	
		//Save the cameras
		//transform first
		for (auto& cam : cams2Align) {
			sibr::Vector3f pos = cam->position();
			sibr::Vector3f center = cam->position() + cam->dir();
			sibr::Vector3f up = cam->position() + cam->up();
			pos = (mFinal * pos.homogeneous()).xyz();
			center = (mFinal * center.homogeneous()).xyz();
			up = (mFinal * up.homogeneous()).xyz();
			cam->setLookAt(pos, center, (up - pos).normalized());
		}
		std::vector<InputCamera::Ptr> outCams;
		for (const auto& cam : cams2Align) {
			outCams.push_back(std::make_shared<InputCamera>(*cam));
		}

		sibr::InputCamera::saveAsBundle(outCams, outPath + "/cameras/bundle.out");

		//Save the images and metadata
		std::ofstream outputSceneMetadata;
		outputSceneMetadata.open(outPath + "/scene_metadata.txt");
		outputSceneMetadata << "Scene Metadata File\n" << std::endl;
		outputSceneMetadata << "[list_images]\n<filename> <image_width> <image_height> <near_clipping_plane> <far_clipping_plane>" << std::endl;

		int im = 0;
		for (const auto& camIm : cams2Align) {

			//std::string extensionFile = boost::filesystem::extension(camIm->name());
			std::ostringstream ssZeroPad;
			ssZeroPad << std::setw(8) << std::setfill('0') << camIm->id();
			std::string newFileName = ssZeroPad.str() + ".jpg";
			imgs2AlignOriginal[im]->save(outPath + "/images/" + newFileName);
			outputSceneMetadata << newFileName << " " << camIm->w() << " " << camIm->h() << " " << camIm->znear() << " " << camIm->zfar() << std::endl;
			im++;
		}
		outputSceneMetadata << "\n// Always specify active/exclude images after list images\n\n[exclude_images]\n<image1_idx> <image2_idx> ... <image3_idx>" << std::endl;
		outputSceneMetadata << "\n\n\n[other parameters]" << std::endl;
		outputSceneMetadata.close();
	}
	else { // just save meshes (transform.txt saved above)
		std::string textureFileName;
		// could preserve texture name, but is probably cleaner to have standard name
		textureFileName = "textured_u1_v1.png";

		alignedMesh.save(outPath + "/mesh.ply", true, textureFileName);
		alignedMesh.save(outPath + "/mesh.obj", true);
		// save the mtl file
		std::string mtlFileName = outPath + "/mesh.mtl";
		std::ofstream mtlFile;
		mtlFile.open(mtlFileName);
		mtlFile << "# File produced by SIBR\n\nnewmtl $Material_0\nKa 1 1 1\nKd 1 1 1\nd 1\nNs 0\nillum 1\nmap_Kd " << textureFileName;
		mtlFile.close();
	}

	return EXIT_SUCCESS;
}



template <typename T> std::vector<size_t> sort_indexes(const std::vector<T>& v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

	return idx;
}



void computeRT(std::vector<sibr::Vector3f> A, std::vector<sibr::Vector3f>B, Matrix3f S, Matrix3f& R, sibr::Vector3f& T) {

	//finding R,T such that B = RA + T;
	const int numPoint = B.size();
	//Scaling the points to align
	std::vector<sibr::Vector3f> AScaled;
	for (int i = 0; i < numPoint; i++) {
		AScaled.push_back(S * A[i]);
	}

	//Computing the centroids
	sibr::Vector3f centroidB(0, 0, 0);
	sibr::Vector3f centroidA(0, 0, 0);
	for (int i = 0; i < numPoint; i++) {
		centroidB += B[i];
		centroidA += AScaled[i];
	}
	centroidB /= numPoint;
	centroidA /= numPoint;


	//Now we estimate the rotation :
	Matrix3f H;
	H << 0, 0, 0,
		0, 0, 0,
		0, 0, 0;

	for (int i = 0; i < numPoint; i++) {
		H += (AScaled[i] - centroidA) * (B[i] - centroidB).transpose();
	}

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	R = svd.matrixV() * svd.matrixU().transpose();
	if (R.determinant() < 0) {
		//R.col(2) *= -1;
		//std::cout << "Warning : determinant of rotation matrix is negative, multiplying last column by -1" << std::endl;
	}

	//Translation estimation :
	T = -R * centroidA + centroidB;

}

float computeS(std::vector<sibr::Vector3f> A, std::vector<sibr::Vector3f>B, float& minScale, float& maxScale) {
	//findin S such that A*S has the same scale as B
	sibr::Vector3f meanPosA(0, 0, 0);
	sibr::Vector3f meanPosB(0, 0, 0);

	for (int i = 0; i < A.size(); i++) {
		meanPosA += A[i];
		meanPosB += B[i];
	}

	meanPosA /= A.size();
	meanPosB /= A.size();

	float scale = 0;
	for (int i = 0; i < A.size(); i++) {

		float scale_i = (B[i] - meanPosB).norm() / (A[i] - meanPosA).norm();
		scale += scale_i;

		if (scale_i > maxScale)
			maxScale = scale_i;
		if (scale_i < minScale)
			minScale = scale_i;
	}

	scale /= A.size();

	return scale;
}


