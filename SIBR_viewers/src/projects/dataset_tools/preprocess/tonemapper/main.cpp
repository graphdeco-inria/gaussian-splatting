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
#include "core/graphics/Image.hpp"
#include "core/graphics/Mesh.hpp"
#include "core/imgproc/MeshTexturing.hpp"
#include "core/scene/BasicIBRScene.hpp"

using namespace sibr;


struct TonemapperAppArgs : virtual AppArgs {
	RequiredArg<std::string> path = { "path", "path to the EXR images directory" };
	Arg<std::string> output = { "output", "", "output directory path" };
	Arg<std::string> outputExtension = { "ext", "png", "output files extension" };
	Arg<float> exposure = { "exposure", 1.0f, "exposure value" };
	Arg<float> gamma = { "gamma", 2.2f, "gamma value" };
};

void tonemap(const sibr::ImageRGB32F& hdrImg, sibr::ImageRGB& ldrImg, float exposure, float gamma) {
	const cv::Mat & tonemaped = hdrImg.toOpenCV();
	const cv::Mat exposed = -exposure * tonemaped;
	cv::Mat tonemaped2;
	cv::exp(exposed, tonemaped2);
	tonemaped2 = cv::Scalar(1.0f, 1.0f, 1.0f) - tonemaped2;
	if (gamma > 0.0f) {
		cv::pow(tonemaped2, 1.0f / gamma, tonemaped2);
	}
	cv::Mat tonemapedRGB;
	tonemaped2.convertTo(tonemapedRGB, CV_8UC3, 255.0f);
	ldrImg.fromOpenCV(tonemapedRGB);
}

int main(int ac, char** av) {

	// Parse Command-line Args
	sibr::CommandLineArgs::parseMainArgs(ac, av);

	TonemapperAppArgs args;

	// Add the extension dot if needed.
	std::string extension = args.outputExtension;
	if (!extension.empty() && extension[0] != '.') {
		extension = "." + extension;
	}
	
	// Input/output paths.
	const std::string inputPath = args.path;
	std::string outputPath = args.output;
	// If we output in the same dir, we want to avoid collisions.
	if (outputPath.empty()) {
		outputPath = inputPath;
		extension = "_ldr" + extension;
	} else {
		sibr::makeDirectory(outputPath);
	}

	const auto files = sibr::listFiles(inputPath, false, false, { "exr" });

	for (const auto& file : files) {
		const std::string src = inputPath + "/" + file;
		const std::string dst = outputPath + "/" + sibr::removeExtension(file) + extension;

		sibr::ImageRGB32F hdrImg;
		sibr::ImageRGB ldrImg;
		hdrImg.load(src);
		tonemap(hdrImg, ldrImg, args.exposure, args.gamma);
		ldrImg.save(dst);
	}

	return 0;
}

