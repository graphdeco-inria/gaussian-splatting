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


#include "VideoUtils.hpp"

#include <core/graphics/Utils.hpp>
#include <algorithm>


#include <opencv2/ximgproc/edge_filter.hpp>
#include <opencv2/optflow.hpp>
namespace sibr {

	std::vector<cv::Mat> cvSplitChannels(cv::Mat mat) {
		std::vector<cv::Mat> out;
		cv::split(mat, out);
		return out;
	}

	Volume3u loadVideoVolume(const std::string & filepath) {
		Video video(filepath);
		if (video.exists()) {
			return loadVideoVolume(video);
		} else {
			SIBR_WRG << filepath << " does not exists" << std::endl;
			return Volume3u();
		}
	}

	Volume3u loadVideoVolume(sibr::Video & video)
	{
		int currentFrame = video.getCurrentFrameNumber();
		video.setCurrentFrame(0);
		Volume3u volume(video.getNumFrames(), video.getResolution()[0], video.getResolution()[1]);
		for (int t = 0; t < video.getNumFrames(); ++t) {
			cv::Mat mat = volume.frame(t);
			video.getCVvideo() >> mat;
		}
		video.setCurrentFrame(currentFrame);
		return volume;
	}

	SIBR_VIDEO_EXPORT uint optimal_num_levels(uint length)
	{
		uint num_levels = 1;
		while (length != 1) {
			length = (length + 1) / 2;
			++num_levels;
		}
		return num_levels;
	}

	SIBR_VIDEO_EXPORT std::vector<sibr::Volume3u> gaussianPyramid(const sibr::Volume3u & vid, uint num_levels)
	{
		if (num_levels == 0) {
			num_levels = optimal_num_levels(vid.l);
		}

		std::vector<sibr::Volume3u> out(1, vid);
		for (int i = 1; i < (int)num_levels; ++i) {
			out.push_back(out.back().pyrDown());
		}
		return out;
	}

	SIBR_VIDEO_EXPORT std::vector<sibr::Volume3u> gaussianPyramidTemporal(const sibr::Volume3u & vid, uint num_levels)
	{
		if (num_levels == 0) {
			num_levels = optimal_num_levels(vid.l);
		}

		std::vector<sibr::Volume3u> out(1, vid);
		for (int i = 1; i < (int)num_levels; ++i) {
			out.push_back(out.back().pyrDownTemporal());
		}
		return out;
	}

	std::vector<sibr::Volume3u> laplacianPyramid(const sibr::Volume3u & vid, uint num_levels)
	{
		if (num_levels == 0) {
			num_levels = optimal_num_levels(vid.l);
		}

		std::vector<sibr::Volume3u> out;
		
		sibr::Volume3f current_v = vid.convertTo<float>(), down, up;
		for (int i = 0; i < (int)num_levels - 1; ++i) {
			down = current_v.pyrDown();
			up = down.pyrUp(current_v.l, current_v.w, current_v.h);
			//current_v.play(30, { 1200,800 });
			//up.play(30, { 1200,800 });
			current_v.substract(up);
			current_v.shift(128);
			out.push_back(current_v.convertTo<uchar>());
			std::swap(current_v, down);
		}
		out.push_back(current_v.convertTo<uchar>());
		return out;
	}

	std::vector<sibr::Volume3u> laplacianPyramidTemporalDouble(const sibr::Volume3u & vid, uint num_levels)
	{
		if (num_levels == 0) {
			num_levels = 1;
			int length = vid.l;
			while (length != 1) {
				length = (length + 1) / 2;
				length = (length + 1) / 2;
				++num_levels;
			}
		}

		std::cout << " num lvls : " << num_levels << std::endl;

		std::vector<sibr::Volume3u> out;
		sibr::Volume3f current_v = vid.convertTo<float>(), down, up;
		for (int i = 0; i < (int)num_levels - 1; ++i) {
			down = current_v.pyrDownTemporal().pyrDownTemporal();
			up = down.pyrUpTemporal((current_v.l + 1) / 2).pyrUpTemporal(current_v.l);
			current_v.substract(up);
			current_v.shift(128);
			out.push_back(current_v.convertTo<uchar>());
			std::cout << i << " : " << current_v.l << std::endl;
			std::swap(current_v, down);
		}
		out.push_back(current_v.convertTo<uchar>());
		return out;
	}

	SIBR_VIDEO_EXPORT sibr::Volume3u collapseLaplacianPyramid(const std::vector<sibr::Volume3u>& pyr, double shift)
	{
		sibr::Volume3f v = pyr.back().convertTo<float>();
		for (int i = (int)pyr.size() - 2; i >= 0; --i) {
			v = v.pyrUp(pyr[i].l, pyr[i].w, pyr[i].h);
			v.add(pyr[i]);
			if (shift != 0) {
				v.shift(shift);
			}
		}
		return v.convertTo<uchar>();
	}

	SIBR_VIDEO_EXPORT sibr::Volume3u laplacianBlending(const sibr::Volume3u & vA, const sibr::Volume3u & vB, std::vector<sibr::Volume1u>& pyrM)
	{
		auto pyrA = laplacianPyramid(vA);
		auto pyrB = laplacianPyramid(vB);

		for (int i = (int)pyrA.size() - 1; i >= 0; --i) {
			pyrA[i] = pyrA[i].applyMask(pyrM[i]);
			pyrM[i].toggle();
			pyrB[i] = pyrB[i].applyMask(pyrM[i]);
			pyrA[i].add(pyrB[i]);
		}

		return collapseLaplacianPyramid(pyrA, -128);
	}


	int VideoUtils::codec_ffdshow = cv::VideoWriter::fourcc('F', 'F', 'D', 'S');
	int VideoUtils::codec_OpenH264 = cv::VideoWriter::fourcc('H', '2', '6', '4');
	int VideoUtils::codec_OpenH264_fallback = 0x31637661;

	// from https://stackoverflow.com/questions/7693561/opencv-displaying-a-2-channel-image-optical-flow
	cv::Mat VideoUtils::getFlowViz(const cv::Mat & flow) {

		cv::Mat xy[2]; //X,Y
		cv::split(flow, xy);

		//calculate angle and magnitude
		cv::Mat magnitude, angle;
		cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

		//translate magnitude to range [0;1]
		double mag_max;
		cv::minMaxLoc(magnitude, 0, &mag_max);
		magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

		//build hsv image
		cv::Mat _hsv[3], hsv;
		_hsv[0] = angle;
		_hsv[1] = magnitude;
		_hsv[2] = cv::Mat::ones(angle.size(), CV_32F);
		cv::merge(_hsv, 3, hsv);

		//convert to BGR and show
		cv::Mat bgr;//CV_32FC3 matrix
		cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

		return bgr;
	}

	cv::Mat VideoUtils::cropFromSize(const cv::Mat & mat, const sibr::Vector2i & size)
	{
		sibr::Vector2i currentSize(mat.cols, mat.rows);
		sibr::Vector2i targetSize = size.cwiseMin(currentSize);
		sibr::Vector2i topLeft = ((currentSize - targetSize).cast<float>() / 2.0).cast<int>();
		sibr::Vector2i bottomRight = ((currentSize + targetSize).cast<float>() / 2.0).cast<int>();
		cv::Rect roi(cv::Point(topLeft[0], topLeft[1]), cv::Point(bottomRight[0], bottomRight[1]));

		//std::cout << "--------------" << std::endl;
		//std::cout << currentSize << std::endl;
		//std::cout << targetSize << std::endl;
		//std::cout << roi << std::endl;

		return mat(roi).clone();
	}

	void VideoUtils::getMeanVariance(cv::VideoCapture & cap, cv::Mat & outMean, cv::Mat & outVariance, const sibr::Vector2i & finalSize)
	{
		cap.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, 0);
		bool first = true;
		cv::Mat mean, meanSq, out;
		float sum = 0;
		int f_id = 0, current_seg = -1;
		bool doResize = (finalSize[0] != cap.get(cv::CAP_PROP_FRAME_WIDTH) || finalSize[1] != cap.get(cv::CAP_PROP_FRAME_HEIGHT));
		while (true) {
			std::cout << "." << std::flush;
			cv::Mat frame, frame_float;
			cap >> frame;
			++f_id;
			if (frame.empty()) {
				break;
			}

			if (doResize) {
				cv::resize(frame, frame, cv::Size(finalSize[0], finalSize[1]));
			}

			frame.convertTo(frame_float, CV_32FC3);
			if (first) {
				mean = frame_float;
				meanSq = frame_float.mul(frame_float);
				first = false;
			} else {
				mean += frame_float;
				meanSq += frame_float.mul(frame_float);
			}
			sum += 1;
		}
		if (first) {
			return;
		}

		mean /= sum;
		cv::Mat var = cv::min(255.0f*255.0f, cv::max(0.0f, meanSq / sum - mean.mul(mean)));
		cv::sqrt(var, var);
		var *= 5.0;
		mean.convertTo(outMean, CV_8UC3);
		var.convertTo(outVariance, CV_8UC3);

	}

	void VideoUtils::getMeanVariance2(cv::VideoCapture & cap, cv::Mat & outMean, cv::Mat & outVariance, const sibr::Vector2i & finalSize, float starting_point_s)
	{
		int starting_frame = (int)(starting_point_s*cap.get(cv::CAP_PROP_FPS));

		cap.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, starting_frame);
		bool first = true;
		cv::Mat mean, meanSq, out;
		float sum = 0;
		int f_id = 0, current_seg = -1;
		bool doResize = (finalSize[0] != cap.get(cv::CAP_PROP_FRAME_WIDTH) || finalSize[1] != cap.get(cv::CAP_PROP_FRAME_HEIGHT));
		while (true) {
			std::cout << "." << std::flush;
			cv::Mat frame, frame_float;
			cap >> frame;
			++f_id;
			if (frame.empty()) {
				break;
			}

			if (doResize) {
				cv::resize(frame, frame, cv::Size(finalSize[0], finalSize[1]));
			}

			cv::GaussianBlur(frame, frame, cv::Size(3, 3), 0);

			frame.convertTo(frame_float, CV_32FC3);
			if (first) {
				mean = frame_float;
				meanSq = frame_float.mul(frame_float);
				first = false;
			} else {
				mean += frame_float;
				meanSq += frame_float.mul(frame_float);
			}
			sum += 1;
		}
		if (first) {
			return;
		}

		mean /= sum;
		cv::Mat var = cv::min(255.0f*255.0f, cv::max(0.0f, meanSq / sum - mean.mul(mean)));
		cv::sqrt(var, var);
		var *= 5.0;
		mean.convertTo(outMean, CV_8UC3);
		var.convertTo(outVariance, CV_8UC3);
	}

	cv::Mat VideoUtils::getMedian(sibr::Video & vid, float time_skiped_begin, float time_skiped_end) {

		cv::Mat volume = vid.getVolume(time_skiped_begin, time_skiped_end);

		//std::cout << "tranpose ";
		//volume = volume.t();
		//std::cout << t.deltaTimeFromLastTic<>() << std::endl;

		//cv::Mat volumeSorted;

		//std::cout << "sort ";
		//cv::sort(volume, volume, CV_SORT_EVERY_COLUMN);
		//std::cout << t.deltaTimeFromLastTic<>() << std::endl;

		cv::Mat median(vid.getResolutionCV(), CV_8UC3);

		const int L = volume.rows;

#pragma omp parallel for
		for (int i = 0; i < median.rows; ++i) {

			for (int j = 0; j < median.cols; ++j) {
				cv::Vec3b medianColor;
				for (int c = 0; c < 3; ++c) {
					std::vector<uchar> values(L);
					for (int t = 0; t < L; ++t) {
						values[t] = volume.at<uchar>(t, 3 * (i*median.cols + j) + c);
					}
					//std::sort(values.begin(), values.end());
					//medianColor[c] = values[values.size() / 2];
					std::nth_element(values.begin(), values.begin() + L / 2, values.end());
					medianColor[c] = values[L / 2];
				}
				median.at<cv::Vec3b>(i, j) = medianColor;
			}


		}

		return median;
	}

	cv::Mat3b VideoUtils::getMedian(const std::string & path, float time_percentage_crop)
	{

		sibr::Video vid(path);
		Volume3u vol = loadVideoVolume(vid);
		cv::Mat3b median(vid.getResolutionCV(), CV_8UC3);

		int crop = (int)(vol.l*std::min(time_percentage_crop, 0.4f));
		int start = crop, end = vol.l - crop;

	
#pragma omp parallel for
		for (int i = 0; i < median.rows; ++i) {
			cv::Mat line = vol.video_line(i);
			std::vector<uchar> values;
			for (int j = 0; j < median.cols; ++j) {
				for (int c = 0; c < 3; ++c) {
					line.col(3 * j + c).rowRange(start, end).copyTo(values);
					std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
					median(i, j)[c] = values[values.size() / 2];
				}		
			}
		}

		return median;
	}

	cv::Mat VideoUtils::getBackgroundImage(sibr::Video & vid, int numBins, float time_skip_begin, float time_skip_end) {
		cv::Mat volume = vid.getVolume(time_skip_begin, time_skip_end);
		volume = volume.t();
		return getBackgroundImage(volume, vid.getResolution()[0], vid.getResolution()[1], numBins);
	}

	cv::Mat VideoUtils::getBackgroundImage(const cv::Mat volume, int w, int h, int numBins)
	{
		cv::Mat bg = cv::Mat(h, w, CV_8UC3);
		const int L = volume.cols;

#pragma omp parallel for
		for (int i = 0; i < bg.rows; ++i) {
			for (int j = 0; j < bg.cols; ++j) {

				std::vector<sibr::Vector3ub> values(L);

				for (int c = 0; c < 3; ++c) {
					for (int t = 0; t < L; ++t) {
						values[t][c] = volume.at<uchar>(3 * (i*bg.cols + j) + c, t);
					}
				}

				TimeHistogram histo = TimeHistogram(0, 255, numBins);
				histo.addValues(values);

				auto mode = histo.getBinMiddle(histo.getHMode());
				for (int c = 0; c < 3; ++c) {
					bg.at<cv::Vec3b>(i, j)[c] = mode[c];
				}
			}
		}
		return bg;
	}

	void VideoUtils::getBackGroundVideo(sibr::Video & vid, PyramidLayer & out_mask, PyramidLayer & out_video, cv::Mat & out_img,
		const sibr::ImageRGB & meanImg, int threshold, int numBins, float time_skip_begin, float time_skip_end)
	{
		cv::Mat volume = vid.getVolume(time_skip_begin, time_skip_end).t();

		const int w = vid.getResolution()[0], h = vid.getResolution()[1], L = volume.cols;
		out_mask.w = w;
		out_mask.l = L;
		out_mask.h = h;

		out_video = out_mask;

		out_mask.volume = cv::Mat(L, 3 * w*h, CV_8UC1);
		out_video.volume = cv::Mat(L, 3 * w*h, CV_8UC1);
		out_img = cv::Mat(h, w, CV_8UC3);

		const bool useMeanImg = !(meanImg.size()[0] == 0);

		std::cout << w << " " << h << " " << L << " use mean img " << useMeanImg << std::endl;
#pragma omp parallel for
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {

				std::vector<sibr::Vector3ub> values(L);
				for (int c = 0; c < 3; ++c) {
					for (int t = 0; t < L; ++t) {
						values[t][c] = volume.at<uchar>(3 * (i*w + j) + c, t);
					}
				}

				TimeHistogram histo = TimeHistogram(0, 255, numBins);
				histo.addValues(values);
				//histo.computeSortedBins();

				auto mode = histo.getHMode();
				auto mode_color = histo.getBinMiddle(histo.getHMode());

				out_img.at<cv::Vec3b>(i, j) = sibr::toOpenCV<uchar, uchar, 3>(mode_color);

				sibr::Vector3ub stdDev;

				if (useMeanImg) {
					const int radius = 4;
					const int diam = 2 * radius + 1;
					const int num = diam * diam;
					sibr::Vector3f sumColor(0, 0, 0), sumColorSq(0, 0, 0);
					for (int di = -radius; di <= radius; ++di) {
						int ii = sibr::clamp(i + di, 0, h - 1);
						for (int dj = -radius; dj <= radius; ++dj) {
							int jj = sibr::clamp(j + dj, 0, w - 1);
							sumColor += meanImg(jj, ii).cast<float>();
							sumColorSq += meanImg(jj, ii).cast<float>().cwiseProduct(meanImg(jj, ii).cast<float>());
						}
					}
					sumColor /= (float)(num);
					sumColorSq = sumColorSq / (float)num - sumColor.cwiseProduct(sumColor);
					stdDev = sumColorSq.cwiseSqrt().cast<uchar>();
					threshold = 15 * stdDev.norm();
				}

				for (int t = 0; t < L; ++t) {
					const auto & color = values[t];
					auto bin = histo.whatBin(color);

					//float cdf = histo.sorted_bins[bin];
					//float outlier_prop = 1.0f - cdf;
					//auto viz_color = sibr::jetColor<uchar>(outlier_prop);

					for (int c = 0; c < 3; ++c) {
						//out_mask.volume.at<uchar>(t, 3 * (i*w + j) + c) = (cdf < 0.75f ? 0 : 255);
						//out_video.volume.at<uchar>(t, 3 * (i*w + j) + c) = viz_color[c];

						if ((color.cast<int>() - mode_color.cast<int>()).norm() < threshold) {
							out_mask.volume.at<uchar>(t, 3 * (i*w + j) + c) = 0;
						} else {
							out_mask.volume.at<uchar>(t, 3 * (i*w + j) + c) = 255;
						}
					}
				}

			}
		}

		volume = volume.t();
		out_video.volume = volume.mul((1.0 / 255)*out_mask.volume);
	}

	sibr::Volume1u VideoUtils::getBackgroundVolume(const sibr::Volume3u & volume, int threshold, int numBins)
	{
		const int L = volume.l;
		sibr::Volume1u out_mask = sibr::Volume1u(L, volume.w, volume.h, 0);

#pragma omp parallel for
		for (int i = 0; i < volume.h; ++i) {
			for (int j = 0; j < volume.w; ++j) {
				std::vector<sibr::Vector3ub> values(L);
				for (int c = 0; c < 3; ++c) {
					for (int t = 0; t < L; ++t) {
						values[t][c] = volume.valueAt(t, i, j, c);
					}
				}

				TimeHistogram histo = TimeHistogram(0, 255, numBins);
				histo.addValues(values);

				auto mode_color = histo.getBinMiddle(histo.getHMode());

				for (int t = 0; t < L; ++t) {
					const auto & color = values[t];

					if ((color.cast<int>() - mode_color.cast<int>()).norm() > threshold) {
						out_mask.pixelAt(t, i, j) = 255;
					}
				}

			}
		}

		return out_mask;
	}

	sibr::Volume1f VideoUtils::getBackgroundVolumeF(const sibr::Volume3u & volume, int numBins)
	{
		const int L = volume.l;
		sibr::Volume1f out_mask = sibr::Volume1f(L, volume.w, volume.h);

#pragma omp parallel for
		for (int i = 0; i < volume.h; ++i) {
			for (int j = 0; j < volume.w; ++j) {
				std::vector<sibr::Vector3ub> values(L);
				for (int c = 0; c < 3; ++c) {
					for (int t = 0; t < L; ++t) {
						values[t][c] = volume.valueAt(t, i, j, c);
					}
				}

				TimeHistogram histo = TimeHistogram(0, 255, numBins);
				histo.addValues(values);

				auto mode_color = histo.getBinMiddle(histo.getHMode());

				for (int t = 0; t < L; ++t) {
					const auto & color = values[t];
					out_mask.pixelAt(t, i, j) = (float)(color.cast<int>() - mode_color.cast<int>()).norm();
				}

			}
		}

		return out_mask;
	}

	void VideoUtils::computeSaveSimpleFlow(sibr::Video & vid, bool show)
	{
		int layers = 5;
		int block_size = 3;
		int max_flow = 5;

		sibr::Volume3u vol = sibr::loadVideoVolume(vid);
		Path path = vid.getFilepath();
		std::string folder = path.parent_path().string() + "/flow/";
		sibr::makeDirectory(folder);

		std::string filepath = folder + "/" + path.stem().string() + "_sflow_" + std::to_string(layers) + "_" +
			std::to_string(block_size) + + "_" + std::to_string(max_flow) + ".mp4";

		sibr::FFVideoEncoder encoder(filepath, 30, { 2 * vol.w,2 * vol.h });
		for (int t = 0; t < vol.l - 1; ++t) {
			cv::Mat flow;
			cv::optflow::calcOpticalFlowSF(vol.frame(t), vol.frame(t + 1), flow, layers, block_size, max_flow);
			cv::Mat viz = vol.frame(t).clone();
			int r = 10;
			for (int i = 0; i < vol.h; i += r) {
				for (int j = 0; j < vol.w; j += r) {
					auto f = flow.at<cv::Vec2f>(i, j);
					if (isfinite(f[0]) && isfinite(f[1])) {
						if (cv::norm(f) > 0.5) {
							cv::line(viz, cv::Point(j, i), cv::Point(int(j + f[1]), int(i + f[0])), { 255,0,255 }, 2);
						}
					} else {
						cv::circle(viz, cv::Point(j, i), 3, { 0,0,0 }, 2);
					}


				}
			}
			cv::resize(viz, viz, cv::Size(2 * viz.cols, 2 * viz.rows), 0, 0, cv::INTER_NEAREST);

			if (show) {
				cv::imshow("flow", viz);
				if (cv::waitKey() == 27) {
					break;
				}
			} else {
				encoder << viz;
			}
			std::cout << "." << std::flush;
		}
		std::cout << "done " << std::endl;
	}

	void VideoUtils::computeSaveVideoMaskF(Video & vid, int threshold, bool viz)
	{
		sibr::Volume3u volume = sibr::loadVideoVolume(vid);
		sibr::Volume1f mask = sibr::VideoUtils::getBackgroundVolumeF(volume, 150);

		sibr::Volume1f bilateral_mask(volume.l, volume.w, volume.h);
		sibr::Volume1f bilateral_mask_median(volume.l, volume.w, volume.h);
		sibr::Volume1u median_bilateral_mask_binary(volume.l, volume.w, volume.h);

		const int radius_bila = 21;
		const double eps = 10;

#pragma omp parallel for
		for (int t = 0; t < volume.l; ++t) {
			cv::ximgproc::guidedFilter(volume.frame(t), mask.frame(t), bilateral_mask.frame(t), radius_bila, eps);
			//cv::medianBlur(mask.frame(t), median_mask.frame(t), 7);
			cv::medianBlur(bilateral_mask.frame(t), bilateral_mask_median.frame(t), 5);
			median_bilateral_mask_binary.frame(t) = bilateral_mask_median.frame(t) > threshold;
		}

		sibr::Volume3u video_masked_bilateral_bin = volume.applyMaskBinary(median_bilateral_mask_binary);

		if (viz) {
			bilateral_mask.play();
			bilateral_mask_median.play();
			median_bilateral_mask_binary.play();
			video_masked_bilateral_bin.play();
		}

		Path filepath = vid.getFilepath();
		const std::string folder = filepath.parent_path().string() + "/masks/bilateral/";
		sibr::makeDirectory(folder);

		const std::string basename = folder + filepath.stem().string() + "_bila_" + std::to_string(radius_bila) + "_" + std::to_string((int)(10 * eps));
		const std::string extension = ".mp4";

		bilateral_mask.saveToVideoFile(basename + "_raw" + extension);
		bilateral_mask_median.saveToVideoFile(basename + "_median" + extension);
		median_bilateral_mask_binary.saveToVideoFile(basename + "_median_binary" + extension);
		video_masked_bilateral_bin.saveToVideoFile(basename + "_video" + extension);
	}

	void VideoUtils::computeSaveVideoMaskBlur(Video & vid, int time_window)
	{
		const Path & filepath = vid.getFilepath();
		const std::string in_filename = filepath.parent_path().string() + "/masks/bilateral/" + filepath.stem().string() + "_bila_21_100_median_binary.mp4";
		const std::string out_folder = filepath.parent_path().string() + "/masks/bilateral_tblur/";
		sibr::makeDirectory(out_folder);
		const std::string out_filename = out_folder + "/" + filepath.stem().string() + "_mask_tblur.mp4";

		sibr::Volume3u volume = sibr::loadVideoVolume(in_filename);
		std::cout << "volume.mat.isContinuous() : " << volume.mat.isContinuous() << std::endl;
		sibr::Volume3u out = sibr::Volume3u(volume.l, volume.w, volume.h, 0);

		int time_win = 10;

		//#pragma omp parallel for
		for (int i = 0; i < out.h; ++i) {
			for (int j = 0; j < out.w; ++j) {
				for (int t = 0; t < out.l; ++t) {
					for (int u = std::max(0, t - time_win); u < std::min(out.l - 1, t + time_win); ++u) {
						if (volume.valueAt(u, i, j, 0) > 128) {
							out.pixelAt(t, i, j) = cv::Vec3b(255, 255, 255);
							break;
						}
					}
				}
			}
		}
		out.saveToVideoFile(out_filename);
	}

	cv::Mat VideoUtils::getTemporalSpatialRatio(sibr::Video & vid, PyramidLayer & out_ratio, const sibr::ImageRGB & spatial_ratio, int numBins, float time_skip_begin, float time_skip_end)
	{
		cv::Mat volume = vid.getVolume(time_skip_begin, time_skip_end).t();

		const int w = vid.getResolution()[0], h = vid.getResolution()[1], L = volume.cols;
		out_ratio.w = w;
		out_ratio.l = L;
		out_ratio.h = h;

		out_ratio.volume = cv::Mat(L, 3 * w*h, CV_8UC1);

#pragma omp parallel for
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {

				std::vector<sibr::Vector3ub> values(L);
				for (int c = 0; c < 3; ++c) {
					for (int t = 0; t < L; ++t) {
						values[t][c] = volume.at<uchar>(3 * (i*w + j) + c, t);
					}
				}

				TimeHistogram histo = TimeHistogram(0, 255, numBins);
				histo.addValues(values);
				//histo.computeSortedBins();

				auto mode = histo.getHMode();
				auto mode_color = histo.getBinMiddle(histo.getHMode());

				for (int t = 0; t < L; ++t) {
					const auto & color = values[t];
					auto bin = histo.whatBin(color);

					sibr::Vector3f norm_temporal = (color.cast<int>() - mode_color.cast<int>()).cwiseAbs().cast<float>();
					sibr::Vector3f norm_spatial = spatial_ratio(j, i).cwiseAbs().cast<float>().array() + 10;
					sibr::Vector3f ratios = norm_temporal.cwiseQuotient(norm_spatial);


					for (int c = 0; c < 3; ++c) {
						out_ratio.volume.at<uchar>(t, 3 * (i*w + j) + c) = sibr::clamp<uchar>((uchar)(128 * ratios[c]), 0, 255);
						//if (ratios.maxCoeff() > 0.5) {
						//	out_ratio.volume.at<uchar>(t, 3 * (i*w + j) + c) = 255;
						//} else {
						//	out_ratio.volume.at<uchar>(t, 3 * (i*w + j) + c) = 0;
						//}
						//out_ratio.volume.at<uchar>(t, 3 * (i*w + j) + c) = sibr::clamp<uchar>((uchar)(64*ratios[c]),0,255);
					}
				}

			}
		}

		return volume.t();
	}

	cv::Mat VideoUtils::getLaplacian(cv::Mat mat, int size, bool smooth, bool absolute)
	{
		cv::Mat grey, laplacian, abs;
		if (smooth) {
			cv::GaussianBlur(mat, mat, cv::Size(size, size), 0, 0, cv::BORDER_DEFAULT);
		}
		grey = getGrey(mat);
		cv::Laplacian(grey, laplacian, CV_16S, size);
		if (absolute) {
			cv::convertScaleAbs(laplacian, abs);
			return abs;
		}
		return laplacian;
	}

	cv::Mat VideoUtils::getCanny(cv::Mat mat)
	{
		cv::Mat grey, canny;
		grey = getGrey(mat);
		cv::Canny(grey, canny, 50, 150);
		return canny;
	}

	int VideoUtils::rotationAngleFromMetadata(const std::string & videoPath)
	{
		namespace bfs = boost::filesystem;

		Path vidPath = bfs::canonical(videoPath);
		std::string parentAbs = bfs::canonical(vidPath.parent_path()).string();
		std::string tmpFilePath = parentAbs + "/" + vidPath.stem().string() + "_tmp.txt";

		std::string cmd = "ffprobe -i \"" + vidPath.string() + "\" > \"" + tmpFilePath + "\" 2>&1";
		//std::cout << cmd << std::endl;

		int cmd_status = std::system(cmd.c_str());
		if (cmd_status != EXIT_SUCCESS) {
			SIBR_WRG << "getMetaData failed to call : " << cmd << std::endl;
		}

		std::ifstream file(tmpFilePath);
		if (!file.is_open()) {
			SIBR_WRG << "getMetaData failed to open " << tmpFilePath << std::endl;
		}
		std::string line, tmp;
		std::stringstream linestream;

		int angle = 0;
		while (safeGetline(file, line)) {
			if (line.find("rotate") != std::string::npos) {
				linestream << line;
				linestream >> tmp >> tmp >> angle;
				break;
			}
		}
		file.close();

		if (!boost::filesystem::remove(tmpFilePath)) {
			SIBR_WRG << "getMetaData failed to remove " << tmpFilePath << std::endl;
		}

		return angle;
	}

	void VideoUtils::ECCtransform(cv::Mat matA, cv::Mat matB, cv::Mat & correctedB, cv::Mat & diff, int cvMotion)
	{
		cv::Mat greyA, greyB, warpBA;
		cv::cvtColor(matA, greyA, cv::COLOR_BGR2GRAY);
		cv::cvtColor(matB, greyB, cv::COLOR_BGR2GRAY);
		try {
			cv::findTransformECC(greyA, greyB, warpBA, cvMotion);
		}
		catch (const std::exception & e) { std::cout << e.what();  return; }

		if (cvMotion == cv::MOTION_HOMOGRAPHY) {
			cv::warpPerspective(matB, correctedB, warpBA, matB.size());
		} else if (cvMotion == cv::MOTION_AFFINE) {
			cv::warpAffine(matB, correctedB, warpBA, matB.size());
		}

		cv::absdiff(matA, correctedB, diff);
	}

	void VideoUtils::smallAlignmentVideo(sibr::Video & vid, const std::string & outputVidPath, bool viz)
	{
		struct Match {
			cv::Point2f in, out;
			float error;
		};

		//cv::VideoWriter out(outputVidPath, codec_OpenH264, vid.getFrameRate(), cv::Size(vid.getResolution()[0], vid.getResolution()[1]));
		sibr::FFVideoEncoder out(outputVidPath, vid.getFrameRate(), vid.getResolution());

		if (!out.isFine()) {
			SIBR_WRG << " cant write video " << outputVidPath << std::endl;
		}
		vid.setCurrentFrame(0);
		cv::Mat initFrame = vid.next();
		cv::Mat initGray = VideoUtils::getGrey(initFrame);

		std::vector<cv::Point2f> features, nextFeatures;
		std::vector<uchar> status;
		std::vector<float> errors;

		cv::Mat totalHomography = cv::Mat::eye(3, 3, CV_32FC1);

		const double magic_expon = 1.6;
		const double ratio = 0.5;
		const double ransac_repro_error = 3.0;
		const double features_to_track_quality = 0.1;
		const double features_min_dist = 0.1; //10
		int nPixels = vid.getResolution().prod();

		int numFeatures = (int)pow(nPixels, 1.0 / magic_expon);
		std::cout << " num features " << numFeatures << std::endl;

		cv::goodFeaturesToTrack(initGray, features, numFeatures, features_to_track_quality, 10);

		cv::Mat nextFrame, gray;

		for (;;) {
			nextFrame = vid.next();
			if (nextFrame.empty()) {
				break;
			}
			gray = VideoUtils::getGrey(nextFrame);

			cv::calcOpticalFlowPyrLK(initGray, gray, features, nextFeatures, status, errors, cv::Size(15, 15), 0);

			std::vector<Match> matchs;
			for (int i = 0; i < (int)status.size(); ++i) {
				if (status[i] == 1) {
					matchs.push_back({ features[i] ,nextFeatures[i] ,errors[i] });
				}
			}
			//std::cout << matchs.size() / (double)status.size() << std::endl;

			std::sort(matchs.begin(), matchs.end(), [](const Match & a, const Match & b) { return a.error < b.error; });

			int numBestMatch = (int)(ratio*matchs.size());
			std::vector<cv::Point2f> inputFeatures(numBestMatch), outputFeatures(numBestMatch);

			for (int i = 0; i < numBestMatch; ++i) {
				inputFeatures[i] = matchs[i].in;
				outputFeatures[i] = matchs[i].out;
			}

			if (viz) {
				cv::Mat corresp_viz = nextFrame.clone();
				for (int i = 0; i < numBestMatch; ++i) {
					cv::circle(corresp_viz, matchs[i].in, 5, cv::Scalar(0, 255, 0), 2);
					cv::circle(corresp_viz, matchs[i].out, 5, cv::Scalar(255, 0, 0), 2);
				}
				cv::imshow("viz", corresp_viz);
				if (cv::waitKey() == 27) {
					viz = false;
					cv::destroyAllWindows();
				}
			}

			cv::Mat homography = cv::findHomography(inputFeatures, outputFeatures, cv::RANSAC, ransac_repro_error);

			cv::Mat correctedFrame;
			cv::warpPerspective(nextFrame, correctedFrame, homography.inv(), nextFrame.size());

			out << correctedFrame;
			std::cout << "." << std::flush;
		}

		if (viz) {
			cv::destroyAllWindows();
		}

		out.close();
		std::cout << " done " << std::endl;
	}

	void VideoUtils::smallAlignmentVideo2(sibr::Video & vid, const std::string & outputVidPath, bool viz)
	{
		struct Match {
			cv::Point2f in, out;
			float error;
		};

		//cv::VideoWriter out(outputVidPath, codec_OpenH264, vid.getFrameRate(), cv::Size(vid.getResolution()[0], vid.getResolution()[1]));
		sibr::FFVideoEncoder out(outputVidPath, vid.getFrameRate(), vid.getResolution());

		if (!out.isFine()) {
			SIBR_WRG << " cant write video " << outputVidPath << std::endl;
		}
		vid.setCurrentFrame(0);
		cv::Mat initFrame = vid.next();
		cv::Mat initGray = VideoUtils::getGrey(initFrame);

		std::vector<cv::Point2f> features, nextFeatures;
		std::vector<uchar> status;
		std::vector<float> errors;

		cv::Mat completeHomography = cv::Mat::eye(3, 3, CV_64FC1);

		const double magic_expon = 2.0;
		const double ratio = 0.5;
		const double ransac_repro_error = 0.5;
		const double features_to_track_quality = 0.1;
		const double features_min_dist = 10; //10
		const double max_displacement = 2;
		int nPixels = vid.getResolution().prod();

		int numFeatures = (int)pow(nPixels, 1.0 / magic_expon);
		std::cout << " num features " << numFeatures << std::endl;

		cv::Mat nextFrame, gray;

		for (;;) {
			nextFrame = vid.next();
			if (nextFrame.empty()) {
				break;
			}
			gray = VideoUtils::getGrey(nextFrame);

			cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

			cv::goodFeaturesToTrack(initGray, features, numFeatures, features_to_track_quality, features_min_dist);

			cv::calcOpticalFlowPyrLK(initGray, gray, features, nextFeatures, status, errors, cv::Size(5, 5), 0);

			std::vector<Match> matchs;
			for (int i = 0; i < (int)status.size(); ++i) {
				auto v = features[i] - nextFeatures[i];

				if (status[i] == 1 && cv::norm(cv::Vec2f(v.x, v.y), cv::NORM_INF) < max_displacement) {
					matchs.push_back({ features[i] ,nextFeatures[i] ,errors[i] });
				}
			}
			//std::cout << matchs.size() / (double)status.size() << std::endl;

			std::sort(matchs.begin(), matchs.end(), [](const Match & a, const Match & b) { return a.error < b.error; });

			int numBestMatch = (int)(ratio*matchs.size());
			std::vector<cv::Point2f> inputFeatures(numBestMatch), outputFeatures(numBestMatch);

			for (int i = 0; i < numBestMatch; ++i) {
				inputFeatures[i] = matchs[i].in;
				outputFeatures[i] = matchs[i].out;
			}

			if (viz) {
				cv::Mat corresp_viz = nextFrame.clone();
				for (int i = 0; i < numBestMatch; ++i) {
					cv::circle(corresp_viz, matchs[i].in, 5, cv::Scalar(0, 255, 0), 2);
					cv::circle(corresp_viz, matchs[i].out, 5, cv::Scalar(255, 0, 0), 2);
				}
				cv::imshow("viz", corresp_viz);
				if (cv::waitKey() == 27) {
					viz = false;
					cv::destroyAllWindows();
				}
			}

			cv::Mat homography = cv::findHomography(inputFeatures, outputFeatures, cv::RANSAC, ransac_repro_error);

			completeHomography *= homography;

			cv::Mat correctedFrame;
			cv::warpPerspective(nextFrame, correctedFrame, completeHomography.inv(), nextFrame.size());

			initGray = gray;

			out << correctedFrame;
			std::cout << "." << std::flush;
		}

		if (viz) {
			cv::destroyAllWindows();
		}

		out.close();
		std::cout << " done " << std::endl;
	}

	cv::Mat VideoUtils::applyFlow(const cv::Mat & prev, const cv::Mat & flow) {
		cv::Mat out, realFlow = flow;
		for (int i = 0; i < prev.rows; ++i) {
			for (int j = 0; j < prev.cols; ++j) {
				realFlow.at<cv::Vec2f>(i, j) += cv::Vec2f(j + 0.5f, i + 0.5f);
			}
		}
		cv::remap(prev, out, realFlow, cv::Mat(), cv::INTER_LINEAR);
		return out;
	}

	void VideoUtils::simpleFlow(cv::VideoCapture & cap, float ratio,
		std::function<bool(cv::Mat prev, cv::Mat next, cv::Mat flow, int flow_id)> f,
		std::function<void(void)> end_function
	) {
		cap.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, 0);
		cv::Mat  prev, flow;
		int flow_id = 0;
		while (true) {

			cv::Mat next;
			cap >> next;
			if (next.empty()) {
				break;
			}
			cv::resize(next, next, cv::Size((int)(ratio*next.size().width), (int)(ratio*next.size().height)));

			if (!prev.empty()) {
				cv::optflow::calcOpticalFlowSF(prev, next, flow, 3, 2, 4);

				if (!f(prev, next, flow, flow_id)) {
					break;
				}

				++flow_id;
			}

			prev = next;
		}

		end_function();
	}

	void VideoUtils::simpleFlowViz(cv::VideoCapture & cap, float ratio)
	{
		simpleFlow(cap, ratio, [](cv::Mat prev, cv::Mat next, cv::Mat flow, int flow_id) {
			cv::Mat viz = getFlowViz(flow);
			cv::resize(viz, viz, cv::Size(2000, 1500));
			cv::Mat diff = VideoUtils::applyFlow(prev, flow);

			cv::imshow("simpleflow", viz);
			cv::imshow("frame", next);
			cv::imshow("applyFlow", diff);
			int key = cv::waitKey(1);
			if (key == 27) {
				return false;
			}

			return true;
		}, []() {
			cv::destroyAllWindows();
		}
		);
	}

	void VideoUtils::simpleFlowSave(cv::VideoCapture & cap, float ratio, std::function<std::string(int flow_id)> naming_f)
	{
		std::cout << " saving flow " << std::flush;
		simpleFlow(cap, ratio, [&](cv::Mat prev, cv::Mat next, cv::Mat flow, int flow_id) {
			std::cout << "." << std::flush;
			cv::Mat viz = getFlowViz(flow);
			viz.convertTo(viz, CV_8UC3, 255.0);
			return cv::imwrite(naming_f(flow_id), viz);
		});
		std::cout << "done" << std::endl;
	}

	void VideoUtils::deepFlow(cv::VideoCapture & cap, float ratio,
		std::function<bool(cv::Mat prev, cv::Mat next, cv::Mat flow, int flow_id)> f,
		std::function<void(void)> end_function)
	{
		cap.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, 0);

		auto deepFlow = cv::optflow::createOptFlow_DeepFlow();
		cv::Mat flow, next, nextGrey, prevGrey;
		int flow_id = 0;
		while (true) {
			cap >> next;
			if (next.empty()) {
				break;
			}

			auto size = cv::Size((int)(ratio*next.size().width), (int)(ratio*next.size().height));
			cv::resize(next, next, size);
			nextGrey = getGrey(next);

			if (!prevGrey.empty()) {
				deepFlow->calc(prevGrey, nextGrey, flow);

				if (!f(prevGrey, nextGrey, flow, flow_id)) {
					break;
				}
				++flow_id;
			}

			prevGrey = nextGrey.clone();
		}

		end_function();

	}

	void VideoUtils::deepFlowViz(cv::VideoCapture & cap, float ratio)
	{
		deepFlow(cap, ratio, [](cv::Mat prev, cv::Mat next, cv::Mat flow, int flow_id) {
			cv::Mat viz = getFlowViz(flow);
			cv::Mat diff = applyFlow(prev, flow);
			cv::imshow("simpleflow", viz);
			cv::imshow("frame", prev);
			cv::imshow("applyFlow", diff);
			int key = cv::waitKey(1);
			if (key == 27) {
				return false;
			}
			return true;
		}, []() {
			cv::destroyAllWindows();
		}
		);
	}

	cv::Mat VideoUtils::getGrey(const cv::Mat & mat)
	{
		cv::Mat out;
		cv::cvtColor(mat, out, cv::COLOR_BGR2GRAY);
		return out;
	}

	void PyramidLayer::show(int s) const
	{
		int slice_y = 0;

		struct Data {
			PyramidLayer A;
		};
		Data data = { *this };

		auto cb = [](int pos, void* userdata) -> void {
			Data & d = *(Data*)userdata;
			cv::Mat sliceA = sibr::slice(d.A, 0, pos);
			cv::resize(sliceA, sliceA, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);
			cv::imshow("sliceA", sliceA);
		};


		int t = 0;
		while (true) {

			cv::Mat slice;
			cv::Mat shifted = (volume.row(t) + 0 * 128.0f);
			shifted.reshape(3, h).convertTo(slice, CV_8UC3);
			cv::imshow("shpw", slice);
			cv::createTrackbar("sy", "shpw", &slice_y, w - 1, cb, &data);
			if (cv::waitKey(s) == 27) {
				break;
			}
			++t;
			if (t == l) {
				std::cout << "." << std::flush;
				t = 0;
			}
		}
	}

	cv::Mat PyramidLayer::getRGB(int frame, bool centered) {
		cv::Mat out;
		if (centered) {
			cv::Mat shifted = (volume.row(frame) + 128.0f);
			shifted.reshape(3, h).convertTo(out, CV_8UC3);
		} else {
			volume.row(frame).reshape(3, h).convertTo(out, CV_8UC3);
		}

		return out;
	}

	void PyramidLayer::saveToVideoFile(const std::string & filename, double framerate)
	{
		sibr::FFVideoEncoder output(filename, framerate, { w,h });
		for (int f = 0; f < l; ++f) {
			cv::Mat frame;
			volume.row(f).reshape(3, h).convertTo(frame, CV_8UC3);
			output << frame;
		}
		output.close();

	}

	void PyramidLayer::show(PyramidLayer A, PyramidLayer B, int s) {
		int t = 0;
		while (true) {
			cv::Mat sliceA = A.getRGB(t);
			cv::Mat sliceB = B.getRGB(t);
			cv::Mat top;
			cv::hconcat(sliceA, sliceB, top);
			cv::imshow("show duo", top);
			if (cv::waitKey(s) == 27) {
				break;
			}
			++t;
			if (t == A.l) {
				std::cout << "." << std::flush;
				t = 0;
			}
		}
	}

	void PyramidLayer::show(PyramidLayer A, PyramidLayer B, PyramidLayer C, int s) {
		int slice_x = 0;
		int slice_y = 0;
		int t = 0;

		struct Data {
			PyramidLayer A, B, C;
		};
		Data data = { A,B,C };

		auto cb = [](int pos, void* userdata) -> void {
			Data & d = *(Data*)userdata;
			cv::Mat sliceA = sibr::slice(d.A, 0, pos);
			cv::Mat sliceB = sibr::slice(d.B, 0, pos);
			cv::Mat sliceC = sibr::slice(d.C, 0, pos);
			cv::resize(sliceA, sliceA, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);
			cv::resize(sliceB, sliceB, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);
			cv::resize(sliceC, sliceC, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);

			cv::imshow("sliceA", sliceA);
			cv::imshow("sliceB", sliceB);
			cv::imshow("sliceC", sliceC);
		};

		while (true) {
			cv::imshow("show A", A.getRGB(t));
			cv::imshow("show B", B.getRGB(t));
			cv::imshow("show C", C.getRGB(t));

			cv::createTrackbar("sy", "show C", &slice_y, A.w - 1, cb, &data);

			int k = cv::waitKey(s);
			if (k == 27) {
				break;
			} else if (k == 'c') {
				t = t > 0 ? t - 1 : A.l - 1;
			} else {
				++t;
				if (t == A.l) {
					std::cout << "." << std::flush;
					t = 0;
				}
			}
		}
	}

	void PyramidLayer::show(PyramidLayer A, PyramidLayer B, PyramidLayer C, PyramidLayer D, int s, bool centered) {
		int slice_y = 0;
		int t = 0;
		struct Data {
			PyramidLayer A, B, C, D;
			bool center;
		};
		Data data = { A,B,C,D, centered };

		auto cb = [](int pos, void* userdata) -> void {
			Data & d = *(Data*)userdata;
			cv::Mat sliceA = sibr::slice(d.A, 0, pos, true, d.center);
			cv::Mat sliceB = sibr::slice(d.B, 0, pos, true, d.center);
			cv::Mat sliceC = sibr::slice(d.C, 0, pos, true, d.center);
			cv::Mat sliceD = sibr::slice(d.D, 0, pos, true, d.center);
			cv::resize(sliceA, sliceA, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);
			cv::resize(sliceB, sliceB, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);
			cv::resize(sliceC, sliceC, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);
			cv::resize(sliceD, sliceD, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);
			cv::imshow("sliceA", sliceA);
			cv::imshow("sliceB", sliceB);
			cv::imshow("sliceC", sliceC);
			cv::imshow("sliceD", sliceD);
		};

		while (true) {
			cv::imshow("show A", A.getRGB(t, data.center));
			cv::imshow("show B", B.getRGB(t, data.center));
			cv::imshow("show C", C.getRGB(t, data.center));
			cv::imshow("show D", D.getRGB(t, data.center));
			cv::createTrackbar("sy", "show C", &slice_y, A.w - 1, cb, &data);

			int k = cv::waitKey(s);
			if (k == 27) {
				std::cout << "clear" << std::endl;
				cv::destroyAllWindows();
				break;
			} else if (k == 'c') {
				t = t > 0 ? t - 1 : A.l - 1;
			} else if (k == 'm') {
				data.center = !data.center;
			} else {
				++t;
				if (t == A.l) {
					std::cout << "." << std::flush;
					t = 0;
				}
			}
		}
	}

	void PyramidLayer::showDiff(PyramidLayer A, PyramidLayer B, int s)
	{
		int t = 0;
		while (true) {
			cv::Mat sliceA = A.getRGB(t);
			cv::Mat sliceB = B.getRGB(t);
			cv::Mat diff;
			cv::absdiff(sliceA, sliceB, diff);
			cv::Mat top;
			//cv::hconcat(sliceB, 5 * diff, top);
			//cv::hconcat(sliceA, top, top);
			cv::imshow("show a", sliceA);
			cv::imshow("show B", sliceB);

			cv::imshow("show diff", diff);
			int k = cv::waitKey(s);
			if (k == 27) {
				break;
			} else if (k == 'c') {
				t = t > 0 ? t - 1 : A.l - 1;
			} else {
				++t;
				if (t == A.l) {
					std::cout << "." << std::flush;
					t = 0;
				}
			}
		}
	}

	PyramidLayer PyramidLayer::operator+(const PyramidLayer & other)
	{
		//assert(w == other.w && h == other.w && l = other.l);

		PyramidLayer out;
		out.w = w;
		out.h = h;
		out.l = l;
		out.volume = volume + other.volume;
		return out;
	}

	PyramidLayer PyramidLayer::operator-(const PyramidLayer & other)
	{
		//assert(w == other.w && h == other.w && l = other.l);

		PyramidLayer out;
		out.w = w;
		out.h = h;
		out.l = l;
		out.volume = volume - other.volume;
		return out;
	}

	PyramidLayer blur(const PyramidLayer & layer, const PyramidParameters &  params)
	{
		PyramidLayer out(layer.w, layer.h, layer.l);

#pragma omp parallel for
		for (int t = 0; t < layer.l; ++t) {
			cv::Mat sliceIn = layer.volume.row(t).reshape(3, layer.h);
			cv::Mat sliceOut = out.volume.row(t).reshape(3, out.h);
			if (params.splacialDS) {
				cv::GaussianBlur(sliceIn, sliceOut, cv::Size(2 * params.spatial_radius + 1, 2 * params.spatial_radius + 1), 0);
			} else {
				sliceIn.copyTo(sliceOut);
			}
		}

		temporalBlur(out, params);

		return out;
	}

	//	PyramidLayer temporalBlur(const PyramidLayer & layer, const PyramidParameters &  params, float scaling)
	//	{
	//		//cv::Mat kernel = cv::Mat::ones(cv::Size(2 * params.temporal_radius + 1, 1), CV_32FC1);
	//		//kernel = kernel / cv::norm(kernel, cv::NORM_L1);
	//
	//		const cv::Mat kernel = (scaling / 16.0f)*(cv::Mat_<float>(1, 5) << 1, 4, 6, 4, 1);
	//
	//		//layer.show(0);
	//
	//		cv::Mat vol = layer.volume.t();
	//
	//		//std::cout << vol.row(3 * (0 * layer.w + 0) + 0) << std::endl;
	//		//std::cout << vol.row(3 * (0 * layer.w + 0) + 1) << std::endl;
	//		//std::cout << vol.row(3 * (0 * layer.w + 0) + 2) << std::endl;
	//
	//		cv::filter2D(vol, vol, -1, kernel, { 2,0 }, 0, cv::BORDER_DEFAULT);
	//
	////#pragma omp parallel for
	////		for (int i = 0; i < layer.h; ++i) {
	////			for (int j = 0; j < layer.w; ++j) {
	////				for (int c = 0; c < 3; ++c) {
	////					cv::Mat slice = vol.row(3 * (i*layer.w + j) + c);
	////					cv::filter2D(slice, slice, -1, kernel, { 2,0 }, 0, cv::BORDER_DEFAULT); // cv::Point(-1,-1) BORDER_ISOLATED  cv::BORDER_REPLICATE
	////				}
	////			}
	////		}
	//
	//		//std::cout << vol.row(3 * (0 * layer.w + 0) + 0) << std::endl;
	//		//std::cout << vol.row(3 * (0 * layer.w + 0) + 1) << std::endl;
	//		//std::cout << vol.row(3 * (0 * layer.w + 0) + 2) << std::endl;
	//
	//		PyramidLayer out(vol.t(), layer.w, layer.h);
	//		//out.cout();
	//		//out.show(0);
	//
	//		return out;
	//	}

	PyramidLayer temporalBlur(const PyramidLayer & layer, const PyramidParameters &  params, float scaling)
	{
		const cv::Mat kernel = (scaling / 16.0f)*(cv::Mat_<float>(5, 1) << 1, 4, 6, 4, 1);
		cv::Mat vol = layer.volume.clone();
		cv::filter2D(vol, vol, -1, kernel, { -1,-1 }, 0.0, cv::BORDER_DEFAULT);
		return PyramidLayer(vol, layer.w, layer.h);
	}

	void temporalBlurInPlace(PyramidLayer & layer, const PyramidParameters & params, float scaling)
	{
		const cv::Mat kernel = (scaling / 16.0f)*(cv::Mat_<float>(5, 1) << 1, 4, 6, 4, 1);
		cv::filter2D(layer.volume, layer.volume, -1, kernel, { -1,-1 }, 0.0, cv::BORDER_DEFAULT);
	}

	PyramidLayer decimate(const PyramidLayer & layer, const PyramidParameters &  params)
	{

		PyramidLayer out((layer.w + 1) / 2, (layer.h + 1) / 2, (layer.l + 1) / 2);

#pragma omp parallel for
		for (int t = 0; t < out.l; ++t) {
			cv::Mat sliceCurrent = layer.volume.row(2 * t).reshape(3, layer.h);
			cv::Mat sliceDecimated = out.volume.row(t).reshape(3, out.h);

			cv::pyrDown(sliceCurrent, sliceDecimated);
			//cv::resize(sliceCurrent, sliceDecimated, sliceDecimated.size(), 0, 0, CV_INTER_NN);

		}

		return out;

	}

	PyramidLayer upscale(const PyramidLayer & layerUp, const PyramidLayer & layerDown, const PyramidParameters &  params)
	{
		//std::cout << layerUp.w << " " << layerUp.h << " " << layerUp.l << std::endl;
		//std::cout << layerDown.w << " " << layerDown.h << " " << layerDown.l << std::endl;

		PyramidLayer out(layerUp.w, layerUp.h, layerUp.l);
		//#pragma omp parallel for
		for (int t = 0; t < layerDown.l; ++t) {
			//if (2 * t + 1 >= layerUp.l) {
			//	continue;
			//}
			//cv::Mat sliceUp = out.volume.row(2 * t + 1).reshape(3, layerUp.h);
			cv::Mat sliceUp = out.volume.row(2 * t).reshape(3, layerUp.h);
			cv::Mat sliceDown = layerDown.volume.row(t).reshape(3, layerDown.h);

			if (params.splacialDS) {
				cv::pyrUp(sliceDown, sliceUp, sliceUp.size());
			} else {
				sliceDown.copyTo(sliceUp);
			}


			//	if (2 * t + 1 < layerUp.l) {
			//		sliceUp.copyTo(out.volume.row(2 * t + 1).reshape(3, layerUp.h));
			//	}
		}
		temporalBlurInPlace(out, params, 2.0f);
		return out;
	}

	PyramidLayer downscale(const PyramidLayer & layer, const PyramidParameters &  params)
	{

		PyramidLayer blured = temporalBlur(layer, params);
		//std::cout << " temporal blur " << std::endl;
		//blured.show();
		//std::cout << " temporal blur end" << std::endl;

		PyramidLayer out;

		if (params.splacialDS) {
			out = PyramidLayer((layer.w + 1) / 2, (layer.h + 1) / 2, (layer.l + 1) / 2);
		} else {
			out = PyramidLayer(layer.w, layer.h, (layer.l + 1) / 2);
		}

		//#pragma omp parallel for
		for (int t = 0; t < out.l; ++t) {
			//if (2 * t + 1 >= layer.l) {
			//	continue;
			//}
			//cv::Mat sliceCurrent = blured.volume.row(2 * t + 1).reshape(3, layer.h);
			cv::Mat sliceCurrent = blured.volume.row(2 * t).reshape(3, layer.h);
			cv::Mat sliceDecimated = out.volume.row(t).reshape(3, out.h);

			if (params.splacialDS) {
				cv::pyrDown(sliceCurrent, sliceDecimated);
			} else {
				sliceCurrent.copyTo(sliceDecimated);
			}

		}

		//PyramidLayer blur (layer.w, layer.h, (layer.l + 1) / 2);

		//for (int i = 0; i < blur.volume.rows; ++i) {
		//	cv::pyrDown(layer.volume.row(i), blur.volume.row(i), blur.volume.row(i).size());
		//}

		//PyramidLayer out;
		//if (params.splacialDS) {
		//	out = PyramidLayer((layer.w + 1) / 2, (layer.h + 1) / 2, blur.l);
		//} else {
		//	out = PyramidLayer(layer.w, layer.h, blur.l);
		//}

		//for (int t = 0; t < out.l; ++t) {
		//	if (2 * t + 1 >= layer.l) {
		//		continue;
		//	}
		//	cv::Mat sliceCurrent = blur.volume.row(2 * t + 1).reshape(3, layer.h);
		//	cv::Mat sliceDecimated = out.volume.row(t).reshape(3, out.h);

		//	if (params.splacialDS) {
		//		cv::pyrDown(sliceCurrent, sliceDecimated);
		//	} else {
		//		sliceCurrent.copyTo(sliceDecimated);
		//	}
		//}

		return out;
	}

	SIBR_VIDEO_EXPORT cv::Mat slice(const PyramidLayer & layer, int i, int j, bool vertical, bool center)
	{
		cv::Mat out;
		if (vertical) {
			out = cv::Mat(layer.l, layer.h, CV_8UC3);

			for (int t = 0; t < layer.l; ++t) {
				for (int i = 0; i < layer.h; ++i) {
					for (int c = 0; c < 3; ++c) {
						out.at<cv::Vec3b>(t, i)[c] = (uchar)sibr::clamp((int)layer.volume.at<float>(t, 3 * (i*layer.w + j) + c) + (center ? 128 : 0), 0, 255);
					}
				}
			}
		}

		//cv::vconcat(out, out, out);
		out = out.t();
		return out;
	}

	PyramidLayer VideoLaplacianPyramid::collapse() const
	{
		PyramidLayer out = layers.back();
		for (int i = (int)layers.size() - 2; i >= 0; --i) {
			PyramidLayer up = upscale(layers[i], out, params);
			out = up + layers[i];
		}
		return out;
	}

	VideoGaussianPyramid buildVideoGaussianPyramid(const cv::Mat & volume, int w, int h, int nLevels, const PyramidParameters & params, bool show)
	{
		VideoGaussianPyramid out;
		out.params = params;
		PyramidLayer currentLayer(volume, w, h);
		out.layers.push_back(currentLayer);

		for (int i = 1; i < nLevels; ++i) {
			PyramidLayer down = downscale(currentLayer, params);
			out.layers.push_back(down);
			currentLayer = down;
			if (show) {
				currentLayer.show();
			}
		}

		return out;
	}

	VideoGaussianPyramid buildVideoGaussianPyramid(sibr::Video & vid, int nLevels, const PyramidParameters & params, bool show)
	{
		return buildVideoGaussianPyramid(vid.getVolume(), vid.getResolution()[0], vid.getResolution()[1], nLevels, params, show);
	}

	SIBR_VIDEO_EXPORT VideoLaplacianPyramid buildVideoLaplacianPyramid(PyramidLayer vid, int nLevels, const PyramidParameters & params, bool show)
	{
		VideoLaplacianPyramid out;
		out.params = params;

		PyramidLayer currentLayer = vid;
		currentLayer.volume.convertTo(currentLayer.volume, CV_32FC1);

		for (int i = 0; i < nLevels - 1; ++i) {
			PyramidLayer down = downscale(currentLayer, params);
			PyramidLayer up = upscale(currentLayer, down, params);
			if (show) {
				up.show();
			}
			out.layers.push_back(currentLayer - up);
			currentLayer = down;
		}

		out.layers.push_back(currentLayer);

		return out;
	}

	VideoLaplacianPyramid buildVideoLaplacianPyramid(sibr::Video & vid, int nLevels, const PyramidParameters & params, bool show) {
		PyramidLayer layer(vid.getVolume(), vid.getResolution()[0], vid.getResolution()[1]);
		return buildVideoLaplacianPyramid(layer, nLevels, params, show);
	}

	VideoLaplacianPyramid buildVideoLaplacianPyramidFullyReduced(PyramidLayer vid, int nLevels, const PyramidParameters & params, bool show)
	{
		VideoLaplacianPyramid standardPyramid = buildVideoLaplacianPyramid(vid, nLevels, params, show);

		VideoLaplacianPyramid out;
		out.params = params;

		out.layers.push_back(standardPyramid.layers[0]);

		for (int i = 1; i < nLevels; ++i) {

			PyramidLayer diff = standardPyramid.layers[i].clone();
			for (int k = i - 1; k >= 0; --k) {
				diff = upscale(standardPyramid.layers[k], diff, params);
			}
			std::cout << " layer " << i << " : ";
			diff.cout();
			out.layers.push_back(diff);
		}

		return out;
	}

	SIBR_VIDEO_EXPORT void convertReducedVideoPyramidTo128(VideoLaplacianPyramid & vid)
	{
		int nLayers = (int)vid.layers.size();
		for (int l = 0; l < nLayers - 1; ++l) {
			vid.layers[l].volume += 128.0;
		}
	}

	PyramidLayer videoLaplacianBlending(sibr::Video & vidA, sibr::Video & vidB, PyramidLayer mask_volume)
	{
		int num_lvls = 6;
		auto pyrA = sibr::buildVideoLaplacianPyramid(vidA, num_lvls);
		auto pyrB = sibr::buildVideoLaplacianPyramid(vidB, num_lvls);
		auto pyrM = sibr::buildVideoGaussianPyramid(mask_volume.volume, mask_volume.w, mask_volume.h, num_lvls);

		VideoLaplacianPyramid out;
		for (int l = 0; l < num_lvls; ++l) {
			sibr::PyramidLayer layer;
			layer.w = pyrA.layers[l].w;
			layer.h = pyrA.layers[l].h;
			layer.l = pyrA.layers[l].l;

			cv::Mat A_lvl = pyrA.layers[l].volume;
			cv::Mat B_lvl = pyrB.layers[l].volume;
			cv::Mat M_lvl = pyrM.layers[l].volume;

			layer.volume = A_lvl.mul(M_lvl) + B_lvl.mul(1.0f - M_lvl);
			out.layers.push_back(layer);
		}

		return out.collapse();
	}

	PyramidLayer videoLaplacianBlending(PyramidLayer vidA, PyramidLayer vidB, PyramidLayer mask_volume, PyramidParameters params, bool show)
	{
		int num_lvls = params.num_levels;

		auto pyrA = sibr::buildVideoLaplacianPyramid(vidA, num_lvls, params, show);
		auto pyrB = sibr::buildVideoLaplacianPyramid(vidB, num_lvls, params, show);
		auto pyrM = sibr::buildVideoGaussianPyramid(mask_volume.volume, mask_volume.w, mask_volume.h, num_lvls, params, show);

		VideoLaplacianPyramid out;
		out.params = params;
		for (int l = 0; l < num_lvls; ++l) {

			std::cout << l << std::endl;

			sibr::PyramidLayer layer;
			layer.w = pyrA.layers[l].w;
			layer.h = pyrA.layers[l].h;
			layer.l = pyrA.layers[l].l;

			cv::Mat A_lvl = pyrA.layers[l].volume;
			cv::Mat B_lvl = pyrB.layers[l].volume;
			cv::Mat M_lvl = pyrM.layers[l].volume;


			cv::Mat rev_Mask = 255 - M_lvl;
			sibr::PyramidLayer test(rev_Mask, pyrM.layers[l].w, pyrM.layers[l].h);

			//pyrA.layers[l].show();
			//pyrB.layers[l].show();
			//pyrM.layers[l].show();

			//test.show();

			cv::Mat normalized_mask = (1 / 255.0)*M_lvl;
			cv::Mat normalized_mask_r = (1 / 255.0)*test.volume;

			layer.volume = A_lvl.mul(normalized_mask) + B_lvl.mul(normalized_mask_r);
			//layer.show();

			out.layers.push_back(layer);
		}

		return out.collapse();
	}

	std::vector<FullContribData> videoLaplacianBlendingContrib(PyramidLayer vidA, PyramidLayer vidB, PyramidLayer mask_volume, PyramidParameters params)
	{
		int num_lvls = params.num_levels;

		auto pyrA = sibr::buildVideoLaplacianPyramid(vidA, num_lvls, params);
		auto pyrB = sibr::buildVideoLaplacianPyramid(vidB, num_lvls, params);
		auto pyrM = sibr::buildVideoGaussianPyramid(mask_volume.volume, mask_volume.w, mask_volume.h, num_lvls, params);

		std::vector<FullContribData> out(num_lvls);

		VideoLaplacianPyramid l_out;
		l_out.params = params;
		l_out.layers.resize(num_lvls);

#pragma omp parallel for
		for (int l = 0; l < num_lvls; ++l) {

			FullContribData data;
			ContribData & data_s = data.scaled;
			ContribData & data_ns = data.notScaled;

			sibr::PyramidLayer layer;

			layer.w = pyrA.layers[l].w;
			layer.h = pyrA.layers[l].h;
			layer.l = pyrA.layers[l].l;

			cv::Mat A_lvl = pyrA.layers[l].volume;
			cv::Mat B_lvl = pyrB.layers[l].volume;
			cv::Mat M_lvl = pyrM.layers[l].volume;

			cv::Mat rev_Mask = 255 - M_lvl;
			sibr::PyramidLayer test(rev_Mask, pyrM.layers[l].w, pyrM.layers[l].h);

			cv::Mat normalized_mask = (1 / 255.0)*M_lvl;
			cv::Mat normalized_mask_r = (1 / 255.0)*test.volume;

			layer.volume = A_lvl.mul(normalized_mask) + B_lvl.mul(normalized_mask_r);
			l_out.layers[l] = layer;

			PyramidLayer mask = pyrM.layers[l];
			PyramidLayer partA = pyrA.layers[l];
			PyramidLayer partB = pyrB.layers[l];

			data_ns.contrib = layer;
			data_ns.mask = mask;
			data_ns.partA = partA;
			data_ns.partB = partB;

			for (int j = l - 1; j >= 0; --j) {
				layer = upscale(pyrA.layers[j], layer, params);
				mask = upscale(pyrA.layers[j], mask, params);
				partA = upscale(pyrA.layers[j], partA, params);
				partB = upscale(pyrA.layers[j], partB, params);
			}

			data_s.contrib = layer;
			data_s.mask = mask;
			data_s.partA = partA;
			data_s.partB = partB;

			out[l] = data;
		}

		out[0].result = l_out.collapse();

		return out;
	}

	void videoLaplacianBlendingDebug(PyramidLayer vidA, PyramidLayer vidB, PyramidLayer mask_volume, PyramidParameters params)
	{
		int num_lvls = params.num_levels;

		auto pyrA = sibr::buildVideoLaplacianPyramid(vidA, num_lvls, params);
		auto pyrB = sibr::buildVideoLaplacianPyramid(vidB, num_lvls, params);
		auto pyrM = sibr::buildVideoGaussianPyramid(mask_volume.volume, mask_volume.w, mask_volume.h, num_lvls, params);

		VideoLaplacianPyramid out;

		//struct LayerData {
		//	int i;
		//} data;


		for (int l = 0; l < num_lvls; ++l) {
			sibr::PyramidLayer layer;
			layer.w = pyrA.layers[l].w;
			layer.h = pyrA.layers[l].h;
			layer.l = pyrA.layers[l].l;

			cv::Mat A_lvl = pyrA.layers[l].volume;
			cv::Mat B_lvl = pyrB.layers[l].volume;
			cv::Mat M_lvl = pyrM.layers[l].volume;

			layer.volume = A_lvl.mul(M_lvl) + B_lvl.mul(1.0f - M_lvl);
			out.layers.push_back(layer);
		}

		auto final_res = out.collapse();

	}

	//int TimeHistogram::getModeId() const {
	//	return  std::distance(bins.begin(), std::max_element(bins.begin(), bins.end()));
	//}
	
}
