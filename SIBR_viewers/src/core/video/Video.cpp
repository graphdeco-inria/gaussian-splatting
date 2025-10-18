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


#include "Video.hpp"
#include <opencv2/videoio.hpp>
// #include "VideoUtils.hpp"

namespace sibr
{
	bool Video::load(const std::string & path)
	{
		cap = cv::VideoCapture(path);
		filepath = path;
		loaded = cap.isOpened();
		if (loaded) {
			nFrames = (int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT);
			frameRate = (double)cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);
			resolution[0] = (int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
			resolution[1] = (int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
			codec = (int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FOURCC);
			SIBR_LOG << "[Video] " << path << " loaded." << std::endl;
		}
		return loaded;
	}

	const sibr::Vector2i & Video::getResolution() { 
		checkLoad();  
		return resolution; 
	}

	cv::Size Video::getResolutionCV() { 
		checkLoad();   
		return cv::Size(resolution[0], resolution[1]);
	}

	int Video::getCurrentFrameNumber() { 
		checkLoad();  
		return (int)cap.get(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES); 
	}
	
	void Video::setCurrentFrame(int i){ 
		checkLoad(); 
		cap.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, i); 
	}
	
	int Video::getNumFrames()  { 
		checkLoad();  
		return nFrames; 
	}
	
	double Video::getFrameRate() { 
		checkLoad(); 
		return frameRate; 
	}
	
	const Path & Video::getFilepath() const { 
		return filepath; 
	}
	
	bool Video::isLoaded() { 
		return loaded; 
	}

	int Video::getCodec() { 
		checkLoad(); 
		return codec; 
	}

	void Video::release()
	{
		cap = cv::VideoCapture();
		loaded = false;
	}

	cv::Mat Video::getVolume(float time_skiped_begin, float time_skiped_end)
	{
		const int starting_frame = (int)(time_skiped_begin * getFrameRate());
		const int finishing_frame = getNumFrames() - (int)(time_skiped_end*getFrameRate()) - 1;
		return getVolume(starting_frame, finishing_frame);
	}

	cv::Mat Video::getVolume(int starting_frame, int ending_frame)
	{
		checkLoad();

		const int w = getResolution()[0];
		const int h = getResolution()[1];
		const int nc = 3;

		const int npixels = w * h;
		const int N = npixels * nc;
		const int L = ending_frame - starting_frame + 1;

		cv::Mat volume(L, N, CV_8UC1);
		setCurrentFrame(starting_frame);
		for (int i = 0; i < L; ++i) {
			cv::Mat mat = volume.row(i).reshape(3, h);
			cap >> mat;
		}
		setCurrentFrame(0);

		return volume;
	}

	cv::Mat Video::next()
	{
		checkLoad();
		cv::Mat frame;
		cap >> frame;
		return frame;
	}

	cv::VideoCapture & Video::getCVvideo()
	{
		checkLoad();
		return cap;
	}

	bool Video::exists() const
	{
		return sibr::fileExists(getFilepath().string());
	}

	void Video::checkLoad()
	{
		if (!loaded) {
			if (!load(filepath.string())) {
				SIBR_ERR << "[Video] Could not open video " << filepath << std::endl;
			}	
		}
	}


	//------------------------------------------------------------

	VideoPlayer::VideoPlayer(const std::string & filepath, const std::function<cv::Mat(cv::Mat)> & f) :
		Video(filepath), transformation(f)
	{
	}

	bool VideoPlayer::load(const std::string & path) {
		VideoPlayer other;
		if (other.Video::load(path)) {
			*this = other;
			return true;
		}
		return false;
	}

	const std::shared_ptr<sibr::Texture2DRGB> & VideoPlayer::getDisplayTex() const
	{
		return displayTex ? ping : pong;
	}

	void VideoPlayer::update()
	{
		checkLoad();

		if (first) {
			loadNext();
			loadingTex = (loadingTex + 1) % 2;
			first = false;
			return;
		}

		if (mode != PLAY) {
			return;
		}

		loadNext();

		displayTex = (displayTex + 1) % 2;
		loadingTex = (loadingTex + 1) % 2;
	}

	void VideoPlayer::onGui(float ratio_display)
	{
		checkLoad();

		if (mode == PAUSE){
			if (ImGui::Button("Play")) {
				mode = PLAY;
			}
		} else if (mode == PLAY) {
			if (ImGui::Button("Pause")) {
				mode = PAUSE;
			}
		}
		ImGui::SameLine();
		ImGui::Checkbox("Repeat when finished", &repeat_when_end);

		current_frame_slider = getCurrentFrameNumber();
		ImGui::Separator();
		ImGui::PushScaledItemWidth(500);
		if (ImGui::SliderInt("timeline", &current_frame_slider, 1, getNumFrames())) {
			setCurrentFrame(current_frame_slider);
			loadingTex = displayTex;
			first = true;
		}
		ImGui::PopItemWidth();

		ImGui::Separator();

		if (getDisplayTex() && getDisplayTex()->handle() ) {
			std::string infos = "size : " + std::to_string((int)getDisplayTex()->w()) + " " + std::to_string((int)getDisplayTex()->h()) + ", framerate : " + std::to_string(getFrameRate());
			ImGui::Text(infos.c_str());
			sibr::Vector2f displayTexSize(getDisplayTex()->w(), getDisplayTex()->h());
			sibr::Vector2i viewResolution = (ratio_display*displayTexSize).cast<int>();
			
			sibr::ImageWithCallback(getDisplayTex()->handle(), viewResolution, callBackData, zoomData.topLeft(), zoomData.bottomRight());

			updateZoom(displayTexSize);
		}

	}

	bool VideoPlayer::updateCPU()
	{

		checkLoad();

		bool alreayEmpty = tmpFrame.empty();
		tmpFrame = next();
		if (!tmpFrame.empty()) {
			tmpFrame = transformation(tmpFrame);
			return true;
		} else {
			if (alreayEmpty) {
				SIBR_WRG << "[Video] Could not load next frames." << std::endl;
				return false;
			}
			if (repeat_when_end) {
				setCurrentFrame(0);
				return updateCPU();
			} else {
				mode = PAUSE;
			}
			return false;
		}	
	}

	void VideoPlayer::updateGPU()
	{
		if (getLoadingTex().get()) {
			getLoadingTex()->update(tmpFrame);
		} else {
			getLoadingTex() = std::shared_ptr<sibr::Texture2DRGB>(new sibr::Texture2DRGB(tmpFrame));
		}
	}

	void VideoPlayer::loadNext()
	{
		if (updateCPU()) {
			updateGPU();
		}
	}

} // namespace sibr
