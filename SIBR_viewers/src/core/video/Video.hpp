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


#pragma once

#include "Config.hpp"

#include <core/graphics/Texture.hpp>
#include <core/graphics/GUI.hpp>
#include <opencv2/opencv.hpp>

// must install ffdshow 
#define CV_WRITER_CODEC cv::VideoWriter::fourcc('F','F','D','S')


namespace sibr
{

	/** Video loaded from a file using OpenCV VideoCapture and FFMPEG.
	* \ingroup sibr_video
	*/
	class SIBR_VIDEO_EXPORT Video
	{
		SIBR_CLASS_PTR(Video);
		
	public:

		/** Constructor.
		\param path the path to the video file
		\note No loading will be performed at construction. Call load.
		*/
		Video(const std::string & path = "") : filepath(path) {}

		/** Load from a given file on disk.
		\param path path to the video
		\return a success flag
		*/
		virtual bool load(const std::string & path);
		
		/** \return the video resolution. */
		const sibr::Vector2i & getResolution();
		
		/** \return the video resolution. */
		cv::Size getResolutionCV();

		/** \return the current frame ID. */
		int getCurrentFrameNumber();

		/** Seek a specific frame.
		\param i the frame ID to seek
		*/
		void setCurrentFrame(int i);

		/** \return the total number of frames. */
		int getNumFrames();

		/** \return the video framerate. */
		double getFrameRate();

		/** \return the path to the video file on disk. */
		const Path & getFilepath() const;

		/** \return true if the video has been loaded. */
		bool isLoaded();

		/** \return the ID of the codec used to decode the video. */
		int getCodec();

		/** Stop reading from the file. */
		virtual void release();

		/** Read a section of the video and store it in a cv::Mat, where
		each row contains a frame, stored as RGBRGBRGB... linearly.
		\param time_skiped_begin time to skip at the beginning of the video, in seconds
		\param time_skiped_end time to skip at the end of the video, in seconds
		\return the frames data stored as described above.
		*/
		cv::Mat getVolume(float time_skiped_begin = 0, float time_skiped_end = 0);

		/** Read a section of the video and store  otin a cv::Mat, where
		each row contains a frame, stored as RGBRGBRGB... linearly.
		\param starting_frame index of the first frame to extract
		\param ending_frame index of the last frame to extract
		\return the frames data stored as described above.
		*/
		cv::Mat getVolume(int starting_frame, int ending_frame);

		/** \return the next frame. */
		cv::Mat next();

		/** \return the underlying VideoCapture object. */
		cv::VideoCapture & getCVvideo();

		/** \return true if the video exists on disk. */
		bool exists() const;

	protected:
		
		/** Check if the video is loaded. */
		virtual void checkLoad();

		cv::VideoCapture cap; ///< Internal capture object.

		Path filepath; ///< The path to the video.
		sibr::Vector2i resolution; ///< Video resolution.
		int nFrames = 0; ///< Number of frames in the video.
		double frameRate = 0.0; ///< Video frame rate.
		int codec = 0; ///< Codec used to read the video.
		bool loaded = false; ///< Video loading status.
	};


	/** Load and display a video in a view, with playback options.
	* \ingroup sibr_video
	*/
	class SIBR_VIDEO_EXPORT VideoPlayer : public Video, public ZoomInterraction
	{

		SIBR_CLASS_PTR(VideoPlayer);

	public:

		/** Replay mode. */
		enum Mode { PAUSE, PLAY, SHOULD_CLOSE };

		using Transformation = std::function<cv::Mat(cv::Mat)>; ///< Image processing function.

		/** Constructor.
		\param filepath the path to the video file
		\param f a function to apply to each frame
		\note No loading will be performed at construction. Call load.
		*/
		VideoPlayer(const std::string & filepath = "", const std::function<cv::Mat(cv::Mat)>&  f = [](cv::Mat m) { return m; });
		
		/** Load a video from disk.
		\param path the path to the video on disk
		\return a success flag
		*/
		bool load(const std::string & path) override;

		/** Set a transformation function to apply to each frame.
		\param f the new transformation
		*/
		void setTransformation(const Transformation & f) { transformation = f; }
		
		/** Set the playback mode.
		\param _mode the new mode
		*/
		void setMode(Mode _mode) { mode = _mode; }

		/** \return the current display texture on the GPU. */
		const std::shared_ptr<sibr::Texture2DRGB> & getDisplayTex() const;

		/** Load the next frame, call once per rendering frame. 
		\note Internally calls updateCPU and updateGPU.
		*/
		void update();

		/** Display playback GUI.
		\param ratio_display a scaling factor that determine the size of the video on screen based on the video intrinsic size.
		*/
		void onGui(float ratio_display);
		
		/** Load the next frame to the CPU.
		\return a success flag
		*/
		bool updateCPU();

		/** Load the next frame to the GPU.
		\note You should call updateCPU first.
		*/
		void updateGPU();

		/** \return a reference to the current frame on the CPU. */
		const cv::Mat & getCurrentFrame() const { return tmpFrame; }

	protected:
		
		/** \return the current loading texture on the GPU. */
		std::shared_ptr<sibr::Texture2DRGB> & getLoadingTex() { return loadingTex ? ping : pong; }
		
		/// Load the next frame, on the CPU then the GPU.
		void loadNext();

		Mode mode = PAUSE; ///< Play mode.
		bool first = true; ///< Are we at the first frame.
		bool repeat_when_end = true; ///< Loop when reaching the end.
		int displayTex = 1; ///< Index of the display texture.
		int loadingTex = 1; ///< Index of the loading texture.
		std::shared_ptr<sibr::Texture2DRGB> ping,pong; ///< Double buffer textures.
		cv::Mat tmpFrame; ///< Scratch frame.
		Transformation transformation; ///< Transformation to apply to each frame.
		int current_frame_slider; ///< Slider position.
	};

	
 } // namespace sibr

