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


#include <string>
#include <core/graphics/Image.hpp>
#include "Video.hpp"
#include "Config.hpp"

// Forward libav declarations.
struct AVFrame;
struct AVFormatContext;
struct AVOutputFormat;
struct AVStream;
struct AVCodecContext;
struct AVCodec;
struct AVPacket;

namespace sibr {

	
	/** Video encoder using ffmpeg.
	Adapted from https://github.com/leixiaohua1020/simplest_ffmpeg_video_encoder/blob/master/simplest_ffmpeg_video_encoder/simplest_ffmpeg_video_encoder.cpp
	\ingroup sibr_video
	*/
	class SIBR_VIDEO_EXPORT FFVideoEncoder {

	public:

		/** Constructor.
		\param _filepath destination file, the extension will be used to infer the container type.
		\param fps target video framerate
		\param size target video size, should be even else a resize will happen
		\param forceResize resize frames that are not at the target dimensions instead of ignoring them
		*/
		FFVideoEncoder(
			const std::string & _filepath,
			double fps,
			const sibr::Vector2i & size,
			bool forceResize = false
		);

		/** \return true if the encoder was properly setup. */
		bool isFine() const;

		/** Close the file. */
		void close();

		/** Encode a frame.
		\param frame the frame to encode
		\return a success flag 
		*/
		bool operator << (cv::Mat frame);

		/** Encode a frame.
		\param frame the frame to encode
		\return a success flag 
		*/
		bool operator << (const sibr::ImageRGB & frame);

		/// Destructor.
		~FFVideoEncoder();

	protected:

		/** Setup the encoder.
		\param size the video target size, prfer using power of two.
		*/
		void init(const sibr::Vector2i & size);
		
		/** Encode a frame to the file.
		\param frame the frame to encode
		\return a success flag.
		*/
//#define HEADLESS
#ifndef HEADLESS
		bool encode(AVFrame *frame);
#endif 

		bool initWasFine = false; ///< Was the encoder init properly.
		bool needFree = false; ///< Is the file open.
		std::string filepath; ///< Destination path.
		int w, h; ///< Dimensions.
		int frameCount = 0; ///< Current frame.
		double fps; ///< Framerate.
		bool _forceResize = false; ///< Resize frames.
		
#ifndef HEADLESS
		AVFrame * frameYUV = NULL; ///< Working frame.
#endif
		cv::Mat cvFrameYUV; ///< Working frame data.
		sibr::Vector2i yuSize; ///< Working size.

#ifndef HEADLESS
		AVFormatContext* pFormatCtx; ///< Format context.
		AVOutputFormat* fmt; ///< Output format.
		AVStream* video_st; ///< Output stream.
		AVCodecContext* pCodecCtx; ///< Codec context.
		AVCodec* pCodec; ///< Codec.
		AVPacket * pkt; ///< Encoding packet.
		
#endif
		static bool ffmpegInitDone; ///< FFMPEG initialization status.

	};

}
