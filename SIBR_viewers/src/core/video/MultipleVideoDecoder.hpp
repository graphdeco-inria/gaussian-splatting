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

#include <core/video/Video.hpp>

namespace sibr
{

	/** Double-buffered texture, used to load/display video frames for instance.
	* \ingroup sibr_video
	*/
	template<uint N>
	struct PingPongTexture {
		using TexPtr = std::shared_ptr<sibr::Texture2D<uchar, N>>;

		/** \return the current loading texture. */
		TexPtr & getLoadingTex();

		/** \return the current display texture. */
		TexPtr & getDisplayTex();

		/** Update the content of the loading texture and swap the two textures.
		\param frame the new data
		*/
		template<typename ImgType>
		void update(const ImgType & frame);

		/** Load the frame into the loading texture.
		\param frame the new data
		*/
		template<typename ImgType>
		void updateGPU(const ImgType & frame);
		
		int displayTex = 1, loadingTex = 1; /// Textures indices.
		TexPtr ping, pong; ///< Textures.
		bool first = true; ///< First update.
	};


	/** Batch decoding of multiple videos at the same time, stored in a texture array.
	* \ingroup sibr_video
	*/
	template<typename T, uint N>
	struct MultipleVideoDecoder {
		using TexArray = sibr::Texture2DArray<T,N>;
		using TexArrayPtr = typename TexArray::Ptr;

		/** Update a set of video players to the next frame.
		\param videos the video players to udpate
		\note Internally calls both updateCPU and updateGPU.
		*/
		void update(const std::vector<sibr::VideoPlayer::Ptr> & videos) {
			updateCPU(videos);
			updateGPU(videos);

			loadingTexArray = (loadingTexArray + 1) % 2;

			if (first) {
				first = false;
			} else {
				displayTexArray = (displayTexArray + 1) % 2;
			}
		}

		/** Load the next frame on the CPU for a set of video players.
		\param videos the video players to udpate
		*/
		void updateCPU(const std::vector<sibr::VideoPlayer::Ptr> & videos) {
			size_t numVids = videos.size();

			for (size_t i = 0; i < numVids; ++i) {
				videos[i]->updateCPU();
			}

		}

		/** Upload the next frame to the GPU for a set of video players.
		\param videos the video players to udpate
		*/
		void updateGPU(const std::vector<sibr::VideoPlayer::Ptr> & videos) {
			size_t numVids = videos.size();
			std::vector<cv::Mat> frames(numVids);
			for (size_t i = 0; i < numVids; ++i) {
				if (std::is_same_v<T, uchar> && N == 3) {
					frames[i] = videos[i]->getCurrentFrame();
				} else {
					std::vector<cv::Mat> cs;
					cv::split(videos[i]->getCurrentFrame(), cs);
					frames[i] = cs[0];
				}			
			}

			if (getLoadingTexArray().get()) {
				getLoadingTexArray()->updateFromImages(frames);
			} else {
				getLoadingTexArray() = TexArrayPtr(new TexArray(frames));
			}
		}

		/** \return the current loading texture array. */
		TexArrayPtr & getLoadingTexArray() { return loadingTexArray ? ping : pong; }

		/** \return the current display texture array. */
		const TexArrayPtr & getDisplayTexArray() const { return displayTexArray ? ping : pong; }

		bool first = true; ///< First frame.
		int loadingTexArray = 1, displayTexArray = 1; ///< Texture indices.
		TexArrayPtr ping, pong; ///< Textures.
	};


	/** Batch decoding of multiple videos at the same time, stored in a texture array.
	* Support updating an arbitrary subset. 
	* \ingroup sibr_video
	*/
	template<typename T, uint N>
	struct MultipleVideoDecoderArray : public MultipleVideoDecoder<T,N> {
		using TexArray = sibr::Texture2DArray<T, N>;
		using TexArrayPtr = typename TexArray::Ptr;

		/** Update a set of video players to the next frame.
		\param videos the video players list
		\param slices the indices of the videos to update
		\note Internally calls both updateCPU and updateGPU.
		*/
		void update(const std::vector<sibr::VideoPlayer::Ptr> & videos, const std::vector<int> & slices) {
			updateCPU(videos, slices);
			updateGPU(videos, slices);

			loadingTexArray = (loadingTexArray + 1) % 2;

			if (first) {
				first = false;
			} else {
				displayTexArray = (displayTexArray + 1) % 2;
			}
		}

		/** Load the next frame on the CPU for a set of video players.
		\param videos the video players list
		\param slices the indices of the videos to update
		*/
		void updateCPU(const std::vector<sibr::VideoPlayer::Ptr> & videos, const std::vector<int> & slices) {
#pragma omp parallel for num_threads(4)
			for (int i = 0; i < (int)slices.size(); ++i) {
				videos[slices[i]]->updateCPU();
			}
		}

		/** Upload the next frame to the GPU for a set of video players.
		\param videos the video players list
		\param slices the indices of the videos to update
		*/
		void updateGPU(const std::vector<sibr::VideoPlayer::Ptr> & videos, const std::vector<int> & slices) {
			int numVids = (int)videos.size();
			int numSlices = (int)slices.size();

			std::vector<cv::Mat> frames(numVids);
			for (int s = 0; s < numSlices; ++s) {
				if (std::is_same_v<T, uchar> && N == 3) {
					frames[slices[s]] = videos[slices[s]]->getCurrentFrame();
				} else {
					std::vector<cv::Mat> cs;
					cv::split(videos[slices[s]]->getCurrentFrame(), cs);
					frames[slices[s]] = cs[0];
				}
			} 

			if (!getLoadingTexArray().get()) {
				getLoadingTexArray() = TexArrayPtr(new TexArray((uint)videos.size(), SIBR_GPU_LINEAR_SAMPLING));
			}

			CHECK_GL_ERROR;
			getLoadingTexArray()->updateSlices(frames, slices);
		}

	};


	// --- TYPEDEFS ----------------

	using PingPong4u = PingPongTexture<4>;
	using PingPong3u = PingPongTexture<3>;
	using PingPong1u = PingPongTexture<1>;
	using MultipleVideoDecoder1u = MultipleVideoDecoder<uchar, 1>;
	using MultipleVideoDecoder3u = MultipleVideoDecoder<uchar, 3>;
	using MultipleVideoDecoderArray1u = MultipleVideoDecoderArray<uchar, 1>;
	using MultipleVideoDecoderArray3u = MultipleVideoDecoderArray<uchar, 3>;

	// --- IMPLEMENTATION ----------------

	template<uint N>
	std::shared_ptr<sibr::Texture2D<uchar, N>> & PingPongTexture<N>::getLoadingTex()
	{
		return loadingTex ? ping : pong;
	}

	template<uint N>
	std::shared_ptr<sibr::Texture2D<uchar,N>> & PingPongTexture<N>::getDisplayTex()
	{
		return displayTex ? ping : pong;
	}

	template<uint N> template<typename ImgType>
	void PingPongTexture<N>::update(const ImgType & frame)
	{
		if (first) {
			updateGPU(frame);
			loadingTex = (loadingTex + 1) % 2;
			first = false;
			return;
		}

		updateGPU(frame);

		displayTex = (displayTex + 1) % 2;
		loadingTex = (loadingTex + 1) % 2;
	}

	template<uint N> template<typename ImgType>
	void PingPongTexture<N>::updateGPU(const ImgType & frame)
	{
		if (getLoadingTex()) {
			getLoadingTex()->update(frame);
		} else {
			getLoadingTex() = TexPtr(new sibr::Texture2D<uchar, N>(frame, SIBR_GPU_LINEAR_SAMPLING));
		}
	}

 } // namespace sibr
