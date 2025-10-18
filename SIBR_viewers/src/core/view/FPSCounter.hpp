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

# include <vector>
# include "core/view/Config.hpp"
# include <core/system/Vector.hpp>

# include <chrono>

namespace sibr
{

	/** Provde a small GUI panel to display the current framerate, smoothed over multiple frames.
	* \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT FPSCounter
	{
	public:
		typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point; ///< Time type.

		/** Constructor.
		\param overlayed if true, the GUI panel is always displayed on top of all others. 
		*/
		FPSCounter(const bool overlayed = true);

		/** Setup at a given screen location.
		\param position the position on screen (in pixels).
		*/
		void init(const sibr::Vector2f & position);

		/** generate the ImGui panel. */
		void render();

		/** Update state using external timing.
		\param deltaTime time elapsed since last udpate. 
		*/
		void update(float deltaTime);

		/** Update state using internal timer.
		\param doRender should the ImGui panel be genrated immediatly
		*/
		void update(bool doRender = true);

		/** Toggle the panel visibility. */
		void toggleVisibility() {
			_hidden = !_hidden;
		}

		/** \return true if the panel visible. */
		bool active() const {
			return !_hidden;
		}

	private:
		time_point							_lastFrameTime; ///< Last frame duration.
		sibr::Vector2f						_position; ///< on screen position.
		std::vector<float>					_frameTimes; ///< Last N frame times.
		size_t								_frameIndex; ///< Current position in the time list.
		float								_frameTimeSum; ///< Current running sum.
		int									_flags; ///< Imgui display flags.
		bool								_hidden; ///< Visibility status.
		std::string							_name; ///< Panel name.
		static int							_count; ///< Internal counter to avoid collision when multiple framerate panels are displayed.
	};

} // namespace sibr
