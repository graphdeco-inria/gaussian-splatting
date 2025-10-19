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

# include <core/system/Config.hpp>
# include <core/graphics/Texture.hpp>
# include <core/graphics/Image.hpp>
# include <core/graphics/Shader.hpp>
# include <core/view/ViewBase.hpp>

namespace sibr {

	/** Basic view to display an image and inspect it.
	 *Two modes are supported:
	 * * interactive, where the user can pan/zoom, rescale the values, display some channels, via the mouse and GUI.
	 * * fixed, where the image is displayed as is, without any modification possible.
	 */
	class SIBR_VIEW_EXPORT ImageView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(ImageView);
	public:

		/** Constructor.
		 * \param interactiveMode should the GUI panel be displayed and the user be able to pan/zoom into the image
		 */
		ImageView(bool interactiveMode = true);

		/** Render the image in the currently bound rendertarget.
		 *\param vpRender the region to render into
		 */
		void onRender(const Viewport & vpRender) override;

		/** Update user interactions.
		 *\param input the user input for the view
		 *\param vp the view viewport
		 */
		void onUpdate(Input& input, const Viewport & vp) override;

		/*** Render GUI panels. */
		void onGUI() override;

		/** Set an attachment of a rendertarget as the texture to display.
		 *\param rt the rendertarget to display
		 *\param handle the index of the attachment to display
		 *\warning Will only be valid until the RT is deleted.
		 */
		void setRenderTarget(const IRenderTarget & rt, uint handle = 0);
		
		/** Set the texture to display.
		 *\param tex the texture to display
		 *\warning Will only be valid until the texture is deleted.
		 */
		void setTexture(const ITexture2D& tex);

		/** Set an image as the texture to display. An internal copy of the image will be sent to the GPU.
		 *\param img the image
		 */
		template<typename T_Type, unsigned T_NumComp>
		void setImage(const Image<T_Type, T_NumComp> & img) {
			// Create texture on the fly.
			std::shared_ptr<Texture2D<T_Type, T_NumComp>> tex(new Texture2D<T_Type, T_NumComp>(img));
			_tex = tex;
			_texHandle = _tex->handle();
			_size.get()[0] = float(_tex->w());
			_size.get()[1] = float(_tex->h());
		}

		/** Set if the GUI panel should be displayed or not.
		 *\param opt display option
		 **/
		void showGUI(bool opt) {
			_showGUI = opt;
		}

		/** Set if the user should be able to pan/zoom the image
		 *\param opt interaction option
		 **/
		void allowInteraction(bool opt) {
			_allowInteraction = opt;
		}
		
	protected:

		ITexture2D::Ptr _tex; ///< Internal texture for the image input case.
		GLuint _texHandle = 0; ///< Texture to display.
		
		GLShader _display; ///< Shader.
		
		GLuniform<sibr::Vector4f> _minVal; ///< Normalization minimum.
		GLuniform<sibr::Vector4f> _maxVal; ///< Normalization maximum.
		bool _lockChannels = true; ///< Use the same normalization values for all channels.
		
		std::array<bool, 4> _showChannels; ///< Display which channels.
		GLuniform<sibr::Vector4f> _channels; ///< Display which channels (shader).

		GLuniform<sibr::Vector2f> _pos; ///< Center position.
		GLuniform<sibr::Vector2f> _size; ///< Image size.
		GLuniform<float> _scale; ///< Image scale.
		GLuniform<bool> _correctRatio; ///< Use proper aspect ratio to display.
		
		sibr::Vector3f _bgColor; ///< Background color.
		bool _showGUI = true; ///< Show the GUI be displayed or not.
		bool _allowInteraction = true; ///< Should the user be able to pan/zoom into the image.
	};

}