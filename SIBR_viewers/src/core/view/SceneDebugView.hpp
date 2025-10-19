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

# include "core/assets/InputCamera.hpp"
# include "core/assets/CameraRecorder.hpp"
# include "core/graphics/Texture.hpp"
# include "core/graphics/Camera.hpp"
# include "core/graphics/Window.hpp"
# include "core/graphics/Shader.hpp"
# include "core/graphics/Mesh.hpp"
# include "core/view/InteractiveCameraHandler.hpp"
# include "core/view/ViewBase.hpp"
# include "core/scene/BasicIBRScene.hpp"
# include "core/system/CommandLineArgs.hpp"

#include <core/view/MultiMeshManager.hpp>

namespace sibr
{

	/** Generate an accurate camera frustum
		\param cam camera to visualize as a stub
		\param znear near value to use for the frustum (if < 0, cam.near() will be used)
		\param zfar far value to use for the frustum (if < 0, cam.far() will be used)
		 \ingroup sibr_view
		*/
	Mesh::Ptr SIBR_VIEW_EXPORT generateCamFrustum(const InputCamera & cam, float znear = -1, float zfar = -1, bool useCam = true);

	/** Generate an accurate camera frustum with a custom color.
		\param cam camera to visualize as a stub
		\param col the mesh line color
		\param znear near value to use for the frustum (if < 0, cam.near() will be used)
		\param zfar far value to use for the frustum (if < 0, cam.far() will be used)
		 \ingroup sibr_view
		*/
	Mesh::Ptr SIBR_VIEW_EXPORT generateCamFrustumColored(const InputCamera & cam, const Vector3f & col, float znear = -1, float zfar = -1);

	/** Generate a quad representing a camera image plane.
	 *\param cam the camera
	 *\param dist the distance in world space from the camera position to the image plane
	  \ingroup sibr_view
	 **/
	Mesh::Ptr SIBR_VIEW_EXPORT generateCamQuadWithUvs(const InputCamera & cam, float dist);

	/** Helper used to display camera labels on screen.
	 * Internally use ImGui to generate labels data.
	  \ingroup sibr_view
	 * */
	struct SIBR_VIEW_EXPORT LabelsManager {

	protected:

		/** Displayed cameras info. */
		struct CameraInfos {
			/** Constructor.
			 *\param cam the camera
			 *\param id the corresponding vector ID
			 *\param highlight should the camera be highlighted. 
			 */
			CameraInfos(const InputCamera& cam, uint id, bool highlight);

			const InputCamera & cam; ///< Camera.
			uint id = 0; ///< Array ID.
			bool highlight = false; ///< Highlight status.
		};

		/** Initialize the shaders. */
		void setupLabelsManagerShader();

		/** Generate labels data based on input camera informations.
		 *\param cams the cameras 
		 */
		void setupLabelsManagerMeshes(const std::vector<InputCamera::Ptr> & cams);

		/** Render the camera labels.
		 *\param eye the current viewpoint
		 *\param vp the view viewport
		 *\param cams_info the current state of the cameras. 
		 * \todo Get rid of the viewport if possible.
		 **/
		void renderLabels(const Camera & eye, const Viewport & vp, const std::vector<CameraInfos> & cams_info);
	

		/** Label geometry info. The mesh is split in two parts, 
		 * one containing the background label shape, 
		 * and one containing the quads that support the text. */
		struct LabelMesh {
			Mesh::Ptr mesh; ///< The generated mesh.
			unsigned int splitIndex = 0; ///< The boundary between foreground and background mesh.
		};

		std::map<unsigned int, LabelMesh> 	_labelMeshes; ///< Generated geometry for each label.
		GLShader							_labelShader; ///< Shader.
		GLuniform<Vector3f>					_labelShaderPosition; ///< Uniform for the label position.
		GLuniform<float>					_labelShaderScale = 1.0f; ///< Uniform for the label scale (used twice per label, with different values derived from _labelScale).
		GLuniform<Vector2f>					_labelShaderViewport; ///< The viewport of the view, for ratio adjustment.
		float								_labelScale = 1.0f; ///< The label scale ons creen.

	};

	/** Helper used to render image planes in front of the camera, 
	 * for both scenes storing 2D separate images or a texture array.
	  \ingroup sibr_view
	 */
	struct SIBR_VIEW_EXPORT ImageCamViewer {

	protected:

		/** Initialize the shaders. */
		void initImageCamShaders();

		/** Render one specific input image on a camera image plane.
		 *\param eye the current viewpoint
		 *\param cam the camera to show the image plane of
		 *\param rts input 2D textures list
		 *\param cam_id the list index associated to the camera
		 */
		void renderImage(const Camera & eye, const InputCamera & cam, const std::vector<RenderTargetRGBA32F::Ptr> & rts, int cam_id);

		/** Render one specific input image on a camera image plane.
		 *\param eye the current viewpoint
		 *\param cam the camera to show the image plane of
		 *\param tex2Darray_handle input images texture array
		 *\param cam_id the array slice associated to the camera
		 */
		void renderImage(const Camera & eye, const InputCamera & cam, uint tex2Darray_handle, int cam_id);

		void renderImage(const Camera& eye, const InputCamera& cam, const RenderTargetRGBA32F::Ptr& rt);

		void updateImgRt(const InputCamera& cam, const sibr::ImageRGBA& img);

		GLShader _shader2D;		///< Shader for the 2D separate case.
		GLShader _shaderArray;  ///< Shader for the texture array case.
		GLuniform<sibr::Matrix4f>	_mvp2D, _mvpArray; ///< MVP matrix.
		GLuniform<float>			_alpha2D = 1.0f; ///< Opacity.
		GLuniform<float>			_alphaArray = 1.0f; ///< Opacity.
		GLuniform<int>				_sliceArray = 1; ///< Slice location (for the texture array case).
		float						_alphaImage = 0.5f; ///< Opacity shared value.

		RenderTargetRGBA32F::Ptr			_imgRt;
		float								_userCameraScaling = 3.f; ///< User camera scaling.
		float								_pathScaling = 0.3f; ///< Input cameras scaling.
		float								_lastPathScaling = 0.2f;
		std::string							_imgToFetch = "";
		uint								_imgTexHandle;
		bool								_displayImg = false;
		sibr::Texture2D<unsigned char, 4>* _imgTex = nullptr;
	};

	/** Scene viewer for IBR scenes with a proxy, cameras and input images. 
	 * It adds camera visualization options (labels, frusta, image planes) on top of the MeshManager.
	  \ingroup sibr_view
	 */
	class SIBR_VIEW_EXPORT SceneDebugView : public MultiMeshManager, public ImageCamViewer, public LabelsManager
	{
		SIBR_CLASS_PTR(SceneDebugView);

	public:

		/** Which camera info should be displayed in the GUI. */
		enum CameraInfoDisplay { SIZE, FOCAL, FOV_Y, ASPECT };

		/** Constructor.
		 * \param scene the scene to display
		 * \param camHandler a camera handler to display as a "user camera"
		 * \param myArgs dataset arguments (needed to load/save the camera location)
		 */
		SceneDebugView(const IIBRScene::Ptr& scene, const InteractiveCameraHandler::Ptr & camHandler, const BasicDatasetArgs& myArgs, const std::string& imagesPath = "");

		/** Constructor.
		 * \param scene the scene to display
		 * \param viewport the view viewport
		 * \param camHandler a camera handler to display as a "user camera"
		 * \param myArgs dataset arguments (needed to load/save the camera location)
		 * \warning Deprecated, use the version without the viewport.
		 */
		SceneDebugView(const IIBRScene::Ptr& scene, const Viewport& viewport, const InteractiveCameraHandler::Ptr& camHandler, const BasicDatasetArgs& myArgs);

		/** Update state based on user input.
		 * \param input user input
		 * \param deltaTime the time elapsed since last update
		 * \param viewport input viewport
		 * \note Used when the view is in a multi-view system.
		 */
		virtual void onUpdate(Input & input, const float deltaTime, const Viewport & viewport = Viewport(0.0f, 0.0f, 0.0f, 0.0f));

		/** Update state based on user input.
		 * \param input user input
		 * \param viewport input viewport
		 * \note Used when the view is in a multi-view system.
		 */
		virtual void onUpdate(Input & input, const Viewport & viewport) override;

		/* Update state based on user input.
		 * \param input user input
		 */
		virtual void onUpdate(Input& input) override;
		
		/** Render content in a window.
		 *\param win destination window
		 */
		virtual void onRender(Window& win) override;

		/** Render content in the currently bound RT, using a specific viewport.
		 * \param viewport destination viewport
		 * \note Used when the view is in a multi-view system.
		 */
		virtual void onRender(const Viewport & viewport) override;

		using MultiMeshManager::onRender;

		/** Update and display GUI panels. */
		virtual void onGUI() override;

		/** Save the top view camera to scene/cameras/topview.txt. */
		void save();

		/** \return the camera handler for the view. */
		const InteractiveCameraHandler & getCamera() const { return camera_handler; }

		/** \return the camera handler for the view. */
		InteractiveCameraHandler & getCamera() { return camera_handler; }

		/** Replace the scene.
		 *\param scene the new scene
		 *\param preserveCamera should the current camera position be preserved
		 **/
		void setScene(const IIBRScene::Ptr & scene, bool preserveCamera = false);

		/** Update the active status of all cameras
		 *\param cams_id the active camera IDs. 
		 */
		void updateActiveCams(const std::vector<uint> & cams_id);

	protected:

		/** Generate the GUI for the display options. */
		void gui_options();

		/** generate the GUI with the camera infos. */
		void gui_cameras();

		/** Setup the view. */
		void setup();

		/** Setup the geometry. */
		void setupMeshes();

		InteractiveCameraHandler::Ptr	_userCurrentCam; ///< The "main view" camera handler (will be displayed as an extra camera).
		IIBRScene::Ptr					_scene; ///< Current displayed scene.
		std::vector<CameraInfos>		_cameras; ///< Additional scene cameras info.
		CameraInfoDisplay				_camInfoOption = SIZE; ///< Camera info to display in the GUI.
		std::string						_topViewPath; ///< Path to the topview saved file.
		int								_snapToImage = 0; ///< ID of the camera to snap to.
		int								_cameraIdInfoGUI = 0; ///< ID of the camera to display info about.
		bool							_showImages = true; ///< Show the image planes.
		bool							_showLabels = false; ///< Show camera labels.

		std::string						_images_path;
		int								_renderingCam; ///< ID of the camera used for rendering
		Mesh::Ptr						_used_cams = std::make_shared<Mesh>();
		Mesh::Ptr						_non_used_cams = std::make_shared<Mesh>();
		Mesh::Ptr						_user_cam = std::make_shared<Mesh>();
	};

} // namespace
