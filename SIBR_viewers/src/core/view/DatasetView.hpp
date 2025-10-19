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

#include "MultiViewManager.hpp"
#include "SceneDebugView.hpp"
#include "ImagesGrid.hpp"
#include "core/scene/BasicIBRScene.hpp"

namespace sibr {

	/** Visualize and explore a MVS dataset. 
	Allow reprojections between one of the input images, scene geometry and other images.
	 \ingroup sibr_view
	 */
	class SIBR_VIEW_EXPORT DatasetView 
		: public MultiViewBase
	{
		SIBR_CLASS_PTR(DatasetView);

	public:

		/** Constructor.
		 * \param scene the IBR scene
		 * \param defaultRenderingRes the mesh view rendering resolution
		 * \param defaultViewRes the window/view resolution
		 */
		DatasetView(const BasicIBRScene & scene, const Vector2u & defaultRenderingRes = { 0,0 }, const Vector2i & defaultViewRes = { 800, 600 });

		/** Reprojection mode. */
		enum ReprojectionMode { NONE, IMAGE_TO_IMAGE, MESH_TO_IMAGE };

		/** Update the GUI. */
		virtual void	onGui(Window& win) override;

		/** Update state based on user input.
		 *\param input the view input
		 */
		virtual void	onUpdate(Input& input) override;

		/** Perform rendering.
		 *\param win the destination window
		 **/
		virtual void	onRender(Window& win) override;

	protected:

		/** Contain data related to the reprojection of a point in input images. */
		struct ReprojectionData {

			/** \return true if point is active */
			operator bool() const { return active; }

			std::vector<MVpixel> repros; ///< Store reprojected pixel positions.
			MVpixel image_input; ///< Initial selected position.

			Vector3f point3D; ///< World space point.
			bool occlusionTest = true; ///< Should occlusion test be applied.
			bool active = false; ///< Is the point active.
		};

		/** populate reprojection information.
		\param data the info to populate
		*/
		void repro(ReprojectionData & data);
		
		/** Visualize the reprojection information.
		\param data the reprojection to display
		*/
		void displayRepro(const ReprojectionData & data);

		/** \return the mesh subview. */
		BasicSubView & getMeshView();

		/** \return the images subview. */
		BasicSubView & getGridView();

		/** \return the mesh display manager. */
		MultiMeshManager::Ptr getMMM();

		/** \return the image grid manager. */
		ImagesGrid::Ptr getGrid();

		/** \return the mesh display data. */
		MeshData & proxyData();

		std::vector<RaycastingCamera> cams; ///< Input cameras.
		ReprojectionData currentRepro; ///< Current selected reprojection.
		ReprojectionMode reproMode = MESH_TO_IMAGE; ///< Current reprojection mode.

		const std::string meshSubViewStr = "dataset view - mesh";
		const std::string gridSubViewStr = "grid";

		
	};
}