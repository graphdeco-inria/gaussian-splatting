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

#include "../Config.hpp"
#include <core/graphics/Texture.hpp>
#include <core/graphics/Input.hpp>
#include <core/graphics/Window.hpp>
#include <core/assets/InputCamera.hpp>
#include <map>
#include "InterfaceUtils.h"
#include "MeshViewer.h"
#include <core/view/ViewBase.hpp>

#include <imgui/imgui.h>

//typedef void (*CallBackFunction)(int event, int x, int y, int flags, void* userdata);

namespace sibr {
	struct PixPos {
		PixPos() : im(-1), isDefined(false) {}
		PixPos(int i, const sibr::Vector2i & px) : im(i), pos(px), isDefined(true) {}
		void cout() const { std::cout << im << " : " << pos.transpose() << std::endl; }
		sibr::Vector2i pos;
		int im;
		bool isDefined;
	};

	struct SubView {
		SubView() : isActive(false) {}

		sibr::Viewport viewport;
		bool isActive;
		sibr::Vector2i getViewportPosition(const sibr::Vector2i & winPos) {
			return winPos - sibr::Vector2f(viewport.finalLeft(), viewport.finalTop()).cast<int>();
		}
	};

	struct ScalingOptions {
		ScalingOptions() : numScale(1), interpolation_method_cv(cv::INTER_CUBIC) {};
		int numScale;
		int interpolation_method_cv;
	};

	struct LayerData {
		LayerData(const std::string & name) : name(name) { }
		std::string name;
	};

	struct ScaleData {
		ScaleData(const sibr::Vector2i & imSizeI) {
			imSize = imSizeI.cast<float>();
			imRatio = imSize[0] / (float)imSize[1];
		}
		sibr::Vector2f imSize;
		float imRatio;
	};

	class MultiViewInterface;

	class SIBR_VIEW_EXPORT MultiViewInterfaceView : public sibr::ViewBase {
	
	public:
		enum class ViewType { IMAGES, MESH };

		SIBR_CLASS_PTR(MultiViewInterfaceView);

		MultiViewInterfaceView() {}
		MultiViewInterfaceView(MultiViewInterface * interfacePtr, ViewType type) : interfacePtr(interfacePtr), viewType(type) {}

		virtual void	onRenderIBR(IRenderTarget& /*dst*/, const sibr::Camera& /*eye*/) {}
		
		virtual void	onUpdate(Input& /*input*/, const sibr::Viewport & viewport);
		virtual void	onRender(const sibr::Viewport & viewport);
		
		PixPos currentActivePos;

	protected:
		MultiViewInterface * interfacePtr;
		ViewType viewType;
	};

	/**
		This class provides basic rendering utilities for a list of images + a mesh.
		\ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT MultiViewInterface {

		SIBR_CLASS_PTR(MultiViewInterface);

	public:

		MultiViewInterface();

		/**
		 Execute the main display loop, with an optional callback.
		 \param window the sibr::Window displayed into
		 \param f an optional callback called at the end of each frame.
		*/
		void displayLoop(sibr::Window & window, std::function<void(MultiViewInterface*)> = [](MultiViewInterface*){});
		void addCameras(const std::vector<InputCamera::Ptr> & input_cams);
		void addMesh(const sibr::Mesh::Ptr & mesh);
		void addMesh(const sibr::Mesh & mesh);

		virtual void update(sibr::Window & window, const sibr::Input & input);

		virtual void updateImageView(const sibr::Viewport & viewport, const sibr::Input & input);

		virtual void render();

		void renderImageView(const sibr::Viewport & viewport);

		sibr::ViewBase::Ptr getViewBase(MultiViewInterfaceView::ViewType type = MultiViewInterfaceView::ViewType::IMAGES);
		void onGui();

		//~MultiViewInterface();

		//virtual void loop();

		sibr::Mesh::Ptr cpuMesh;
		sibr::Mesh::Ptr reproMesh;
		sibr::Mesh::RenderMode reproMeshMode;
		bool reproMeshBackFace;

		sibr::Mesh::Ptr highlightedPixelsMesh;
		std::vector<sibr::PixPos> highlightedPixels;
		bool hightligthChanged;

		sibr::Vector2f imgPixelScreenSize;

		SubView imagesView;
		SubView meshView;
		
		sibr::Input imagesInput;
		sibr::Input meshInput;


	public:
		//std::unique_ptr<Window> window;

		MultiViewInterfaceView::Ptr imagesViewBase;
		MultiViewInterfaceView::Ptr meshViewBase;

		InterfaceUtilities utils;

		ScalingOptions scalingOptions;
		std::map<std::string, int> name_to_layer_map;

		RectangleData viewRectangle;
		DragData drag;
		DoubleClick<sibr::Mouse::Left> dclick;
		SelectionData zoomSelection;
		sibr::MeshViewer meshViewer;

		std::vector<InputCamera::Ptr> cams;

		sibr::PixPos currentActivePos;

		sibr::Vector2i grid;
		sibr::Vector2f imSizeF;
		sibr::Vector2f winSize;
		float imRatio;

		int currentScale;
		int currentLayer;
		int numImgs;
		std::vector<std::vector<sibr::ITexture2DArray::Ptr>> imagesLayers; //for each scale, each image layer
		std::vector<std::vector<const sibr::IImage*>> imagesPtr;
		std::vector<std::vector<sibr::IImage::Ptr> > imagesFromLambdasPtr;
		std::vector<LayerData> layersData;
		std::vector<ScaleData> scalesData;

		PixPos pixFromScreenPos(const sibr::Vector2i & pos, const sibr::Vector2f & winSize);

		UV01 screenPos(const PixPos & pix);
		UV01 screenPosPixelCenter(const PixPos & pix);

		sibr::Vector2i screenPosPixels(const PixPos & pix, const sibr::Vector2f & winSize);
		sibr::Vector2f screenPosPixelsFloat(const PixPos & pix, const sibr::Vector2f & winSize);

		void addHighlightPixel(const PixPos & pix, const sibr::Vector2f & winSize);
		void renderHighlightPixels();

		void highlightPixel(const PixPos & pix, const sibr::Viewport & viewport, const sibr::Vector3f & color = { 0, 1, 0 }, const sibr::Vector2f & pixScreenSize = { 5.0f,5.0f } );

		virtual void displayImages(const sibr::Viewport & viewport);
		virtual void displayMesh(const sibr::Viewport & viewport);

		void displayZoom(const sibr::Viewport & viewport);
		void displayHighlightedPixels(const sibr::Vector3f & color, float alpha);

		virtual void updateMeshView(const sibr::Input & input, sibr::Window & window);
		virtual void updateMeshView(const sibr::Input & input, const sibr::Viewport & viewport);
		void updateZoomBox(const sibr::Input & input, const sibr::Vector2f & winSize);
		void updateCurrentLayer(const sibr::Input & input);
		void updateZoomScroll(const sibr::Input & input);
		void updateCenter(const sibr::Input & input, const sibr::Vector2f & winSize);
		void updateDrag(const sibr::Input & input, const sibr::Vector2f & winSize);

		template<typename T_Type, unsigned int T_NumComp>
		bool checkNewLayer(const std::vector<sibr::Image<T_Type, T_NumComp> > & images) 
		{
			if (imagesLayers.size() == 0) {
				imagesLayers.resize(scalingOptions.numScale);
				for (int scale = 0; scale < scalingOptions.numScale; ++scale) {
					int w_s = static_cast<int>(std::ceil(images[0].w()*pow(2.0f, -scale)));
					int h_s = static_cast<int>(std::ceil(images[0].h()*pow(2.0f, -scale)));
					scalesData.push_back(ScaleData(sibr::Vector2i(w_s, h_s)));
				}
				numImgs = (int)images.size();
			}

			const auto & baseScaleImageLayer = imagesLayers[0];
			if (baseScaleImageLayer.size() > 0) {
				if ((uint)images.size() != baseScaleImageLayer[0]->depth()) {
					SIBR_ERR << "not enough images" << std::endl;
				}
			}

			if (images.size() == 0) {
				SIBR_ERR << "empty image vector" << std::endl;
			}

			return true;
		}

	public:

		template <typename T_Type, unsigned int T_NumComp>
		void addImageLayer(const std::vector<sibr::ImagePtr<T_Type, T_NumComp> > & images, const std::string & name = "") {
			std::vector<sibr::Image<T_Type, T_NumComp> > imgs(images.size());
			for (int i = 0; i < (int)images.size(); ++i) {
				imgs[i] = images[i]->clone();
 			}
			addImageLayer(imgs, name);
		}

		template <typename T_Type, unsigned int T_NumComp>
		void addImageLayer(const std::vector<sibr::Image<T_Type, T_NumComp> > & images, const std::string & name = "")
		{
			
			checkNewLayer(images);

			for (int scale = 0; scale < scalingOptions.numScale; ++scale) {
				std::vector<sibr::Image<T_Type, T_NumComp> > resized_imgs(images.size());
				if (scale != 0) {
#pragma omp parallel for
					for (int im = 0; im < (int)images.size(); ++im) {
						const sibr::Vector2i scaleSize = scalesData[scale].imSize.cast<int>();
						resized_imgs[im] = images[im].resized(scaleSize[0], scaleSize[1], scalingOptions.interpolation_method_cv);
					}
				} else {
					std::vector<const sibr::IImage*> layerPtrs(images.size()); 
					for (int im = 0; im < (int)images.size(); ++im) {
						layerPtrs[im] = &images[im];
					}
					imagesPtr.push_back(layerPtrs);
				}
				const std::vector<sibr::Image<T_Type, T_NumComp> > & imgs = (scale == 0 ?  images : resized_imgs);
				auto layer = std::make_shared<sibr::Texture2DArray<T_Type, T_NumComp>>();

				layer->createFromImages(imgs);

				sibr::ITexture2DArray::Ptr layerBase(layer);
				imagesLayers[scale].push_back(layerBase);
			}

			std::string layerName = (name == "" ? "Layer" + std::to_string(layersData.size()) : name);
			name_to_layer_map[layerName] = (int)layersData.size();
			layersData.push_back(LayerData(layerName));
			
		}

		template <typename T_Type, unsigned int T_NumComp, typename LambdaType>
		void addImageLayerWithLambda(const std::vector<sibr::Image<T_Type, T_NumComp> > & images, LambdaType lambda, const std::string & name = "") {
			
			checkNewLayer(images);

			using Lambda_Out_Image_Type = decltype(lambda(images[0]));
			using Lambda_Out_Type = typename Lambda_Out_Image_Type::Type;
			const int Lambda_Out_N = Lambda_Out_Image_Type::e_NumComp;

			for (int scale = 0; scale < scalingOptions.numScale; ++scale) {
				std::vector<sibr::Image<T_Type, T_NumComp> > resized_imgs(images.size());
				if (scale != 0) {
#pragma omp parallel for
					for (int im = 0; im < (int)images.size(); ++im) {
						const sibr::Vector2i scaleSize = scalesData[scale].imSize.cast<int>();
						resized_imgs[im] = images[im].resized(scaleSize[0], scaleSize[1], scalingOptions.interpolation_method_cv);
					}
				} 
				const std::vector<sibr::Image<T_Type, T_NumComp> > & imgs = (scale == 0 ? images : resized_imgs);

				std::vector<Lambda_Out_Image_Type> lambdaImgs(images.size());
#pragma omp parallel for
				for (int im = 0; im < (int)images.size(); ++im) {
					lambdaImgs[im] = lambda(imgs[im]);
					//sibr::show(lambdaImgs[im], "test");
				}
				
				if (scale == 0) {
					std::vector<sibr::IImage::Ptr> firstLayerPtrs(images.size());
					std::vector<const sibr::IImage*> layerPtrs(images.size());
					for (int im = 0; im < (int)images.size(); ++im) {
						auto lambdaImgPtr = std::make_shared<Lambda_Out_Image_Type>(lambdaImgs[im].clone());

						firstLayerPtrs[im] = std::static_pointer_cast<IImage>(lambdaImgPtr);
						layerPtrs[im] = firstLayerPtrs[im].get();
					}
					imagesFromLambdasPtr.push_back(firstLayerPtrs);
					imagesPtr.push_back(layerPtrs);
				}

				typename sibr::Texture2DArray<Lambda_Out_Type, Lambda_Out_N>::Ptr layer = std::make_shared<sibr::Texture2DArray<Lambda_Out_Type, Lambda_Out_N> >();
				layer->createFromImages(lambdaImgs);
				sibr::ITexture2DArray::Ptr layerBase(layer);
				imagesLayers[scale].push_back(layerBase);
			}

			std::string layerName = (name == "" ? "Layer" + std::to_string(layersData.size()) : name);
			name_to_layer_map[layerName] = (int)layersData.size();
			layersData.push_back(LayerData(layerName));
		}
	};


} //namespace sibr
