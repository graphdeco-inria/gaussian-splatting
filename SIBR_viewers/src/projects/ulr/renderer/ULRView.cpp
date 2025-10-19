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


/// \todo TODO: make shorter
#include "Config.hpp"
#include <core/assets/Resources.hpp>
#include <projects/ulr/renderer/ULRView.hpp>
#include <core/system/Vector.hpp>
#include <core/graphics/Texture.hpp>
#include <core/view/ViewBase.hpp>
#include <map>

namespace sibr { 

ULRView::~ULRView( )
{
	_altMesh.reset();
}

ULRView::ULRView( const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h ) :
	_scene(ibrScene),
		sibr::ViewBase(render_w, render_h)
{
	_altMesh.reset();
	_altMesh = nullptr;
	_numDistUlr = 10, _numAnglUlr = 14;
    std::cerr << "\n[ULRenderer] setting number of images to blend "<< _numDistUlr << " " << _numAnglUlr << std::endl;

	_ulr.reset(new ULRRenderer(render_w, render_h));
	
	_inputRTs = ibrScene->renderTargets()->inputImagesRT();
}

void ULRView::onRenderIBR( sibr::IRenderTarget& dst, const sibr::Camera& eye ) {
    // Select subset of input images for ULR
	std::vector<uint> imgs_ulr = chosen_cameras(eye);
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
	_ulr->process(
		/* input -- images chosen */ imgs_ulr, 
		/* input -- camera position */ eye, 
		/* input -- scene */ _scene, 
		/* input -- alt mesh if available */ _altMesh, 
		/* input -- input RTs -- can be RGB or alpha */  _inputRTs,
		/* output */ dst);
}

// -----------------------------------------------------------------------
/// \todo Select a subset from imput images speed up URL
/// \todo TODO: This function needs serious cleanup
//
std::vector<uint> ULRView::chosen_cameras(const sibr::Camera& eye) {
    std::vector<uint> imgs_id;
    std::multimap<float,uint> distMap;									// distance wise closest input cameras
	std::multimap<float,uint> dang;									// angular distance from inputs to novel camera
    for (uint i=0; i<_scene->cameras()->inputCameras().size(); i++ ) {
        const sibr::InputCamera& inputCam = *_scene->cameras()->inputCameras()[i];
        if (inputCam.isActive()) {
			// Convert following to Eigen versions
            float dist = sibr::distance(inputCam.position(), eye.position());
            float angle = sibr::dot(inputCam.dir(),eye.dir());
 //           if (angle > 0.707) {									// cameras with 45 degrees
                distMap.insert(std::make_pair(dist,i));					// sort distances in increasing order
				dang.insert(std::make_pair( acos(angle),i));				// sort angles in increasing order
//			}
        }
    }

   // HACK GD -- should really look at camera angles as well and sort them
////   bool not_enough = false;
	// if you have < 2 cameras, choose the (NUM_DIST+NUM_ANGL)/2 closest ones
////   if( dang.size() + distMap.size() < 2 )
////		not_enough = true;
	for (uint i=0; i< _scene->cameras()->inputCameras().size(); i++) {
        const sibr::InputCamera& inputCam = *_scene->cameras()->inputCameras()[i];
        if (inputCam.isActive() && distMap.size() <= (_numDistUlr+_numAnglUlr)/2 ) {
            float dist = sibr::distance(inputCam.position(),eye.position());
            distMap.insert(std::make_pair(dist,i));					// sort distances in increasing order
			}
	}

    std::multimap<float,uint>::const_iterator d_it(distMap.begin());	// select the _numDistUlr closest cameras
	for (int i=0; d_it!=distMap.end() && i<_numDistUlr; d_it++,i++) {
        imgs_id.push_back(d_it->second);
    }

	std::multimap<float,uint>::const_iterator a_it(dang.begin());    // select the NUM_ANG_ULR closest cameras
	for (int i=0; a_it!=dang.end() && i<_numAnglUlr; a_it++,i++) {
        imgs_id.push_back(a_it->second);
    }

	std::sort( imgs_id.begin(), imgs_id.end() );				// Avoid repetitions
	imgs_id.erase( std::unique( imgs_id.begin(), imgs_id.end() ), imgs_id.end() );

	SIBR_ASSERT(imgs_id.size() <= _numDistUlr + _numAnglUlr);
    return imgs_id;
}

void ULRView::setMasks( const std::vector<RenderTargetLum::Ptr>& masks ) {
		_ulr->setMasks(masks);
}

} /*namespace sibr*/ 
