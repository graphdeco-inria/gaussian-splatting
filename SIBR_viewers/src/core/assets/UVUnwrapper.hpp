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

#include <core/assets/Config.hpp>
#include <core/graphics/Mesh.hpp>

namespace xatlas {
	struct Atlas;
}

namespace sibr
{
	/** Unwraps a mesh onto a plane, generating texture coordinates for each vertex.
	 * Internaly relies on xatlas for unwrapping.
	\ingroup sibr_assets
	*/
	class SIBR_ASSETS_EXPORT UVUnwrapper {
	public:

		/** Constructor.
		 *\param mesh the mesh to unwrap, if UVs are already present they will be used as a guide
		 *\param res the target texture width, will determine UV accuracy
		 */
		UVUnwrapper(const sibr::Mesh& mesh, unsigned int res);

		/** Unwrap the mesh, return a copy with UV coordinates. Note that some vertices might be duplicated if they are assigned different UVs in two faces.
		 * \return the unwrapped mesh
		 */
		sibr::Mesh::Ptr unwrap();

		/** For each vertex of the unwrapped mesh, the mapping give the index of the corresponding vertex in the input mesh.
		 * \return a reference to the mapping vector
		 */
		const std::vector<uint> & mapping() const;
		
		/** Generate debug visualization by rasterizing the meshes in texture space.
		 * \return a set of images, one per atlas
		 */
		std::vector<ImageRGB::Ptr> atlasVisualization() const;

		/// Destructor.
		~UVUnwrapper();
		
	private:
		
		const sibr::Mesh& _mesh; ///< Unwrapped mesh.
		unsigned int _size; ///< Width of the atlas, detemrine the accuracy of the estimated UVs.
		xatlas::Atlas* _atlas; ///< Atlas object.
		std::vector<uint> _mapping; ///< Mapping from the new vertices to the old (some might be duplicated with different UV values).
		
	};
}
