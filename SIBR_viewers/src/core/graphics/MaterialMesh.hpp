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
# include <map>
# include <sstream>

# include "core/graphics/Config.hpp"
# include "core/system/Vector.hpp"
# include "core/graphics/Image.hpp"
# include "core/graphics/Mesh.hpp"
# include "core/graphics/MeshBufferGL.hpp"
# include "core/graphics/Texture.hpp"

namespace sibr
{

	
	/** Store both CPU and GPU data for a geometric mesh.
		Specifically designed for synthetic scenes with material information.
		Provide many processing and display methods.
	\ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT MaterialMesh : public sibr::Mesh
	{
	public:
		typedef std::vector<int>								MatIds;
		typedef std::vector<int>								MeshIds;
		typedef std::vector<std::string>						MatId2Name;

		typedef std::map<std::string, sibr::ImageRGB::Ptr>		OpacityMaps;
		typedef std::map<std::string, sibr::ImageRGBA::Ptr>		DiffuseMaps;

		typedef sibr::ImageRGB::Ptr								TagsMap;
		typedef std::map<std::string, sibr::ImageRGB::Ptr>		TagsCoveringMaps;

		typedef std::vector<Mesh>								SubMeshes;
		typedef std::vector<sibr::ImageRGBA>					AlbedoTextures;

		typedef std::map<std::string, bool>						SwitchTagsProperty;

		SIBR_CLASS_PTR(MaterialMesh);

		/** Synthetic data rendering options. */
		enum class RenderCategory
		{
			classic,
			diffuseMaterials,
			threesixtyMaterials,
			threesixtyDepth
		};

		/** Ambient occlusion options. */
		struct AmbientOcclusion {
			bool AoIsActive = false;
			float AttenuationDistance = 1.f;
			float IlluminanceCoefficient = 1.f;
			float SubdivideThreshold = 10.f;
		};
		typedef struct AmbientOcclusion AmbientOcclusion;

		std::string vertexShaderAlbedo =
			"#version 450										\n"
			"layout(location = 0) in vec3 in_vertex;			\n"
			"layout(location = 1) in vec3 in_colors;			\n"
			"layout(location = 2) in vec2 in_uvCoords;			\n"
			"layout(location = 3) in vec3 in_normal;			\n"
			"layout(location = 4) in float in_ao;			\n"
			"//layout(location = 4) in float in_material;			\n"
			"layout (location = 2) out vec2 uvCoords;			\n"
			"//out float material;			\n"
			"layout (location = 3) out vec3 normal;									\n"
			"out float ao ;									\n"
			"out vec3 pos_vertex;									\n"
			"layout (location = 1) out vec3 colors;									\n"
			"uniform mat4 MVP;									\n"
			"uniform bool lightIsPresent;									\n"
			"uniform vec3 lightPos;									\n"
			"void main(void) {									\n"
			"	normal = in_normal;		\n"
			"	ao = in_ao;		\n"
			"	uvCoords = in_uvCoords;		\n"
			"	colors= in_colors;		\n"
			"	pos_vertex= in_vertex;		\n"
			"	//material= float(in_material);		\n"
			"	gl_Position = MVP*vec4(in_vertex,1) ;		\n"
			"}													\n";

		std::string fragmentShaderAlbedo =
			"#version 450														\n"
			"layout(binding = 0) uniform sampler2D tex;				\n"
			"layout(binding = 2) uniform sampler2D opacity;				\n"
			"uniform int layer;													\n"
			"uniform bool AoIsActive;													\n"
			"uniform vec2 grid;												\n"
			"uniform float IlluminanceCoefficient;												\n"
			"uniform bool lightIsPresent;									\n"
			"uniform float scaleTags;													\n"
			"uniform float intensityLight;									\n"
			"uniform vec3 lightPos;									\n"
			"layout (location = 2) in vec2 uvCoords;													\n"
			"layout (location = 3) in vec3 normal ;									\n"
			"layout (location = 1) in vec3 colors;									\n"
			"out vec4 out_color;												\n"
			"void main(void) {													\n"
			"	vec4 opacityColor;												\n"
			"	vec3 colorsModified = colors;\n"
			"	float lighter_ao = colors.x * IlluminanceCoefficient; \n"
			"	if (lighter_ao > 1.f ) lighter_ao = 1.f;\n"
			"	colorsModified.x = lighter_ao;\n"
			"	colorsModified.y = lighter_ao;\n"
			"	colorsModified.z = lighter_ao;\n"
			"	opacityColor = texture(opacity,vec2(uvCoords.x,1.0-uvCoords.y));\n"
			"	if (opacityColor.x < 0.1f && opacityColor.y < 0.1f && opacityColor.z < 0.1f ) discard;\n"
			"							\n"
			"	out_color = texture(tex,vec2(uvCoords.x,1.0-uvCoords.y));\n"
			//"	if (out_color.a != 0.f ) discard; \n"
			//"							\n"
			"	if (AoIsActive ) {						\n"
			"	out_color = out_color * vec4(colorsModified,1);\n}"
			"	out_color = vec4(out_color.x,out_color.y,out_color.z,out_color.a);\n"
			"}																	\n";


		std::string fragmentShaderAlbedoTag =
			"#version 450														\n"
			"layout(binding = 0) uniform sampler2D tex;				\n"
			"layout(binding = 1) uniform sampler2D tags;				\n"
			"layout(binding = 2) uniform sampler2D opacity;				\n"
			"uniform int layer;													\n"
			"uniform float scaleTags;													\n"
			"uniform bool AoIsActive;													\n"
			"uniform vec2 grid;												\n"
			"uniform float IlluminanceCoefficient;												\n"
			"uniform bool lightIsPresent;									\n"
			"uniform float intensityLight;									\n"
			"uniform vec3 lightPos;									\n"
			"layout (location = 2) in vec2 uvCoords;													\n"
			"layout (location = 3) in vec3 normal ;									\n"
			"layout (location = 1) in vec3 colors;									\n"
			"out vec4 out_color;												\n"
			"in vec3 pos_vertex;									\n"
			"void main(void) {													\n"
			"	vec4 opacityColor;												\n"
			"	vec3 colorsModified = colors;\n"
			"	float lighter_ao = colors.x * IlluminanceCoefficient; \n"
			"	if (lighter_ao >= 1.f ) lighter_ao = 1.f;\n"
			"	colorsModified.x = lighter_ao;\n"
			"	colorsModified.y = lighter_ao;\n"
			"	colorsModified.z = lighter_ao;\n"
			"	opacityColor = texture(opacity,vec2(uvCoords.x,1.0-uvCoords.y));\n"
			"	if (opacityColor.x < 0.1f || opacityColor.y < 0.1f || opacityColor.z < 0.1f ) discard;\n"
			"							\n"
			"							\n"
			"	out_color = texture(tex,vec2(uvCoords.x,1.0-uvCoords.y));\n"
			"	if (out_color.a < 0.1f ) discard; \n"
			//"							\n"
			"	out_color = texture(tags,vec2((uvCoords.x)*scaleTags,(1.0-(uvCoords.y))*scaleTags));\n"
			"							\n"
			"	if (out_color.x == 1.f && out_color.y == 1.f && out_color.z == 1.f)		\n"
			"	out_color = texture(tex,vec2(uvCoords.x,1.0-uvCoords.y));\n"
			"							\n"
			"							\n"
			"	float coeffLight = 1.f;						\n"
			"	if( lightIsPresent) {						\n"
			"				vec3 vertexToLight = normalize( lightPos - pos_vertex );\n"
			"				coeffLight = abs(intensityLight*dot( vertexToLight, normal )) ; \n"
			//"				coeffLight = max(0.0,powerLight* dot( vertexToLight, normal )) ; \n"
			"				coeffLight = 0.50+coeffLight/2.0 ; \n"
			"							\n"
			"							\n"
			"							\n"
			"	}						\n"
			"							\n"
			"	if (AoIsActive ) {						\n"
			"	out_color = out_color * vec4(colorsModified,1);\n}"
			"	out_color = out_color * vec4(coeffLight,coeffLight,coeffLight,1);\n"
			"	out_color = vec4(out_color.x,out_color.y,out_color.z,out_color.a);\n"
			//"	if (out_color.x < 0.01f && out_color.y < 0.01f && out_color.z < 0.01f) discard;		\n"
			"}																	\n";

	public:

		/** Constructor.
		\param withGraphics init associated OpenGL buffers object (requires an openGL context)
		*/
		MaterialMesh(bool withGraphics = true) : Mesh(withGraphics) {
		}

		/** Constructor from a basic mesh.
		\param mesh the mesh to copy
		*/
		MaterialMesh(sibr::Mesh& mesh) : Mesh(mesh) {}

		/** Set material IDs (per triangle)
		\param matIds the new ids
		*/
		inline void	matIds(const MatIds& matIds);
		
		/** \return a reference to the per-triangle material IDs. */
		inline const MatIds& matIds(void) const;
		
		/** \return a reference to the per-vertex material IDs. */
		inline const MatIds& matIdsVertices(void) const;

		/** \return true if each triangle has a material ID assigned. */
		inline bool	hasMatIds(void) const;

		/** \return the mapping between IDs and material names. */
		inline const MatId2Name& matId2Name(void) const;
		
		/** Set the mapping between IDs and material names.
		\param matId2Name the new mapping
		*/
		inline void matId2Name(const MatId2Name& matId2Name);

		/** Set the mesh ID of each vertex.
		\param meshIds the new ids
		*/
		inline void meshIds(const MeshIds& meshIds);
		
		/** \return a reference to the per-vertex mesh IDs. */
		inline const MeshIds& meshIds(void) const;
		
		/** \return true if source mesh information is available for each vertex. */
		inline bool hasMeshIds(void) const;

		/** Query a material opacity map.
		\param matName the material name
		\return the opacity texture if it exist
		*/
		inline sibr::ImageRGB::Ptr opacityMap(const std::string& matName) const;
		
		/** Set all material opacity maps.
		\param maps the new maps
		*/
		inline void opacityMaps(const OpacityMaps & maps);
		
		/** \return all opacity maps. */
		inline const OpacityMaps& opacityMaps(void) const;

		/// Set the switchTag 
		inline void switchTag(const SwitchTagsProperty& switchTag);
		/// get the switchTag 
		inline const SwitchTagsProperty& switchTag(void) const;

		/// Return the pointer to oppacity texture if it exist
		/** Query a material diffuse map.
		\param matName the material name
		\return the diffuse texture if it exist
		*/
		inline sibr::ImageRGBA::Ptr diffuseMap(const std::string& matName) const;
		
		/** Set all material diffuse maps.
		\param maps the new maps
		*/
		inline void diffuseMaps(const DiffuseMaps & maps);
		
		/** \return all diffuse maps. */
		inline const DiffuseMaps& diffuseMaps(void) const;
		
		/** Indicate if the mesh has an associated tag file (for calibration).
		\param hasOrNot the flag
		*/
		inline void hasTagsFile(bool hasOrNot);
		
		/** \return true if the mesh has an associated tag file. */
		inline const bool hasTagsFile(void) const;
		
		/** Set the tag map.
		\param map the new map
		*/
		inline void tagsMap(const TagsMap & map);
		
		/** \return the current tag map. */
		inline const TagsMap& tagsMap(void) const;
		
		/** Indicate if the mesh has an associated covering tag file (for calibration).
		\param hasOrNot the flag
		*/
		inline void hasTagsCoveringFile(bool hasOrNot);
		
		/** \return true if the mesh has an associated covering tag file. */
		inline const bool hasTagsCoveringFile(void) const;
		
		/** Set the covering tag map.
		\param map the new map
		*/
		inline void tagsCoveringMaps(const TagsCoveringMaps & map);
		
		/** \return the current covering tag map. */
		inline const TagsCoveringMaps& tagsCoveringMaps(void) const;
		/// Return the pointer to oppacity texture if it exist
		inline sibr::ImageRGB::Ptr tagsCoveringMap(const std::string& matName) const;
		
		/** Set the sub meshes.
		\param subMeshes a list of submeshes
		*/
		inline void subMeshes(const SubMeshes& subMeshes);
		
		/** \return the list of submeshes. */
		inline const SubMeshes& subMeshes(void) const;

		/** Set the synthetic rendering mode.
		\param type the new mode
		*/
		inline void typeOfRender(const RenderCategory& type);
		
		/** \return the current synthetic rendering mode. */ 
		inline const RenderCategory& typeOfRender(void) const;

		/** Set the ambient occlusion options and compute AO values, storing them in the vertex colors.
		\param ao the new options
		*/
		void ambientOcclusion(const AmbientOcclusion& ao);

		/** \return the current ambient occlusion options. */
		inline const AmbientOcclusion& ambientOcclusion(void);

		/** Set the function used to compute ambient occlusion at each vertex. 
		\param aoFunction the new function to use 
		*/
		inline void aoFunction(std::function<sibr::Mesh::Colors(
			sibr::MaterialMesh&,
			const int)>& aoFunction);

		/** Load a mesh from the disk.
		\param filename the file path
		\return a success flag
		\note Supports OBJ and PLY for now.
		*/
		bool	load(const std::string& filename);

		/** Load a scene from a set of mitsuba XML scene files (referencing multiple OBJs/PLYs). 
		It handles instances (duplicating the geometry and applying the per-instance transformation).
		\param xmlFile the file path
		\param loadTextures should the material textures be loaded
		\return a success flag
		*/
		bool	loadMtsXML(const std::string& xmlFile, bool loadTextures = true);

		/*
		Load tags image files from a list of file paths.
		\param listFilesTags a list of image paths
		*/
		void	loadCoveringTagsTexture(const std::vector<std::string>& listFilesTags);

		/** Attribute a random color at each vertex based on the material IDs of the faces it belongs to. */
		void	fillColorsWithIndexMaterials();

		/** Store the material ID of each vertex in its color attribute (R: bits 0-7, G: 8-15, B: 16-23). */
		void	fillColorsWithMatIds();

		/** Merge another mesh into this one.
		\param other the mesh to merge
		\sa makeWhole
		*/
		void	merge(const MaterialMesh& other);

		/** Make the mesh whole, ie it will have default values for all components (texture, materials, colors, etc)
		  It is useful when merging two meshes. If the second one is missing some attributes, the merging will break the mesh state if it isn't made whole.
		*/
		void	makeWhole(void);

		/** Split the mesh geometry in multiple submeshes based on each vertex material ID. */
		void	createSubMeshes(void);

		/** \return a copy of the mesh with "doubled" faces (obtained by merging the current mesh with a copy with inverted faces. */
		sibr::MaterialMesh::Ptr invertedFacesMesh2() const;

		/** Force upload of data to the GPU. */
		void	forceBufferGLUpdate(void) const;
		
		/** Delete GPU mesh data. */
		void	freeBufferGLUpdate(void) const;

		/** Subdivide a mesh triangles until a triangle area threshold is reached.
		\param threshold the maximum deviation from the average triangle area allowed
		*/
		void subdivideMesh2(float threshold);

		/** Subdivide a mesh triangles until an edge length threshold is reached.
		\param threshold the maximum deviation from the average edge length allowed
		*/
		void	subdivideMesh(float threshold);

		/** Add an environment sphere to the mesh, surrounding the existing geometry.
		\param forcedCenterX optional sphere center x coordinate
		\param forcedCenterY optional sphere center y coordinate
		\param forcedCenterZ optional sphere center z coordinate
		\param forcedRadius optional sphere radius
		*/
		void addEnvironmentMap(float* forcedCenterX = nullptr,
			float* forcedCenterY = nullptr,
			float* forcedCenterZ = nullptr,
			float* forcedRadius = nullptr);

		/** Render the geometry using OpenGL.
		\param depthTest should depth testing be performed
		\param backFaceCulling should culling be performed
		\param mode the primitives rendering mode
		\param frontFaceCulling should the culling test be flipped
		\param invertDepthTest should the depth test be flipped (GL_GREATER_THAN)
		\param tessellation should the rendering call tesselation shaders
		\param adjacency should we get adjacent triangles info in geometry shader
		*/
		void	render(
			bool depthTest = true,
			bool backFaceCulling = true,
			RenderMode mode = FillRenderMode,
			bool frontFaceCulling = false,
			bool invertDepthTest = false,
			bool tessellation = false,
			bool adjacency = false
		) const;

		/** Render the geometry with albedo and tag textures.
		\param depthTest should depth testing be performed
		\param backFaceCulling should culling be performed
		\param mode the primitives rendering mode
		\param frontFaceCulling should the culling test be flipped
		\param invertDepthTest should the depth test be flipped (GL_GREATER_THAN)
		\param specificMaterial should we use a specific material
		\param nameOfSpecificMaterial name of the specific material
		*/
		void	renderAlbedo(
			bool depthTest = true,
			bool backFaceCulling = true,
			RenderMode mode = FillRenderMode,
			bool frontFaceCulling = false,
			bool invertDepthTest = false,
			bool specificMaterial = false,
			std::string nameOfSpecificMaterial = ""
		) const;

		/** Render the geometry for 360 environment maps.
		\param depthTest should depth testing be performed
		\param backFaceCulling should culling be performed
		\param mode the primitives rendering mode
		\param frontFaceCulling should the culling test be flipped
		\param invertDepthTest should the depth test be flipped (GL_GREATER_THAN)
		*/
		void	renderThreeSixty(
			bool depthTest,
			bool backFaceCulling,
			RenderMode mode,
			bool frontFaceCulling,
			bool invertDepthTest
		) const;

		/** Upload the material textures to the GPU. */
		void	initAlbedoTextures(void);

		/** Generate a mesh containing all triangles with a given material.
		\param material the material ID
		\return the submesh
		*/
		Mesh generateSubMaterialMesh(int material) const;

	private:


		MatIds		_matIds; ///< Per triangle material ID.
		MatIds		_matIdsVertices; ///< Per vertex material ID.
		MatId2Name	_matId2Name; ///< ID to name material mapping.

		MeshIds		_meshIds; ///< Per-vertex submesh ID.
		size_t		_maxMeshId = 0; ///< Maximum submesh ID encounter.

		OpacityMaps _opacityMaps; ///< Material opacity images.
		DiffuseMaps _diffuseMaps; ///< Material diffuse images.

		//std::vector<std::string>	_uniformColorMtlList;
		TagsMap		_tagsMap;  ///< Material tag images.
		TagsCoveringMaps _tagsCoveringMaps;  ///< Material covering tag images.
		std::vector<std::string> uniformColorMtlList; ///< List of materials with a diffuse map.

		SubMeshes	_subMeshes; ///< Submeshes, one per material, for rendering them separately.
		RenderCategory _typeOfRender = RenderCategory::diffuseMaterials; ///< Synthetic rendering mode.

		bool _albedoTexturesInitialized = false; ///< Are the texture initialized.
		std::vector<sibr::Texture2DRGBA::Ptr> _albedoTextures; ///< Albedo textures.
		std::vector<GLuint> _idTextures; ///< Texture handles.
		std::vector<sibr::Texture2DRGB::Ptr> _opacityTextures;///< Opacity textures.
		std::vector<GLuint> _idTexturesOpacity;///< Opacity texture handles.

		bool _hasTagsFile = false; ///< Is a tag file associated to the mesh.
		sibr::Texture2DRGB::Ptr _tagTexture; ///< Tag texture.
		GLuint _idTagTexture = 0; ///< Tag texture handle.

		bool _hasTagsCoveringFile = false; ///< Is a covering tag file associated to the mesh.
		sibr::Texture2DRGB::Ptr _tagCoveringTexture;///< Convering tag texture.
		GLuint _idTagCoveringTexture = 0; ///< Covering tag texture handle.

		std::vector<sibr::ImageRGB::Ptr> _listCoveringImagesTags;
		std::map<std::string,sibr::Texture2DRGB::Ptr> _tagsCoveringTexture;
		std::map<std::string,GLuint> _idTagsCoveringTexture;

		SwitchTagsProperty _switchTags;

		//AO attributes
		float _currentThreshold;
		AmbientOcclusion _ambientOcclusion; ///< AO options.
		std::function<sibr::Mesh::Colors(sibr::MaterialMesh&, const int)> _aoFunction; ///< AO generation function.
		bool _aoInitialized = false; ///< Is AO data initialized.
		float _averageSize = 0.0f; ///< Average maximum edge length.
		float _averageArea = 0.0f; ///< Average triangle area.

	};

	///// DEFINITION /////



	void	MaterialMesh::matIds(const MatIds& matIds) {
		_matIds = matIds;
	}
	const MaterialMesh::MatIds& MaterialMesh::matIds(void) const {
		return _matIds;
	}
	bool	MaterialMesh::hasMatIds(void) const {
		return (_triangles.size() > 0 && _triangles.size() == _matIds.size());
	}
	const MaterialMesh::MatIds& MaterialMesh::matIdsVertices(void) const {
		return _matIdsVertices;
	}
	const MaterialMesh::MatId2Name& MaterialMesh::matId2Name(void) const {
		return _matId2Name;
	}
	void MaterialMesh::matId2Name(const MatId2Name & matId2Name)
	{
		_matId2Name = matId2Name;
	}

	void	MaterialMesh::meshIds(const MeshIds& meshIds) {
		_meshIds = meshIds;
	}
	const MaterialMesh::MeshIds& MaterialMesh::meshIds(void) const {
		return _meshIds;
	}
	bool	MaterialMesh::hasMeshIds(void) const {
		return (!_meshIds.empty() && _meshIds.size() == _vertices.size());
	}

	// Opacity map function
	ImageRGB::Ptr MaterialMesh::opacityMap(const std::string& matName) const
	{
		std::map<std::basic_string<char>, sibr::ImagePtr<unsigned char, 3> >::const_iterator el = _opacityMaps.find(matName);
		if (el != _opacityMaps.end()) {
			return el->second;
		}
		return nullptr;
	}
	const MaterialMesh::OpacityMaps& MaterialMesh::opacityMaps(void) const
	{
		return _opacityMaps;
	}

	void MaterialMesh::hasTagsFile(bool hasOrNot) 
	{
		_hasTagsFile = hasOrNot;
	}

	const bool MaterialMesh::hasTagsFile(void) const 
	{
		return _hasTagsFile;
	}

	void MaterialMesh::hasTagsCoveringFile(bool hasOrNot) 
	{
		_hasTagsCoveringFile = hasOrNot;
	}

	const bool MaterialMesh::hasTagsCoveringFile(void) const 
	{
		return _hasTagsCoveringFile;
	}

	void MaterialMesh::opacityMaps(const OpacityMaps& maps)
	{
		_opacityMaps = maps;
	}

	void MaterialMesh::tagsMap(const TagsMap & map) {
		_tagsMap = map;
	}

	const MaterialMesh::TagsMap& MaterialMesh::tagsMap(void) const {
		return _tagsMap;
	}

	void MaterialMesh::tagsCoveringMaps(const TagsCoveringMaps & map) {
		_tagsCoveringMaps = map;
	}

	const MaterialMesh::TagsCoveringMaps& MaterialMesh::tagsCoveringMaps(void) const {
		return _tagsCoveringMaps;
	}

	sibr::ImageRGB::Ptr MaterialMesh::tagsCoveringMap(const std::string& matName) const {
		std::map<std::basic_string<char>, sibr::ImagePtr<unsigned char, 3> >::const_iterator el = _tagsCoveringMaps.find(matName);

		if (el != _tagsCoveringMaps.end()) {
			return el->second;
		}
		else return nullptr;
	}

	/// Set the switchTag 
	void MaterialMesh::switchTag(const SwitchTagsProperty& switchTag) {
		_switchTags = switchTag;
	}
	/// get the switchTag 
	const MaterialMesh::SwitchTagsProperty& MaterialMesh::switchTag(void) const {
		return _switchTags;
	}

	ImageRGBA::Ptr MaterialMesh::diffuseMap(const std::string& matName) const
	{
		std::map<std::basic_string<char>, sibr::ImagePtr<unsigned char, 4> >::const_iterator el = _diffuseMaps.find(matName);

		if (el != _diffuseMaps.end()) {
			return el->second;
		}
		else return nullptr;
	}

	/*ImageRGB MaterialMesh::diffuseMap(const std::string& matName)
	{
		auto & el =_diffuseMaps.find(matName);
		if (el != _diffuseMaps.end()) {
			return el->second;
		}
	}*/

	const MaterialMesh::DiffuseMaps& MaterialMesh::diffuseMaps(void) const
	{
		return _diffuseMaps;
	}

	void MaterialMesh::diffuseMaps(const DiffuseMaps& maps)
	{
		_diffuseMaps = maps;
	}

	const MaterialMesh::SubMeshes& MaterialMesh::subMeshes(void) const
	{
		return _subMeshes;
	}


	void MaterialMesh::subMeshes(const SubMeshes& subMeshes)
	{
		_subMeshes = subMeshes;
	}

	const MaterialMesh::RenderCategory& MaterialMesh::typeOfRender(void) const {
		return _typeOfRender;
	}


	inline const MaterialMesh::AmbientOcclusion & MaterialMesh::ambientOcclusion(void)
	{
		return _ambientOcclusion;
	}

	inline void MaterialMesh::aoFunction(std::function<sibr::Mesh::Colors
	(sibr::MaterialMesh&, const int)>&
		aoFunction)
	{
		_aoFunction = aoFunction;
	}

	void MaterialMesh::typeOfRender(const RenderCategory& type) {
		_typeOfRender = type;
	}


} // namespace sibr
