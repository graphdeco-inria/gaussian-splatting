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
# include "core/graphics/MeshBufferGL.hpp"
# include "core/graphics/Image.hpp"

// Be sure to use STL objects from client's dll version by exporting this declaration (see warning C4251)
//template class SIBR_GRAPHICS_EXPORT std::vector<Vector3f>;
//template class SIBR_GRAPHICS_EXPORT std::vector<Vector3u>;

namespace sibr
{
	/** Store both CPU and GPU data for a geometric mesh.
		Provide many processing and display methods.
	\ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT Mesh
	{
		SIBR_CLASS_PTR(Mesh);

	public:

		typedef	std::vector<Vector3f>	Vertices;
		typedef	std::vector<Vector3f>	Normals;
		typedef std::vector<Vector3u>	Triangles;
		typedef	std::vector<Vector3f>	Colors;
		typedef	std::vector<Vector2f>	UVs;

		/** Mesh rendering mode. */
		enum RenderMode
		{
			PointRenderMode,
			LineRenderMode,
			FillRenderMode
		};

		/** Mesh rendering options. */
		struct RenderingOptions {
			bool depthTest = true; ///< Should depth test be performed.
			bool backFaceCulling = true; ///< Should back faces be culled.
			RenderMode mode = FillRenderMode; ///< Rendering mode: points, lines, filled.
			bool frontFaceCulling = false; ///< Cull fornt faces.
			bool invertDepthTest = false; ///< Invert the depth test.
			bool tessellation = false; ///< Is there a tessellation shader
			bool adjacency = false;
		};


	public:

		/** Constructor.
		\param withGraphics init associated OpenGL buffers object (requires an openGL context)
		*/
		Mesh( bool withGraphics = true );

		/** Set vertices.
		\param vertices the new vertices
		*/
		inline void	vertices(const Vertices& vertices);

		/** Set vertices from a vector of floats (linear).
		\param vertices the new vertices
		*/
		void vertices( const std::vector<float>& vertices );
		
		/** \return a reference to the vertices. */
		inline const Vertices& vertices( void ) const;

		/** Update a specific vertex position
		\param vertex_id the vertex location in the list
		\param v the new value
		\note If the mesh is used by the GPU, data will be udpated.
		*/
		inline void replaceVertice(int vertex_id, const sibr::Vector3f & v) ;

		/** \return a deep copy of the mesh. */
		Mesh::Ptr clone() const;

		/** \return vertices in an array using the following format:
		 {0x, 0y, 0z, 1x, 1y, 1z, 2x, 2y, 2z, ...}.
		 Useful for rendering and converting to another mesh
		 struct.
		 */
		inline const float* vertexArray( void ) const;

		/** Set triangles. Each triangle contains 3 indices and
		 each indice correspond to a vertex in the vertices list.
		 \param triangles the list of indices to use
		 */
		inline void	triangles(const Triangles& triangles);
		
		/** Set triangles. Using a flat vector of uints.
		\param triangles the new indices
		*/
		void	triangles(const std::vector<uint>& triangles);
		
		/** \return a reference to the triangles list. */
		inline const Triangles& triangles( void ) const;
		
		/** \return triangles in an array using the following format:
		 {0a, 0b, 0c, 1a, 1b, 1c, 2a, 2b, 2c, ...}.
		 Useful for rendering and converting to another mesh
		 struct.
		 */
		inline const uint* triangleArray( void ) const;

		/** Set vertex colors.
		\param colors the new vertex colors
		*/
		inline void	colors( const Colors& colors );

		/** \return a reference to the vertex color list. */
		inline const Colors& colors( void ) const;

		/** \return true if each vertex has a color assigned. */
		inline bool	hasColors( void ) const;

		/** \return colors in an array using the following format:
		 {0r, 0g, 0b, 1r, 1g, 1b, 2r, 2g, 2b, ...}.
		 Useful for rendering and converting to another mesh
		 struct.*/
		inline const float* colorArray( void ) const;

		/** Set vertex texture coordinates.
		\param texcoords the new vertex texture coordinates
		*/
		inline void	texCoords( const UVs& texcoords );

		/** Set texture coordinates using a flat vector of floats.
		\param texcoords the new vertex texture coordinates
		*/
		void		texCoords( const std::vector<float>& texcoords );

		/** \return a reference to the vertex texture coordinates list. */
		inline const UVs& texCoords( void ) const;

		/** \return true if each vertex has texture coordinates assigned. */
		inline bool	hasTexCoords( void ) const;

		/** \return texture coordinate in an array using the following format:
		 {0u, 0v, 1u, 1v, 2u, 2v, ...}
		 Useful for rendering and converting to another mesh
		 struct.*/
		inline const float* texCoordArray( void ) const;

		/** \return texture image file name */
		std::string getTextureImageFileName()	const { return _textureImageFileName; }

		/** Set vertex normals.
		\param normals the new vertex normals
		*/
		inline void	normals(const Normals& normals);

		/** Set normals using a flat vector of floats.
		\param normals the new vertex normals
		*/
		void	normals(const std::vector<float>& normals);
		
		/** \return a reference to the vertex normals list. */
		inline const Normals& normals( void ) const;

		/** \return true if each vertex has a normal assigned. */
		inline bool	hasNormals( void ) const;
		
		/** \return normals in an array using the following format:
		 {0x, 0y, 0z, 1x, 1y, 1z, 2x, 2y, 2z, ...}.
		 Useful for rendering and converting to another mesh
		 struct.*/
		inline const float* normalArray( void ) const;

		/** Make the mesh whole, ie it will have default values for all components (texture, materials, colors, etc)
		  It is useful when merging two meshes. If the second one is missing some attributes, the merging will break the mesh state if it isn't made whole.
		  */
		void	makeWhole(void);

		/** Generate vertex normals by using the average of
		 all triangle normals around a each vertex.
		 */
		void	generateNormals( void );

		/** Generate smooth vertex normals by using the average of
		 all triangle normals around a each vertex and iterating this process.
		 Takes also the area of triangles as a weight.
		 \param numIter iteration count
		 */
		void	generateSmoothNormals(int numIter);

		/** Generate smooth vertex normals by using the average of
		 all triangle normals around a each vertex and iterating this process.
		 Takes also the area of triangles as a weight.
		 This methods also consider the fact that because of texture coordinates we may have duplicates vertices
		 \param numIter iteration count
		 */
		void	generateSmoothNormalsDisconnected(int numIter);

		/** Perform laplacian smoothing on the mesh vertices.
		\param numIter smoothing iteration count
		\param updateNormals should the normals be recomputed after smoothing
		*/
		void laplacianSmoothing(int numIter, bool updateNormals);

		/** Perform adaptive Taubin smoothing on the mesh vertices.
		\param numIter smoothing iteration count
		\param updateNormals should the normals be recomputed after smoothing
		*/
		void adaptativeTaubinSmoothing(int numIter, bool updateNormals);

		/** Generate a new mesh given a boolean function that
		  state if each vertex should be kept or not.
		  \param func the function that, based on a vertex ID, tells if it should be kept or not
		  \return the submesh
		  */
		Mesh generateSubMesh(std::function<bool(int)> func) const;

		/** Load a mesh from the disk.
		\param filename the file path
		\return a success flag
		\note Supports OBJ and PLY for now.
		*/
		bool	load( const std::string& filename, const std::string& dataset_path = "" );
		/* test for SfM */
		bool	loadSfM( const std::string& filename, const std::string& dataset_path = "" );
		
		/** Load a scene from a set of mitsuba XML scene files (referencing multiple OBJs/PLYs). 
		It handles instances (duplicating the geoemtry and applying the per-instance transformation).
		\param filename the file path
		\return a success flag
		*/
		bool	loadMtsXML(const std::string& filename);

		/** Save the mesh to the disk. When saving as a PLY, use the universal flag to indicate if you want this mesh to be readable
		 by most 3d applications (e.g. MeshLab). In this other case, the mesh will be saved with higher-precision custom PLY attributes.
		\param filename the file path
		\param universal if true, the mesh will be compatible with external 3D viewers
		\param textureName name of a texture to reference in the file (Meshlab compatible)
		\note Supports OBJ (ignoring the universal flag) and PLY for now.
		 */
		void	save( const std::string& filename, bool universal=false, const std::string& textureName="TEXTURE_NAME_TO_PUT_IN_THE_FILE" ) const;

		/** Save the mesh to .ply file (using the binary version).
		 \param filename the file path
		 \param universal indicates if you want this mesh to be readable by most 3d viewer application (e.g. MeshLab). In this other case, the mesh will be saved with higher-precision custom PLY attributes.
		 \param textureName name of a texture to reference in the file (Meshlab compatible) 
		*/
		bool		saveToBinaryPLY( const std::string& filename, bool universal=false, const std::string& textureName = "TEXTURE_NAME_TO_PUT_IN_THE_FILE") const;
		
		/** Save the mesh to .ply file (using the ASCII version).
		 \param filename the file path
		 \param universal indicates if you want this mesh to be readable by most 3d viewer application (e.g. MeshLab).
		 \param textureName name of a texture to reference in the file (Meshlab compatible)
		 In this other case, the mesh will be saved with higher-precision custom PLY attributes.
		*/
		bool		saveToASCIIPLY( const std::string& filename, bool universal=false, const std::string& textureName="TEXTURE_NAME_TO_PUT_IN_THE_FILE" ) const;

		/** Save the mesh to .obj file.
		 \param filename the file path
		 \warning the vertex colros won't be saved
		*/
		bool		saveToObj( const std::string& filename) const;

		/* Export to OFF file format stream, can be used to convert to CGAL mesh data structures.
		\param verbose display additional info
		\return a stream containing the content of the mesh in the OFF format
		*/
		std::stringstream getOffStream(bool verbose = false) const;

		/* Load from OFF file format stream, can be used to convert from CGAL mesh data structures.
		\param stream a stream containing the content of the mesh in the OFF format
		\param generateNormals should the normals be generated
		*/
		void fromOffStream(std::stringstream& stream, bool generateNormals = true);

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

		/** Render a part of the geometry (taken either from the index buffer or directly in the vertex buffer) using OpenGL.
		\param begin first item to render index
		\param end last item to render index
		\param depthTest should depth testing be performed
		\param backFaceCulling should culling be performed
		\param mode the primitives rendering mode
		\param frontFaceCulling should the culling test be flipped
		\param invertDepthTest should the depth test be flipped (GL_GREATER_THAN)
		*/
		void	renderSubMesh(unsigned int begin, unsigned int end,
			bool depthTest = true,
			bool backFaceCulling = true,
			RenderMode mode = FillRenderMode,
			bool frontFaceCulling = false,
			bool invertDepthTest = false
		) const;

		/** Force upload of data to the GPU.
		\param adjacency should we give adjacent triangles info in buffer
		*/
		void	forceBufferGLUpdate( bool adjacency = false ) const;

		/** Delete GPU mesh data. */
		void	freeBufferGLUpdate(void) const;

		/** Render the mesh vertices as points.
		\param depthTest should depth testing be performed
		*/
		void	render_points(bool depthTest) const;

		/** Render the mesh vertices as points.
		\note The depth test state will be whatever it is currently set to.
		*/
		void	render_points(void) const;

		/** Render the mesh vertices as successive lines.
		\note The depth test state will be whatever it is currently set to.
		*/
		void	render_lines(void) const;

		/** Merge another mesh into this one.
		\param other the mesh to merge
		\sa makeWhole
		*/
		void		merge( const Mesh& other );

		/** Erase some of the triangles.
		\param faceIDList a list of triangle IDs to erase
		*/
		void		eraseTriangles(const std::vector<uint>& faceIDList);

		/** Submesh extraction options. */
		enum class VERTEX_LIST_CHOICE { KEEP, REMOVE };
		
		/** Submesh structure. */
		struct SubMesh {
			std::shared_ptr<sibr::Mesh> meshPtr; ///< Mesh composed of the extracted vertices.
			std::vector<int> complementaryVertices; /// < Vertices present in at least one removed triangle, can be used as an arg to extractSubMesh() to get the complementary mesh.
		};

		/** Extract a submesh based on a list of vertices to keep/remove.
		\param vertices a list of vertex indices
		\param v_choice should the listed vertices be removed or kept
		\return the extracted submesh with additional information
		*/
		SubMesh extractSubMesh(const std::vector<int> & vertices, VERTEX_LIST_CHOICE v_choice) const;

		/** \return a copy of the mesh with inverted faces (flipping IDs). */
		sibr::Mesh invertedFacesMesh() const;

		/** \return a copy of the mesh with "doubled" faces (obtained by merging the current mesh with a copy with inverted faces. */
		sibr::Mesh::Ptr invertedFacesMesh2() const;

		/** \return the path the mesh was loaded from. */
		inline const std::string getMeshFilePath( void ) const;

		/** \return the mesh bouding box. */
		Eigen::AlignedBox<float,3>	getBoundingBox( void ) const;

		/** \return the mesh centroid. */
		sibr::Vector3f centroid() const;

		/** Estimated the mesh bounding sphere.
		\param outCenter will contain the sphere center
		\param outRadius will contain the sphere radius
		\param referencedOnly if true, only consider vertices that are part of at least one face
		\param usePCcenter if true, only consider vertices for center computation. Intended to be true when using the function on point clouds.
		*/
		void getBoundingSphere(Vector3f& outCenter, float& outRadius, bool referencedOnly=false, bool usePCcenter=false) const;

		/** Subdivide a mesh triangles until an edge size threshold is reached.
		\param limitSize the maximum edge length allowed
		\param maxRecursion maximum subdivision iteration count
		\return the subdivided mesh
		\bug SR: Can be stuck in a loop in some cases.
		*/
		sibr::Mesh::Ptr subDivide(float limitSize, size_t maxRecursion = std::numeric_limits<size_t>::max()) const;

		/** \return the mean edge size computed over all triangles. */
		float meanEdgeSize() const;

		/** Split a mesh in its connected components. 
		\return a list of list of vertex indices, each list defining a component
		*/
		std::vector<std::vector<int> > removeDisconnectedComponents();

		/** Generate a simple cube with normals.
		\param withGraphics should the mesh be on the GPU
		\return a cube mesh
		*/
		static sibr::Mesh::Ptr getTestCube(bool withGraphics = true);

		/** Generate a sphere mesh.
		\param center the sphere center
		\param radius the sphere radius
		\param withGraphics should the mesh be on the GPU
		\param precision number of subdivisions on each dimension
		\return the sphere mesh
		*/
		static Mesh::Ptr getSphereMesh(const Vector3f& center, float radius, bool withGraphics = true, int precision = 50);

		/** Environment sphere generation options: top/bottom parts or both. */
		enum class PartOfSphere {WHOLE, UP, BOTTOM};

		/* Generate a simple environment sphere with UVs coordinates to use with lat-long environment maps.
		\param center the sphere center
		\param radius the sphere radius
		\param zenith the up axis to orient the sphere around
		\param north the horizontal north axis
		\param part which aprt of the sphere should be generated
		\return the sphere mesh
		*/
		static sibr::Mesh::Ptr getEnvSphere(sibr::Vector3f center, float radius, sibr::Vector3f zenith, sibr::Vector3f north,
											PartOfSphere part = PartOfSphere::WHOLE);

	protected:

		/** Wrapper around a MeshBuffer, used to prevent copying OpenGL object IDs. */
		struct BufferGL
		{
			/** Default constructor. */
			BufferGL(void) : dirtyBufferGL(true), bufferGL(nullptr) { }

			/** Copy constructor.  Copy the GPU buffers wrapper if it exists.
			\param other object to copy.
			*/
			BufferGL(const BufferGL& other) { operator =(other); }

			/** Copy operator.  Copy the GPU buffers wrapper if it exists.
			\param other object to copy.
			\return itself
			*/
			BufferGL& operator =(const BufferGL& other) {
				bufferGL.reset(other.bufferGL? new MeshBufferGL() : nullptr);
				dirtyBufferGL = (other.bufferGL!=nullptr);
				return *this;
			}

			bool			dirtyBufferGL; ///< Should GL data be updated.
			std::unique_ptr<MeshBufferGL>	bufferGL; ///< Internal OpenGL data.
		};
		public: mutable BufferGL	_gl; ///< Internal OpenGL data.

		// Seb: It would be better if MeshBufferGL (and GL stuffs) were outside this class.
		// It should work exactly like Image (CPU, here it's Mesh) and Texture(GPU version
		// of Image, here it's MeshBufferGL)

		Vertices	_vertices; ///< Vertex positions.
		Triangles	_triangles; ///< Triangle indices.
		Normals		_normals; ///< Vertex normals.
		Colors		_colors; ///< Vertex colors.
		UVs			_texcoords; ///< Vertex UVs.

	private:
		std::string _meshPath; ///< Source path, can be used to reload the mesh with/without graphics option in constructor
		std::string _textureImageFileName; // filename of texture image
		mutable RenderingOptions _renderingOptions; // Keeps last rendering options
	};

	///// DEFINITION /////

	void	Mesh::vertices( const Vertices& vertices ) {
		_vertices = vertices; _gl.dirtyBufferGL = true;
	}

	const Mesh::Vertices& Mesh::vertices( void ) const {
		return _vertices;
	}

	inline void Mesh::replaceVertice(int vertex_id, const sibr::Vector3f & v)
	{
		if (vertex_id >= 0 && vertex_id < (int)(vertices().size())) {
			_vertices[vertex_id] = v;
			_gl.dirtyBufferGL = true;
		}
	}

	void	Mesh::triangles( const Triangles& triangles ) {
		_triangles = triangles; _gl.dirtyBufferGL = true;
	}

	const Mesh::Triangles& Mesh::triangles( void ) const {
		return _triangles;
	}

	void	Mesh::colors( const Colors& colors ) {
		_colors = colors; _gl.dirtyBufferGL = true;
	}
	const Mesh::Colors& Mesh::colors( void ) const {
		return _colors;
	}
	bool	Mesh::hasColors( void ) const {
		return (_vertices.size() > 0 && _vertices.size() == _colors.size());
	}

	void	Mesh::normals( const Normals& normals ) {
		_normals = normals; _gl.dirtyBufferGL = true;
	}
	const Mesh::Normals& Mesh::normals( void ) const {
		return _normals;
	}
	bool	Mesh::hasNormals( void ) const {
		return (_vertices.size() > 0 && _vertices.size() == _normals.size());
	}

	const float* Mesh::vertexArray( void ) const {
		return _vertices.empty()? nullptr : &(_vertices[0][0]);
	}
	const uint* Mesh::triangleArray( void ) const {
		return _triangles.empty() ? nullptr : &(_triangles[0][0]);
	}
	const float* Mesh::colorArray( void ) const {
		return _colors.empty() ? nullptr : &(_colors[0][0]);
	}
	const float* Mesh::normalArray( void ) const {
		return _normals.empty() ? nullptr : &(_normals[0][0]);
	}

	void	Mesh::texCoords( const UVs& texcoords ) {
		_texcoords = texcoords; _gl.dirtyBufferGL = true;
	}

	const Mesh::UVs& Mesh::texCoords( void ) const {
		return _texcoords;
	}

	bool	Mesh::hasTexCoords( void ) const {
		return (_vertices.size() > 0 && _vertices.size() == _texcoords.size());
		//return !_texcoords.empty();
	}

	const float* Mesh::texCoordArray( void ) const {
		return _texcoords.empty() ? nullptr : &(_texcoords[0][0]);
	}

	//*/

} // namespace sibr
