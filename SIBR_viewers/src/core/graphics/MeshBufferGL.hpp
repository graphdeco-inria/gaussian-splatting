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

# include <array>
# include <vector>
# include "core/graphics/Config.hpp"


namespace sibr
{
	template <typename TTo, typename TFrom>
	static std::vector<TTo> prepareVertexData(const std::vector<TFrom>& fromData, uint numVertices);

	template <typename TTo, typename TFrom>
	static void appendVertexData(std::vector<TTo>& toData, const std::vector<TFrom>& fromData);

	template <typename T>
	static uint getVectorDataSize(const std::vector<T>& v);


	class Mesh;

	/**
	* This class is used to render mesh. It act like a vertex buffer object
	* (in reality there also a vertex array object and maybe other data).
	* \ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT MeshBufferGL
	{
	public:

		/** Predefined shader attribute location. */
		enum AttribLocation
		{
			VertexAttribLocation	= 0,
			ColorAttribLocation		= 1,
			TexCoordAttribLocation	= 2,
			NormalAttribLocation	= 3,

			AttribLocationCount
		};
		
		/** Predefined buffer location. */
		enum
		{
			BUFINDEX	= 0,
			BUFVERTEX	= 1,
			BUFADJINDEX	= 2,

			BUFCOUNT
		};

		// A hash function used to hash a pair of any kind 
		struct hash_pair { 
			template <class T1, class T2> 
			size_t operator()(const std::pair<T1, T2>& p) const
			{ 
				auto hash1 = std::hash<T1>{}(p.first); 
				auto hash2 = std::hash<T2>{}(p.second); 
				return hash1 ^ hash2; 
			} 
		}; 

	public:

		/// Constructor.
		MeshBufferGL( void );

		/// Destructor.
		~MeshBufferGL( void );

		/// Move constructor.
		MeshBufferGL( MeshBufferGL&& other );

		/// Move operator.
		MeshBufferGL& operator =( MeshBufferGL&& other );

		/** Fetch indices from a mesh to insert them in the element buffer
		* \param mesh the mesh to upload
		* \param adjacency tells whether the indices should contain adjacents vertices
		* \note This function can't fail (errors stop the program with a message).
		*/
		void	fetchIndices( const Mesh& mesh, bool adjacency = false );
		
		/** Build from a mesh so you can then draw() it to render it.
		* \param mesh the mesh to upload
		* \param adjacency tells whether the indices should contain adjacents vertices
		* \note This function can't fail (errors stop the program with a message).
		*/
		void	build( const Mesh& mesh, bool adjacency = false );

		/** Delete the GPU buffer, freeing memory. */
		void	free(void);

		/** This bind and draw elements stored in the buffer.
			\param adjacency adds adjacent triangles info to geometry shader
		*/
		void	draw(bool adjacency = false) const;

		/** This bind and draw elements in [begin, end[ stored in the buffer.
			\param begin ID of the first primitive to render
			\param end ID after the last primitive to render
			\param adjacency adds adjacent triangles info to geometry shader
		*/
		void	draw(unsigned int begin, unsigned int end, bool adjacency = false) const;

		/** This bind and draw elements stored in the buffer with tessellation shader enabled. */
		void  drawTessellated(void) const;
		
		/** This bind and draw elements stored in the buffer, using pairs of indices to draw lines. */
		void draw_lines(void) const;

		/** This bind and draw vertex points stored in the buffer. */
		void	draw_points( void ) const;

		/** This bind and draw vertex points in [begin, end[ stored in the buffer.
			\param begin ID of the first primitive to render
			\param end ID after the last primitive to render
		*/
		void	draw_points( unsigned int begin, unsigned int end ) const;

		/** Bind the vertex and index buffers. */
		void	bind(void) const;

		/** Bind only indexes (useful for combining with other form of mesh. e.g. SlicedMesh) */
		void	bindIndices(void) const;

		/** Unbind arrays and buffers. */
		void	unbind(void) const;

		/** Copy constructor (disabled). */
		MeshBufferGL(const MeshBufferGL&) = delete;

		/** Copy operator (disabled). */
		MeshBufferGL& operator =(const MeshBufferGL&) = delete;

	private:
		
		GLuint 							_vaoId; ///< Vertex array object ID.
		std::array<GLuint, BUFCOUNT>	_bufferIds; ///< Buffers IDs.
		uint 							_indexCount; ///< Number of elements in the index buffer.
		uint							_adjacentIndexCount; ///< Number of elements in the triangles_adjacency index buffer.
		uint							_vertexCount; ///< Number of elements in the vertex buffer.

		bool initVertexBuffer = false,
			 initIndexBuffer = false,
			 initAdjacentIndexBuffer = false;
	};

	///// DEFINITION /////


} // namespace sibr
