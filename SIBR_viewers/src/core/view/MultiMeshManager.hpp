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

#include <core/graphics/Mesh.hpp>
#include <core/graphics/Shader.hpp>
#include <core/view/ViewBase.hpp>
#include <core/view/InteractiveCameraHandler.hpp>
#include <core/raycaster/CameraRaycaster.hpp>

#include <list>

namespace sibr {

	class MultiMeshManager;
	class MeshData;

	// Hierarchy of shader wrappers, so there is no duplication for uniforms, init(), set() and render().

	/** Shader wrapper for sending mesh display options to the GPU (while avoiding duplicated uniforms) . 
	 * Contains an MVP matrix and an opacity value.
	  \ingroup sibr_view
	 */
	class SIBR_VIEW_EXPORT ShaderAlphaMVP {
		SIBR_CLASS_PTR(ShaderAlphaMVP);
	public:
		/** Initialize the shader. 
		 *\param name the shader name
		 *\param vert the vertex shader content
		 *\param frag the fragment shader ocntent
		 *\param geom the geometry shader content
		 */
		virtual void initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom = "");

		/* Set uniforms based on the camera position and mesh options.
		 * \param eye the current viewpoint
		 * \param data the mesh display options
		 */
		virtual void setUniforms(const Camera & eye, const MeshData & data);

		/** Render using the passed information.
		 * \param eye the current viewpoint
		 * \param data the mesh display options
		 */
		virtual void render(const Camera & eye, const MeshData & data);

	protected:
		GLShader				shader; ///< Base shader object.
		GLuniform<Matrix4f>		mvp; ///< MVP matrix.
		GLuniform<float>		alpha = 1.0; ///< Opacity.
	};

	/** Shader wrapper for sending mesh display options to the GPU (while avoiding duplicated uniforms) .
	 * Adds a user-defined color. \sa ShaderAlphaMVP
	  \ingroup sibr_view
	 */
	class SIBR_VIEW_EXPORT ColorMeshShader : public ShaderAlphaMVP {
	public:
		/** Initialize the shader.
		 *\param name the shader name
		 *\param vert the vertex shader content
		 *\param frag the fragment shader ocntent
		 *\param geom the geometry shader content
		 */
		virtual void initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom = "");

		/* Set uniforms based on the camera position and mesh options.
		 * \param eye the current viewpoint
		 * \param data the mesh display options
		 */
		virtual void setUniforms(const Camera & eye, const MeshData & data);

	protected:
		GLuniform<Vector3f>	user_color; ///< user-defined constant color.
	};

	/** Shader wrapper for sending mesh display options to the GPU (while avoiding duplicated uniforms) .
	 * Adds a point size. \sa ShaderAlphaMVP
	  \ingroup sibr_view
	 */
	class SIBR_VIEW_EXPORT PointShader : public ColorMeshShader {
	public:
		/** Initialize the shader.
		 *\param name the shader name
		 *\param vert the vertex shader content
		 *\param frag the fragment shader ocntent
		 *\param geom the geometry shader content
		 */
		void initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom = "") override;

		/* Set uniforms based on the camera position and mesh options.
		 * \param eye the current viewpoint
		 * \param data the mesh display options
		 */
		virtual void setUniforms(const Camera & eye, const MeshData & data) override;

		/** Render using the passed information.
		 * \param eye the current viewpoint
		 * \param data the mesh display options
		 */
		virtual void render(const Camera & eye, const MeshData & data) override;

	protected:
		GLuniform<int> radius; ///< Point screenspace radius.
	};

	/** Shader wrapper for sending mesh display options to the GPU (while avoiding duplicated uniforms) .
	 * Adds shading parameters. \sa ShaderAlphaMVP
	  \ingroup sibr_view
	 */
	class SIBR_VIEW_EXPORT MeshShadingShader : public ColorMeshShader {
	public:
		/** Initialize the shader.
		 *\param name the shader name
		 *\param vert the vertex shader content
		 *\param frag the fragment shader ocntent
		 *\param geom the geometry shader content
		 */
		void initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom = "") override;

		/* Set uniforms based on the camera position and mesh options.
		 * \param eye the current viewpoint
		 * \param data the mesh display options
		 */
		virtual void setUniforms(const Camera & eye, const MeshData & data) override;

	protected:
		GLuniform<Vector3f>		light_position; ///< Light position for shading.
		GLuniform<bool>			phong_shading, use_mesh_color; ///< Should the mesh be shaded, which color should be used.
	};

	/** Shader wrapper for sending mesh display options to the GPU (while avoiding duplicated uniforms) .
	 * Adds line length option. \sa ShaderAlphaMVP
	  \ingroup sibr_view
	 */
	class SIBR_VIEW_EXPORT NormalRenderingShader : public ColorMeshShader {
	public:
		/** Initialize the shader.
		 *\param name the shader name
		 *\param vert the vertex shader content
		 *\param frag the fragment shader ocntent
		 *\param geom the geometry shader content
		 */
		void initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom = "") override;

		/* Set uniforms based on the camera position and mesh options.
		 * \param eye the current viewpoint
		 * \param data the mesh display options
		 */
		virtual void setUniforms(const Camera & eye, const MeshData & data) override;

	protected:
		GLuniform<float> normals_size; ///< Normal line length.
	};


	/** Helper class containing all information relative to how to render a mesh for debugging purpose in a MultiMeshManager.
	 * You can chain setters to modify multiple properties sequentially (chaining).
	\sa MultiMeshManager
	 \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT MeshData {
		SIBR_CLASS_PTR(MeshData);

	public:
		friend class MultiMeshManager;

		/** Color mode: constant defined by the user, or per-vertex attribute. */
		enum ColorMode : int { USER_DEFINED, VERTEX };
		/** Type of mesh: points, lines or faces. Dummy is for unitialized objects. */
		enum MeshType : int { POINTS = 0, LINES = 1, TRIANGLES = 2, DUMMY };
		/** When displaying normals, use the per-face or per-vertices normals */
		enum NormalMode { PER_TRIANGLE, PER_VERTEX };

		/** COnstructor.
		 *\param _name the object name
		 *\param mesh_ptr the geoemtry to display
		 *\param mesh_type the type of mesh
		 *\param render_mode for triangle meshes, should they be displayed filled, as wireframes, or point clouds (\sa Mesh).
		 */
		MeshData(const std::string & _name = "", Mesh::Ptr mesh_ptr = {}, MeshType mesh_type = TRIANGLES, Mesh::RenderMode render_mode = Mesh::FillRenderMode);

		/** Render the geometry. */
		void	renderGeometry() const;

		/** Display the GUI list item associated to this object.
		 *\param name additional display name
		 */
		void	onGUI(const std::string & name);

		/** \return if the object is valid. */
		operator bool() const;

		/** \return a string describing the geometry. */
		std::string getInfos() const;

		/** Set the color.
		 *\param col the color to use
		 *\return the options object, for chaining.
		 *\note To see the color, you might also have to specify the color mode if your mesh has vertex colors.
		 */
		MeshData & setColor(const Vector3f & col);

		/** Set the backface culling.
		 *\param bf should culling be performed
		 *\return the options object, for chaining.
		 */
		MeshData & setBackFace(bool bf);

		/** Set the depth test.
		 *\param dt should depth testing be enabled
		 *\return the options object, for chaining.
		 */
		MeshData & setDepthTest(bool dt);

		/** Set a random constant color.
		 *\return the options object, for chaining.
		 */
		MeshData & setColorRandom();

		/** Set the size of points for point-based display.
		 *\param rad the point radius in screenspace
		 *\return the options object, for chaining.
		 */
		MeshData & setRadiusPoint(int rad);

		/** Set the size of points for point-based display.
		 *\param scale
		 *\return the options object, for chaining.
		 */
		MeshData& setScale(float s);

		/** Set Transformation matrix.
		 *\param model matrix
		 *\return the options object, for chaining.
		 */
		MeshData& setTransformation(sibr::Matrix4f& tr);

		/** Set the opacity.
		 *\param alpha the opacity value for the whole object
		 *\return the options object, for chaining.
		 */
		MeshData & setAlpha(float alpha);

		/** Set the color mode (either user-defined constant or vertex color).
		 *\param mode the color mode 
		 *\return the options object, for chaining.
		 */
		MeshData & setColorMode(ColorMode mode);

		/** Get the display options of the additional normals geometry.
		 *\return the normals options.
		 */
		MeshData getNormalsMeshData() const;

		std::string			name; ///< Mesh name.

		Mesh::Ptr			meshPtr; ///< Geometry.
		MeshType			meshType; ///< Type of mesh.
		Mesh::RenderMode	renderMode; ///< Render mode for triangle meshes.

		Matrix4f			transformation = Matrix4f::Identity(); ///< Additional model transformation.

		Raycaster::Ptr		raycaster; ///< Associated raycaster (optional)

		bool				depthTest = true; ///< Perform depth test.
		bool				backFaceCulling = true; ///< Perform culling.
		bool				frontFaceCulling = false; ///< Swap front and back faces for culling.
		bool				invertDepthTest = false; ///< Switch the depth test to "greater than".
		bool				active = true; ///< Should the object be displayed.
		bool				phongShading = false; ///< Apply Phong shading to the object.

		// Points
		int					radius = 5; ///< Point screenspace radius.
		float				scale = 1.0f;

		// Colors
		ColorMode			colorMode = USER_DEFINED; ///< Color mode.
		Vector3f			userColor = { 0.5,0.5,0.5 }; ///< Constant user-defined  color.
		float				alpha = 1.0f; ///< Opacity.

		// Normals
		Vector3f normalsColor = { 1,0,1 }; ///< Normal lines color.
		float normalsLength = 1.0f; ///< Normal lines length.
		NormalMode normalMode = PER_TRIANGLE; ///< Which normals should be displayed.
		bool normalsInverted = false; ///< Flip the normal lines orientation.
		bool showNormals = false; ///< Should the normals be displayed.

	protected:
		
		static MeshData dummy; ///< OPtions object used for non-existing objects.
	};


	/** Provide a view to render and interact with several meshes, 
	 * useful for debugging purposes for instance.
	 * The API supports chaining when setting mesh display options. You can for instance do:
	 * manager.addMesh("my mesh", mesh).setDepthtest(true).setAlpha(0.5f); 
	  \ingroup sibr_view
	*/
	class SIBR_VIEW_EXPORT MultiMeshManager : public ViewBase {
		SIBR_CLASS_PTR(MultiMeshManager);

	public:

		/** Constructor
		\param _name Name used for GUI panels as a prefix to avoid collision.
		\note Requires an OpenGL context setup
		*/
		MultiMeshManager(const std::string & _name = "MultiMeshManager");
	
		/** Add a mesh to the visualization.
		\param name name used for the object, if it already exist it will update the geometry and preserve display options
		\param mesh the mesh
		\param use_raycaster should a raycaster be setup, for trackball centering for instance
		\return a reference to the object display options (for chained modifications).
		*/
		MeshData & addMesh(const std::string & name, Mesh::Ptr mesh, bool use_raycaster = true);

		/** Add a mesh to the visualization.
		\param name name used for the object, if it already exist it will update the geometry and preserve display options
		\param mesh the mesh
		\param raycaster existing raycaster, for trackball centering for instance
		\param create_raycaster should a raycaster be setup if the passed raycaster is null
		\return a reference to the object display options (for chained modifications).
		*/
		MeshData & addMesh(const std::string & name, Mesh::Ptr mesh, Raycaster::Ptr raycaster, bool create_raycaster = false);

		/** Add lines to the visualization, using the mesh vertices as line endpoints.
		\param name name used for the object, if it already exist it will update the geometry and preserve display options
		\param mesh the mesh
		\return a reference to the object display options (for chained modifications).
		*/
		MeshData & addMeshAsLines(const std::string & name, Mesh::Ptr mesh);

		/** Add lines to the visualization, defined by their endpoints.
		\param name name used for the object, if it already exist it will update the geometry and preserve display options
		\param endPoints the line endpoints
		\param color the display color to use
		\return a reference to the object display options (for chained modifications).
		*/
		MeshData & addLines(const std::string & name, const std::vector<Vector3f> & endPoints, const Vector3f & color = { 0,1,0 });

		/** Add points to the visualization.
		\param name name used for the mesh, if it already exist it will update the geometry and preserve display options
		\param points the points
		\param color the display color to use
		\return a reference to the display options (for chained modifications).
		*/
		MeshData & addPoints(const std::string & name, const std::vector<Vector3f> & points, const Vector3f & color = { 1,0,0 });

		/** Accessor to the options of a visualized object.
			\param name the object name to look for
			\return a reference to the object options if it exists, or to MeshData::dummy if no match was found.
		*/
		MeshData & getMeshData(const std::string & name);

		/** Remove a object from the viewer.
		\param name the name of the object to remove
		*/
		void		removeMesh(const std::string & name);

		void		setIntialView(const std::string& dataset_path);

		// ViewBase interface

		/** Update state based on user input.
		 * \param input user input
		 * \param vp input viewport
		 * \note Used when the view is in a multi-view system.
		 */
		virtual void	onUpdate(Input& input, const Viewport & vp) override;

		/** Render content in the currently bound RT, using a specific viewport.
		 * \param viewport destination viewport
		 * \note Used when the view is in a multi-view system.
		 */
		virtual void	onRender(const Viewport & viewport) override;

		/** Render content in a RT, using the RT viewport.
		 * \param dst destination RT
		 */
		virtual void	onRender(IRenderTarget & dst) ;

		/** Display GUI. */
		virtual void	onGUI() override;

		/** \return the view camera handler */
		InteractiveCameraHandler & getCameraHandler() { return camera_handler; }

		/** \return the colored mesh shader */
		MeshShadingShader & getMeshShadingShader() { return colored_mesh_shader; }

	protected:

		/** Helper to add some geometry to the view. 
		\param data the object to add
		\param update_raycaster should the associated raycaster be updated with the new geometry
		\return the update object options
		*/
		MeshData & addMeshData(MeshData & data, bool update_raycaster = false);

		/** Create the shaders */
		void initShaders();

		/** Render all the registered meshes. */
		void renderMeshes();

		/** Generate the list of objects in the GUI panel of the view. */
		void list_mesh_onGUI();

		using ListMesh = std::list<MeshData>;
		using Iterator = ListMesh::iterator;
	
		std::string							name; ///< View name.
		ListMesh							list_meshes; ///< Meshes to display.
		Iterator 							selected_mesh_it; ///< Currently selected mesh.
		bool								selected_mesh_it_is_valid = false; ///< Is there a valid currently selected mesh.
		
		InteractiveCameraHandler			camera_handler; ///< View camera handler.

		PointShader							points_shader; ///< Shader for points.
		MeshShadingShader					colored_mesh_shader; ///< Shader for meshes.
		NormalRenderingShader				per_vertex_normals_shader, per_triangle_normals_shader; ///< Shaders for visualizing an object normals.

		Vector3f							backgroundColor = { 0.7f, 0.7f, 0.7f }; ///< Background clear color.
	};

}
