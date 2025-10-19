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
# include <string>
# include "core/graphics/Config.hpp"
# include "core/system/Matrix.hpp"

#define SIBR_SHADER(version, shader)  std::string("#version " #version "\n" #shader)

namespace sibr
{
	/**
	  OpenGL shader wrapper.
	* \ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT GLShader
	{
		SIBR_DISALLOW_COPY( GLShader );
		SIBR_CLASS_PTR(GLShader);
	public:

		/** Macro-like substitution in shaders. */
		struct Define
		{
			typedef std::vector<Define>	List;

			/** Constructor.
			\param nameToSearch_ the macro value to replace
			\param valueToSet_ the replacement value (stringified)
			*/
			template <typename TValue>
			Define( const std::string& nameToSearch_, const TValue& valueToSet_ ) :
				nameToSearch(nameToSearch_) {
					std::ostringstream oss;
					oss << valueToSet_;
					valueToSet = oss.str();
			}

			std::string nameToSearch; ///< the macro value to replace
			std::string valueToSet; ///< String representation of the replacement value.
		};

	public:

		/// Constructor.
		GLShader( void );

		/// Destructor.
		~GLShader( void );

		/** Create and compile a GPU program composed of a vertex/fragment shader (and optionally geometry/tesselation shaders).
		\param name the name of the shader (for logging)
		\param vp_code vertex shader code string
		\param fp_code fragment shader code string
		\param gp_code geometry shader code string
		\param exitOnError should the application exit on a shader compilation error
		\param tcs_code tesselation control shader code string
		\param tes_code tesselation evaluation shader code string
		\return a success flag
		*/
		bool init(std::string name,
			std::string vp_code, std::string fp_code,
			std::string gp_code = std::string(),
			bool exitOnError = true,
			std::string tcs_code = std::string(),
			std::string tes_code = std::string());

		/** Recompile a GPU program with updated shaders.
		\param vp_code vertex shader code string
		\param fp_code fragment shader code string
		\param gp_code geometry shader code string
		\return a success flag
		*/
		bool reload(
			std::string vp_code, std::string fp_code,
			std::string gp_code = std::string());

		/**
		Query the dissassembly of the shader program.
		\param binary will contain the compiled shader code
		\note This is not supported on all GPUs.
		*/
		void getBinary(std::vector<char> & binary);

		/** Bind (activate) the sahder for rendering. */
		SIBR_OPT_INLINE		void	begin( void );

		/** Unbind the shader. */
		SIBR_OPT_INLINE		void	end( void );

		/** Init from an existing GPU program.
		\param s_handle the existing program handle
		*/
		void			init      ( GLuint s_handle );

		/** Cleanup and delete the program. */
		void			terminate( void );

		/** If set to true, uniforms that are linked but not referenced 
		by the shader will cause an error to be raised.
		\param s the validation level
		*/
		void			setStrict ( bool s );

		/** \return the program handle. */
		GLuint			shader    ( void )  const;

		/** \return the shader name. */
		std::string		name      ( void )  const;

		/** \return true if the shader is properly setup. */
		bool			isReady   ( void )  const;

		/** \return true if the shader is currently bound for drawing. */
		bool			isActive  ( void )  const;

		/** \return true if the shader will validate linked uniforms. */
		bool			isStrict  ( void )  const;

	private:

		/** Compile a shader for a given stage.
		\param shader_code the string containing the sahder code
		\param type the stage to compile for
		\return the compiled shader stage handle.
		*/
		GLuint	compileShader( const char* shader_code, GLuint type );

		/** Check if the shader is properly setup, or raise an error. */
		SIBR_OPT_INLINE		void	authorize( void ) const;

		GLuint      m_Shader; ///< Shader program handle.
		std::string m_Name; ///< Shader name.
		bool        m_Strict; ///< Should uniforms be validated.
		bool        m_Active; ///< Is the shader currently bound.
	};

	// ------------------------------------------------------------------------


	/**
	  OpenGL shader uniform wrapper. Prefer using GLuniform instead.
	  \sa GLuniform
	* \ingroup sibr_graphics
	*/
	class SIBR_GRAPHICS_EXPORT GLParameter
	{
	public:

		/// Constructor.
		GLParameter( void );

		/** Link the uniform to a shader.
		\param shader the shader to link to
		\param name the name of the uniform in the shader
		*/
		void	init( sibr::GLShader& shader, std::string name );

		/** \return true if the uniform was linked to a shader. */
		bool	isInitialized( void ) const;

		/** \return the OpenGL uniform location handle. */
		GLint	handle( void ) const;

		/** Set the uniform float value.
		\param f the new value
		*/
		SIBR_OPT_INLINE void	set( float f );

		/** Set the uniform vec2 value.
		\param a first component
		\param b second component
		*/
		SIBR_OPT_INLINE void	set( float a, float b );
		
		/** Set the uniform vec3 value.
		\param a first component
		\param b second component
		\param c third component
		*/
		SIBR_OPT_INLINE void	set( float a, float b, float c );
		
		/** Set the uniform vec4 value.
		\param a first component
		\param b second component
		\param c third component
		\param d fourth component
		*/
		SIBR_OPT_INLINE void	set( float a, float b, float c, float d );
		
		/** Set the uniform mat4 value.
		\param matrix the 16 matrix components, in row order
		*/
		SIBR_OPT_INLINE void	set( const float *matrix );

		/** Set the uniform sampler value.
		\param tex the new value (ie the binding location of the texture)
		*/
		SIBR_OPT_INLINE void	set( GLuint tex );

		/** Set the uniform integer value.
		\param v the new value
		*/
		SIBR_OPT_INLINE void	set( int v );

		/** Set the uniform boolean value (converted to an int).
		\param b the new value
		*/
		SIBR_OPT_INLINE void	set( bool b);

		/** Set the uniform values defined as an array of floats.
		\param pv pointer to the float array
		\param size number of elements
		*/
		SIBR_OPT_INLINE void	setArray( const float *pv, int size );

		/** Set the uniform values defined as an array of integers.
		\param pv pointer to the int array
		\param size number of elements
		*/
		SIBR_OPT_INLINE void	setArray( const std::vector<int>& pv, int size );

		/** Set the uniform ivec2 value.
		\param v the new value
		*/
		SIBR_OPT_INLINE void	set(const Vector2i& v);

		/** Set the uniform ivec3 value.
		\param v the new value
		*/
		SIBR_OPT_INLINE void	set(const Vector3i& v);

		/** Set the uniform ivec4 value.
		\param v the new value
		*/
		SIBR_OPT_INLINE void	set(const Vector4i& v);
		
		/** Set the uniform vec2 value.
		\param v the new value
		*/
		SIBR_OPT_INLINE void	set( const Vector2f& v );

		/** Set the uniform vec3 value.
		\param v the new value
		*/
		SIBR_OPT_INLINE void	set( const Vector3f& v );

		/** Set the uniform vec4 value.
		\param v the new value
		*/
		SIBR_OPT_INLINE void	set( const Vector4f& v );

		/** Set the uniform mat4 value.
		\param m the new value
		*/
		SIBR_OPT_INLINE void	set( const Matrix4f& m );

		/** Set the uniform values defined as an array of mat4s  (row major).
		\param m pinter to the beginning of the matrix array
		\param num number of matrices
		*/
		SIBR_OPT_INLINE void setMatrixArray(const float * m, int num);

	private:

		sibr::GLShader*		m_Shader; ///< Linked shader.
		GLint				m_Handle; ///< Uniform location.
		std::string			m_Name; ///< Uniform name.
		bool				m_Strict; ///< Should the program raise an error if the uniform is not found in the linked shader.
		
		/** Check if the uniform/shader link is valid. */
		SIBR_OPT_INLINE void	authorize( void ) const;
	};

	/** Load a file from disk and apply macro-like substitutions.
	\param filename the file path
	\param defines a list of substitutions to apply
	\return the loaded string
	\ingroup sibr_graphics
	*/
	SIBR_GRAPHICS_EXPORT std::string	loadFile( const std::string& filename, const GLShader::Define::List& defines );

	/**
	  OpenGL shader uniform wrapper with additional update/storage functions.
	  It will behave as an element of the type it wraps, but can also be set/sent to the GPU.
	  This limits duplication, as you can replace a float+GLParameter by a GLuniform<float>
	  When you need a reference to the value (for instance in imGui), use uniform.get().
	  \sa GLParameter
	* \ingroup sibr_graphics
	*/
	template<typename T> class GLuniform {

	public:
		/** Constructor.
		\param t initial value to use
		*/
		GLuniform(const T & t) : value(t) {}

		/** \return a reference to the value. */
		operator T & () { return value; }

		/** \return a reference to the value. */
		T & get() { return value; }

		/** Copy operator. Update the stored value.
		\param t the new value
		\return a reference to the value.
		*/
		T & operator=(const T & t) { value = t; return *this; }
		
		/** Copy operator. Update the stored value using the one of the other uniform.
		\param other uniform to get the new value from
		\return a reference to itself
		*/
		GLuniform & operator=(const GLuniform& other) { value = other.value; return *this; };

		/** not-equal-to operator. Compares the stored values with the argument and returns
		the not-equal-to operator.
		\param t value to compare to.
		\return the boolean result of the operation.
		*/
		bool operator!=(const T& t) { return value != t; }

		/** Copy constructor. Update the stored value using the one of the other uniform.
		\param other uniform to get the new value from
		*/
		explicit GLuniform(const GLuniform&other) : value(other.value) { };

		/// Default constructor.
		GLuniform() = default;

		/** Link the uniform to a shader.
		\param shader the shader to link to
		\param name the name of the uniform in the shader
		*/
		void init(sibr::GLShader& shader, std::string name) {
			parameter.init(shader, name);
		}

		/** Send the value to the shader if it was initialized.
		\note the shader has to be active
		*/
		void send() {
			if (parameter.isInitialized()) {
				parameter.set(value);
			}
		}

		/** Set the value and send it to the shader if it was initialized.
		\param t the new value
		\note the shader has to be active
		*/
		void set(const T & t) {
			value = t;
			send();
		}
		
		/** Set a list of values and send it to the shader if it was initialized.
		\param t the new values
		\param size the number of values
		\note the shader has to be active
		*/
		void setArray(const T & t,int size) {
			value = t;
			if (parameter.isInitialized()) {
				parameter.setArray(value,size);
			}
		}

	protected :
		T value = T(); ///< The underlying value.
		sibr::GLParameter parameter; ///< The underlying uniform.
	};

} // namespace sibr

# ifdef SIBR_MAXIMIZE_INLINE
#  include "Shader.inl"
# endif
