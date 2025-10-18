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

# include "core/system/Config.hpp"
# include "core/system/Utils.hpp"

//#define GLEW_STATIC
#include <GL/glew.h>

# include <functional>

# define GLFW_INCLUDE_GLU
# include <GLFW/glfw3.h>


// (used by Image)
# pragma warning(push, 0)
#  include <opencv2/opencv.hpp>
#  include <opencv2/core.hpp>
#  include <opencv2/highgui.hpp>
# pragma warning(pop)



# ifdef SIBR_OS_WINDOWS
//// Export Macro (used for creating DLLs) ////
#  ifdef SIBR_STATIC_DEFINE
#    define SIBR_EXPORT
#    define SIBR_NO_EXPORT
#  else
#    ifndef SIBR_GRAPHICS_EXPORT
#      ifdef SIBR_GRAPHICS_EXPORTS
          /* We are building this library */
#        define SIBR_GRAPHICS_EXPORT __declspec(dllexport)
#      else
          /* We are using this library */
#        define SIBR_GRAPHICS_EXPORT __declspec(dllimport)
#      endif
#    endif
#    ifndef SIBR_NO_EXPORT
#      define SIBR_NO_EXPORT 
#    endif
#  endif
# else
# define SIBR_GRAPHICS_EXPORT
# endif


/** Macro to check OpenGL error and throw \p std::runtime_error if found */
# undef CHECK_GL_ERROR
# define CHECK_GL_ERROR  {								\
  GLenum err = glGetError();							\
  if (err) {											\
	std::string errorStr = "Unknown";					\
	switch (err) {										\
	case GL_INVALID_ENUM:								\
		errorStr = "GL_INVALID_ENUM";					\
		break;											\
	case GL_INVALID_VALUE:								\
		errorStr = "GL_INVALID_VALUE";					\
		break;											\
	case GL_INVALID_OPERATION:							\
		errorStr = "GL_INVALID_OPERATION";				\
		break;											\
	case GL_STACK_OVERFLOW:								\
		errorStr = "GL_STACK_OVERFLOW";					\
		break;											\
	case GL_STACK_UNDERFLOW:							\
		errorStr = "GL_STACK_UNDERFLOW";				\
		break;											\
	case GL_OUT_OF_MEMORY:								\
		errorStr = "GL_OUT_OF_MEMORY";					\
		break;											\
	case GL_INVALID_FRAMEBUFFER_OPERATION:				\
		errorStr = "GL_INVALID_FRAMEBUFFER_OPERATION";	\
		break;											\
	default:											\
		break;											\
	}													\
  SIBR_ERR << "OpenGL error 0x0" << std::hex << err  << std::dec  << " (" << int(err) << ") " << errorStr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
  }														\
}

#define SIBR_GLSL(version, shader)  "#version " #version "\n" #shader


namespace sibr
{
	/** Clamp a value.
	\param value value to clamp
	\param min min value
	\param max max value
	\return min(max(value, min), max)
	\ingroup sibr_graphics
	*/
	template <typename T>
	inline T	clamp( T value, T min, T max ) {
		return std::max(min, std::min(max, value));
	}

} // namespace sibr

