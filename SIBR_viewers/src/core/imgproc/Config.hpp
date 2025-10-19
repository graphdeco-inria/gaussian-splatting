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

# ifdef SIBR_OS_WINDOWS
//// Export Macro (used for creating DLLs) ////
#  ifdef SIBR_STATIC_DEFINE
#    define SIBR_EXPORT
#    define SIBR_NO_EXPORT
#  else
#    ifndef SIBR_IMGPROC_EXPORT
#      ifdef SIBR_IMGPROC_EXPORTS
          /* We are building this library */
#        define SIBR_IMGPROC_EXPORT __declspec(dllexport)
#      else
          /* We are using this library */
#        define SIBR_IMGPROC_EXPORT __declspec(dllimport)
#      endif
#    endif
#    ifndef SIBR_NO_EXPORT
#      define SIBR_NO_EXPORT 
#    endif
#  endif
# else
# define SIBR_IMGPROC_EXPORT
# endif



