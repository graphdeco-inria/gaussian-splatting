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

# include <core/graphics/Config.hpp>


# ifdef SIBR_OS_WINDOWS
#  ifdef SIBR_STATIC_RAYCASTER_DEFINE
#    define SIBR_RAYCASTER_EXPORT
#    define SIBR_NO_RAYCASTER_EXPORT
#  else
#    ifndef SIBR_RAYCASTER_EXPORT
#      ifdef SIBR_RAYCASTER_EXPORTS
          /* We are building this library */
#        define SIBR_RAYCASTER_EXPORT __declspec(dllexport)
#      else
          /* We are using this library */
#        define SIBR_RAYCASTER_EXPORT __declspec(dllimport)
#      endif
#    endif
#  endif
# else
#  define SIBR_RAYCASTER_EXPORT
# endif

