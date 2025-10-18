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

# include "core/graphics/Config.hpp"
# include <iomanip>


#ifdef SIBR_OS_WINDOWS
//// Export Macro (used for creating DLLs) ////
# ifdef SIBR_STATIC_DEFINE
#   define SIBR_EXPORT
#   define SIBR_NO_EXPORT
# else
#   ifndef SIBR_SCENE_EXPORT
#     ifdef SIBR_SCENE_EXPORTS
         /* We are building this library */
#       define SIBR_SCENE_EXPORT __declspec(dllexport)
#     else
         /* We are using this library */
#       define SIBR_SCENE_EXPORT __declspec(dllimport)
#     endif
#   endif
#   ifndef SIBR_NO_EXPORT
#     define SIBR_NO_EXPORT
#   endif
# endif
# else
#  define SIBR_SCENE_EXPORT
# endif

