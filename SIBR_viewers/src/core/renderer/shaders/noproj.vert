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


/** \file ibr.vp
 *
 * Vertex shader WITHOUT projection and modelview transformations.
 */

#version 420

layout(location = 0) in vec4 in_vertex;   /**< Input vertex coordinates */
layout(location = 1) in vec4 in_texcoord; /**< Input texture coordinates */
layout(location = 2) in vec4 in_color;    /**< Input colour value */

out vec4 texcoord;                        /**< Output texture coordinates */
out vec4 color;                           /**< Output color value */

void main(void) {
  gl_Position = in_vertex;
  texcoord    = in_texcoord;
  color       = in_color;
}
