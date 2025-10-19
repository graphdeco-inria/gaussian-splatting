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


/**
 * \file topview.fp
 *
 * Basic shader to render textured/colored geometry
 */

#version 420

layout(binding = 0) uniform sampler2D tex;/**< Input texture */
layout(location= 0) out vec4 out_color;   /**< Output texture map */

uniform vec4 in_color;                    /**< Uniform color */
in vec4 texcoord;                         /**< Texture coords at current pixel */

void main(void) {
  vec4 c1 = texture(tex,texcoord.xy);
  vec4 c2 = in_color;
  out_color = c2.a*in_color + clamp(1.0-c2.a,0.0,1.0)*c1;
}
