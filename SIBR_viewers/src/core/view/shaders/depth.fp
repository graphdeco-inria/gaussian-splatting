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


#version 420

layout(location = 0) out vec4 out_color;
layout(binding  = 0) uniform sampler2D image; /// \todo TODO: remove

//in vec3 vertex_coord;
//uniform vec3 iCamPos;

uniform vec2 size;

void main(void) {
  vec2 tC       = gl_FragCoord.xy / size;
  out_color.xyz = vec3(0.0);//texture(image, tC.xy).xyz;
  out_color.w   = gl_FragCoord.z;
  //out_color.w   = distance(vertex_coord, iCamPos);
}
