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

uniform mat4 MVPinv;

layout(triangles) in;
layout (triangle_strip) out;
layout (max_vertices = 3) out;

out vec3 GtoF_normal;

void main(void) {
  
  vec3 P0 = (MVPinv *gl_in[0].gl_Position).xyz;
  vec3 P1 = (MVPinv *gl_in[1].gl_Position).xyz;
  vec3 P2 = (MVPinv *gl_in[2].gl_Position).xyz;
  
  vec3 V0 = P0 - P1;
  vec3 V1 = P2 - P1;
  
  vec3 N = normalize( cross(V1, V0) );

  int i;

    for (i = 0; i < gl_in.length(); i++)
    {
		GtoF_normal = N; // Specs say we need to set again output values after calling EmitVertex
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();

}