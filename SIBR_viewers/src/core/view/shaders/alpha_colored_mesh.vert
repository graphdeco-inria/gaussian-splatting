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

layout(location = 0) in vec3 in_vertex;   
layout(location = 1) in vec3 in_color; 
layout(location = 3) in vec3 in_normal;  
  
uniform mat4 mvp;    
          
out vec3 color;
out vec3 normal;
out vec3 position;                      

void main(void) {
    gl_Position = mvp * vec4(in_vertex, 1.0);
    color = in_color;
    normal = in_normal;
    position = in_vertex;
}
