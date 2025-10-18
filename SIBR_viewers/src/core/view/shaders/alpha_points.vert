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

uniform mat4 mvp;    
uniform int radius;
out vec3 color;

void main(void) {
    gl_Position = mvp * vec4(in_vertex, 1.0);
    gl_PointSize = radius;
    color = in_color;
}
