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

uniform mat4 MVP;

layout(location = 0) in vec3 in_vertex;
layout(location = 3) in vec3 in_normal;

out vec3 position;
out vec3 normal;

void main(void) {
	gl_Position = MVP * vec4(in_vertex,1.0);
    position = in_vertex;
    normal = in_normal;
}
