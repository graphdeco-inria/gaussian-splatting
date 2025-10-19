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

uniform float scale;
uniform vec2 pos;
uniform vec2 size;
uniform bool correctRatio;

out vec2 texcoord;

void main(void) {
	vec2 position = scale * vec2(1.0, correctRatio ? (size.y/size.x) : 1.0) * in_vertex.xy + pos; 
	gl_Position = vec4(in_vertex.xy, 0.0, 1.0);
	texcoord = position * 0.5 + 0.5;
}
