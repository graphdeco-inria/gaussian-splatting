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

layout (location = 0) out vec4 fragColor;

in INTERFACE {
	vec4 col;
	vec2 uv;
} In ;

uniform sampler2D tex;

void main(){
	fragColor = In.col * texture(tex, In.uv);
}
