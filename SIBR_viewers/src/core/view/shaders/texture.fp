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

layout(binding = 0) uniform sampler2D tex;
layout(location= 0) out vec4 out_color;

in vec2 tex_coord;

void main(void) {
    vec2 texcoord = tex_coord ;
    out_color = texture(tex,texcoord);
}
