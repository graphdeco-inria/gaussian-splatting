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
layout(binding = 0) uniform sampler2DArray input_rgbs;

in vec2 out_uv;

uniform float alpha;
uniform int slice;

void main() {
    vec3 uv_cam = vec3(out_uv, slice);
    out_color = vec4(texture(input_rgbs, uv_cam).xyz, alpha);
}
