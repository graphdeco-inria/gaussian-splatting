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

uniform vec3 user_color;                    
uniform float alpha;
in vec3 color;

void main(void) {
//    out_color = vec4(user_color, alpha);
    out_color = vec4(color, alpha);
}
