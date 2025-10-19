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

in vec3 GtoF_normal;
layout(location = 0) out vec4 out_color;

void main(void) {
	
	vec3 colorN=(GtoF_normal+1.0)/2.0;
    out_color = vec4(colorN,1.0);
    //out_color = fragTexCoord.x*vec4(1.0,0.0,0.0,1.0);
}

