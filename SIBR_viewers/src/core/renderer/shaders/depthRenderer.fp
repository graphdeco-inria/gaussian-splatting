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

out float out_depth;

void main(void) {
	
    out_depth = 2.0*gl_FragCoord.z-1.0;
    //out_color = fragTexCoord.x*vec4(1.0,0.0,0.0,1.0);
}

