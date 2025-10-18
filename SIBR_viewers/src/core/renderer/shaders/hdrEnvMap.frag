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

uniform sampler2D tex;

out vec4 out_color;

in vec2 vertUV;

float lin2sRGBF(float inF){
	
	if(inF<0.0031308){
		return 12.92*inF;
	}
	else{
		return 1.055*pow(inF,1.0/2.4)-0.055;
	}
	
}

vec4 lin2sRGB(vec4 inVec){
	return vec4(lin2sRGBF(inVec.x),lin2sRGBF(inVec.y),lin2sRGBF(inVec.z),inVec.w);
}

void main(void) {
	vec2 uv = vertUV;
	uv.y = 1.0 - uv.y; /// \todo TODO: Why Texture are flipped in y ?
	out_color = lin2sRGB(texture(tex, uv));
}