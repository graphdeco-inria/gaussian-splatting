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

uniform vec3 lightDir;
uniform vec3 cameraEye;

layout(binding=0) uniform sampler2D tex;
layout(binding=1) uniform sampler2D smTex;


out vec4 out_color;

in vec2 vertUV;
in float vertAO;
in vec3 VtoF_normal;
in vec3 vertPos;

float sRGB2LinF(float inF){
	if(inF<0.04045){
		return inF/12.92;
	}
	else{
		return pow((inF+0.055)/(1.055),2.4);
	}
}

float lin2sRGBF(float inF){
	
	if(inF<0.0031308){
		return 12.92*inF;
	}
	else{
		return 1.055*pow(inF,1.0/2.4)-0.055;
	}
	
}

vec4 sRGB2Lin(vec4 inVec){
	return vec4(sRGB2LinF(inVec.x),sRGB2LinF(inVec.y),sRGB2LinF(inVec.z),inVec.w);
}

vec4 lin2sRGB(vec4 inVec){
	return vec4(lin2sRGBF(inVec.x),lin2sRGBF(inVec.y),lin2sRGBF(inVec.z),inVec.w);
}

float getFogFactor(float d)
{
    const float FogMax = 70.0;
    const float FogMin = 10.0;

    if (d>=FogMax) return 1;
    if (d<=FogMin) return 0;

    return 1 - (FogMax - d) / (FogMax - FogMin);
}


void main(void) {
	vec2 uv = vertUV;
	uv.y = 1.0 - uv.y; /// \todo TODO: Why Texture are flipped in y ?

	vec4 sky_color_lin = vec4(0.7,1.0,1.2,1.0)*sRGB2Lin(texture(tex, uv));

	float shadowVal=texture(smTex,gl_FragCoord.xy/textureSize(smTex,0).xy).x;

	vec4 sun_color_lin = max(0.0,dot(VtoF_normal,lightDir))*shadowVal*sRGB2Lin(texture(tex, uv))/(vertAO+0.0001);

	float d = distance(cameraEye, vertPos);
    float alpha = getFogFactor(d);

    vec4 composed_color = lin2sRGB(sun_color_lin+0.1*sky_color_lin);

    out_color = mix(composed_color, vec4(0.5,0.5,0.5,1.0), alpha);
}

