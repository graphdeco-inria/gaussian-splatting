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

uniform bool hasNormal = true;
out vec4 out_color;

in vec3 color_vert;
in vec3 vertexPos; 
in vec3 normalPos;

void main(void) {
	float kd = 0.8;
	float ks = 0.15;
	float diffuse = 1.0;	
	float specular = 0.0;

	if(hasNormal){
		vec3 L = normalize(lightDir);				
		vec3 N = normalize(normalPos);							
		vec3 R = reflect(L,N);//2.0*dot(L,N)*N - N;						
		vec3 V = L;				
		diffuse = max(0.0, dot(L,N));	
		specular = max(0.0, dot(R,V));
	}
	out_color.rgb = (1.0-kd-ks)*color_vert + kd*diffuse*color_vert + ks*specular;
	out_color.a = 1.0;
	
}
