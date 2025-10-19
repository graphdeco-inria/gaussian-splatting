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
  
uniform vec3    light_position;
uniform vec3    user_color;                    
uniform float   alpha;
uniform bool    phong_shading;
uniform bool    use_mesh_color;

in vec3 color;
in vec3 normal;
in vec3 position;            

void main(void) {

    vec3 col;
    if(use_mesh_color){
        col = color;
    } else {
        col = user_color;
    }
    
    out_color = vec4(col, alpha);
    
    if(phong_shading){
        float kd = 0.2;
        float ks = 0.1;
        vec3 L = normalize(light_position - position);	
        vec3 N = normalize(normal);
        vec3 R = - reflect(L,N);
        vec3 V = L;		//light pos = eye
        float diffuse = max(0.0, dot(L,N));
        float specular = max(0.0, dot(R,V));
        out_color.xyz = (1.0 - kd - ks)*col + (kd*diffuse + ks*specular)* vec3(1, 1, 1);
    }  
}
