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

layout(triangles) in;
layout(line_strip, max_vertices = 2) out;

uniform mat4 mvp;
uniform float normals_size;

out vec3 color;
out vec3 normal;
out vec3 position;

void main(void) {
	vec3 a = gl_in[0].gl_Position.xyz;
	vec3 b = gl_in[1].gl_Position.xyz;
	vec3 c = gl_in[2].gl_Position.xyz;

	vec3 tri_normal = normalize(cross(b-a,c-b));
	vec3 tri_center = (a+b+c)/3.0;
	gl_Position = mvp*vec4(tri_center,1.0);
	color = vec3(0.0);
	normal = vec3(0.0);
	position = vec3(0.0);
	EmitVertex();
	gl_Position = mvp*vec4(tri_center + normals_size*tri_normal, 1.0);
	color = vec3(0.0);
	normal = vec3(0.0);
	position = vec3(0.0);
	EmitVertex();
	EndPrimitive();
}
