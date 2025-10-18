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

in vec3 position;
in vec3 normal;

uniform vec3 cameraPos;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec3 outDirection;

void main(void) {
	outPosition = position;
	outDirection = reflect(normalize(position - cameraPos), normal);
}

