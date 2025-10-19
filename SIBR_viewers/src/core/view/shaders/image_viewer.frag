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

in vec2 texcoord;

layout(binding = 0) uniform sampler2D in_texture;

uniform vec4 minVal = vec4(0.0);
uniform vec4 maxVal = vec4(1.0);
uniform vec4 channels = vec4(1.0);

layout(location = 0) out vec4 out_color;

void main(void)
{
	if(any(greaterThan(texcoord, vec2(1.0))) || any(lessThan(texcoord, vec2(0.0)))){
		discard;
	}
	
	vec4 col = texture(in_texture, texcoord);
	// Rescale.
	out_color = channels*(col - minVal)/(maxVal - minVal);

	// If only one channel is enabled, no alpha and B&W image.
	if(dot(channels, vec4(1.0)) == 1.0){
		float val = dot(out_color, channels);
		out_color.rgb = vec3(val);
		out_color.a = 1.0;
	}

	// Ensure visibility when alpha is disabled.
	if(channels[3] == 0.0f){
		out_color.a;
	}
}
