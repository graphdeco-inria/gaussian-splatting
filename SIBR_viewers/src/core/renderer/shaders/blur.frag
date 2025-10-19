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

uniform sampler2D 	image;
uniform vec2  		in_image_size;


in vec2 tex_coord;


void main(void) {

	vec4 color = texture(image, tex_coord);	
	
	const float blurPixelStep = 1.25;
	const float blurSizeX = (1.0 / in_image_size.x)*blurPixelStep;
	const float blurSizeY = (1.0 / in_image_size.y)*blurPixelStep;
	
	const int nbVisit = 4;
	ivec2 visit[nbVisit] = ivec2 [](
	// cross
	// ivec2( 0, -1),
	// ivec2( 0,  1),
	// ivec2( 1,  0),
	// ivec2(-1,  0)
	
	// corner (better, detect more edge)
	ivec2(-1, -1),
	ivec2(-1,  1),
	ivec2( 1, -1),
	ivec2( 1,  1)
	);
	
	vec4 avcolor;
	vec4 bgColor = vec4(color.xyz, 0.0);
	for (int i = 0; i < nbVisit; ++i)
	{
		vec4 col = texture( image, 
		vec2(tex_coord.x + visit[i].x* blurSizeX, tex_coord.y + visit[i].y * blurSizeY) );
		
		bgColor = (col.a > bgColor.a)? col : bgColor;
		avcolor += col;
	}
	
	avcolor /= float(nbVisit);
	
	float dColor = abs(avcolor.a - color.a);//length(avcolor - color);
	const float maxDColor = 0.015f;
	float mixFactor = 1.0 - min(1.0, dColor/maxDColor);
	vec4 correction = bgColor;//mix(bgColor, avcolor, mixFactor);

	// [USEFUL FOR DEBUGGING]: show detected edge
	//out_color = (dColor > maxDColor)? vec4(1.0, 0.0, 1.0, 1.0) : color;

	out_color = (dColor > maxDColor)? mix(correction, color, 0.5) : color;

	// [USEFUL FOR DEBUGGING]: change nothing
	//out_color = color;

}
