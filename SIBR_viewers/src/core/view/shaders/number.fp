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

in vec2 uv_coord;

uniform float value;
uniform int count;

out vec4 out_color;

const float digits[10] = float[](0x69996,0x26222,0x6924F,0x69396,0x99F11,0xF861E,0x68E96,0xF1248,0x69696,0x69716);

float printDigit(int digit, vec2 position){
	// Margin scaling/shift
	position *= 1.4;
	position -= 0.2;
	// Early discard.
	if(position.x < 0.0 || position.x > 1.0 || position.y < 0.0 || position.y > 1.0){
		return 0.0;
	}
	// [0,1] -> discrete[0,4]x[0,5]
	vec2 newPos = floor(vec2(4.0-4.0*position.x,5.0*position.y));
	// -> corresponding bit
	float index = 4*newPos.y + newPos.x;
	// -> get the index-th bit
	float isIn = mod(floor(digits[digit]/pow(2.0,index)),2.0);
	return isIn;
}

float printPoint(vec2 position){
	position *= 1.4;
	position -= 0.02;
	if(position.x < 0.0 || position.x > 1.0 || position.y < 0.0 || position.y > 1.0){
		return 0.0;
	}
	return length(position - vec2(0.2, 0.4)) < 0.182 ? 1.0 : 0.0;
	
}

void main(void) {
	float deca = printDigit(int(mod(value/10,10)), uv_coord);
	float unit = printDigit(int(mod(value,10)), uv_coord-vec2(1.0,0.0));
	float deci = printDigit(int(mod(value*10,10)), uv_coord-vec2(2.5,0.0));
	float centi = printDigit(int(mod(value*100,10)), uv_coord-vec2(3.5,0.0));
	float point = printPoint(uv_coord-vec2(2.0,0.0));
	float color = clamp(deca+unit+deci+centi+point,0.0,1.0);
  	out_color = vec4(color,color, color, 1.0);
}
