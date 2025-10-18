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

#define NUM_CAMS (12)
#define ULR_STREAMING (0)

in vec2 vertex_coord;
layout(location = 0) out vec4 out_color;

// 2D proxy texture.
layout(binding=0) uniform sampler2D proxy;

// Input cameras.
struct CameraInfos
{
  mat4 vp;
  vec3 pos;
  int selected;
  vec3 dir;
};
// They are stored in a contiguous buffer (UBO), lifting most limitations on the number of uniforms.
layout(std140, binding=4) uniform InputCameras
{
  CameraInfos cameras[NUM_CAMS];
};

// Uniforms.
uniform int camsCount;
uniform vec3 ncam_pos;
uniform bool occ_test = true;
uniform bool invert_mask = false;
uniform bool is_binary_mask = true;
uniform bool discard_black_pixels = true;
uniform bool doMasking = false;
uniform bool flipRGBs = false;
uniform bool showWeights = false;
uniform float epsilonOcclusion = 1e-2;
#define INFTY_W 100000.0
#define BETA 	1e-1  	/* Relative importance of resolution penalty */

// Textures.
// To support both the regular version (using texture arrays) and the streaming version (using 2D RTs),
// we wrap the texture accesses in two helpers that hide the difference.

#if ULR_STREAMING

uniform sampler2D input_rgbds[NUM_CAMS];
uniform sampler2D input_masks[NUM_CAMS];

vec4 getRGBD(vec3 xy_camid){
	if(flipRGBs){
		xy_camid.y = 1.0 - xy_camid.y;
	}
	vec4 rgbd = texture(input_rgbds[int(xy_camid.z)], xy_camid.xy);
	if(flipRGBs){
		xy_camid.y = 1.0 - xy_camid.y;
	}
	return rgbd;
}

float getMask(vec3 xy_camid){
	return texture(input_masks[int(xy_camid.z)], xy_camid.xy).r;
}

#else

layout(binding=1) uniform sampler2DArray input_rgbs;
layout(binding=2) uniform sampler2DArray input_depths;
layout(binding=3) uniform sampler2DArray input_masks;

vec4 getRGBD(vec3 xy_camid){
	if(flipRGBs){
		xy_camid.y = 1.0 - xy_camid.y;
	}
	vec3 rgb = texture(input_rgbs, xy_camid).rgb;
	if(flipRGBs){
		xy_camid.y = 1.0 - xy_camid.y;
	}
	float depth = texture(input_depths, xy_camid).r;
    return vec4(rgb,depth);
}

float getMask(vec3 xy_camid){
	return texture(input_masks, xy_camid).r;
}

#endif

// Helpers.

vec3 project(vec3 point, mat4 proj) {
  vec4 p1 = proj * vec4(point, 1.0);
  vec3 p2 = (p1.xyz/p1.w);
  return (p2.xyz*0.5 + 0.5);
}

bool frustumTest(vec3 p, vec2 ndc, int i) {
  vec3 d1 = cameras[i].dir;
  vec3 d2 = p - cameras[i].pos;
  return !any(greaterThan(ndc, vec2(1.0))) && dot(d1,d2)>0.0;
}

vec3 getRandomColor(int x);

void main(void){
  		
	vec4 point = texture(proxy, vertex_coord);
	  // discard if there was no intersection with the proxy
	if ( point.w >= 1.0) {
		discard;
	}

	vec4  color_sum = vec4(0.0,0.0,0.0,0.0);
	//vec3  color_sum_simple = vec3(0.0,0.0,0.0);
	//vec3  color_sum_square = vec3(0.0,0.0,0.0);
	//vec3  color_sum_square_w = vec3(0.0,0.0,0.0);
	//float num = 0.0;
	
	vec3 v2 = (point.xyz - ncam_pos);
	float dist_n2p 	= length(v2);
	  
	  for(int i = 0; i < NUM_CAMS; i++){
		if(i>=camsCount){
			continue;
		}
		if(cameras[i].selected == 0){
			break;
		}
		
		vec3 uvd = project(point.xyz, cameras[i].vp);
		vec2 ndc = abs(2.0*uvd.xy-1.0);
		
		if (!frustumTest(point.xyz, ndc, i)) {
			continue;
		}
		
		vec3 xy_camid = vec3(uvd.xy,i);
		vec4 color = getRGBD(xy_camid);
		
		// Support output weights as random colors for debug.
		if(showWeights){
			color.xyz = getRandomColor(i);
		}

		if(doMasking){	
			float masked = getMask(xy_camid);
			 
			if( invert_mask ){
				masked = 1.0 - masked;
			}
			
			if( is_binary_mask ){
				if( masked < 0.5) {
					continue;
				}
			} else {
				color.xyz = masked*color.xyz;
			}	
		}		

		if (discard_black_pixels){
			if(all(equal(color.xyz, vec3(0.0)))){
				continue;
			}
		}
		
		if (occ_test){
			if(abs(uvd.z-color.w) >= epsilonOcclusion) {	  
				continue;
			}
		} 
		
		vec3 v1 = (point.xyz - cameras[i].pos);
		float dist_i2p 	= length(v1);
		
		float penalty_ang = float(occ_test) * max(0.0001, acos(dot(v1,v2)/(dist_i2p*dist_n2p)));

		float penalty_res = max(0.0001, (dist_i2p - dist_n2p)/dist_i2p );
		 
		color.w = penalty_ang + BETA*penalty_res;
		  
		color.w = 1.0 / color.w;
		
		color_sum.xyz += color.w * color.xyz;
		color_sum.w += color.w;
		
		//color_sum_simple += color.xyz;
		//color_sum_square += color.xyz*color.xyz;
		//color_sum_square_w += color.w * color.xyz*color.xyz;
		//num++;
	}
	
	//vec3 mean = color_sum_simple / num;
	//vec3 variance = color_sum_square / num - mean*mean;
	//vec3 deviation = sqrt(variance);
	
	//vec3 mean = color_sum.xyz / color_sum.w;
	//vec3 variance = color_sum_square_w / color_sum.w - mean*mean;
	//vec3 deviation = 3.0*sqrt(variance);
	
	color_sum.xyz /= color_sum.w;
	
    // blending
    out_color.w = 1.0;
    out_color.xyz = color_sum.xyz;
	//out_color.xyz = deviation;
	
	gl_FragDepth = point.w;
}


// Random number generation:
// "Quality hashes collection" (https://www.shadertoy.com/view/Xt3cDn)
// by nimitz 2018 (twitter: @stormoid)
// The MIT License
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/** Compute the based hash for a given index.
	\param p the index
	\return the hash
*/
uint baseHash(uint p) {
	p = 1103515245U*((p >> 1U)^(p));
	uint h32 = 1103515245U*((p)^(p>>3U));
	return h32^(h32 >> 16);
}

/** Generate a random vec3 from an index seed (see http://random.mat.sbg.ac.at/results/karl/server/node4.html).
	\param x the seed
	\return a random vec3
*/
vec3 getRandomColor(int x) {
	uint n = baseHash(uint(x));
	uvec3 rz = uvec3(n, n*16807U, n*48271U);
	return vec3(rz & uvec3(0x7fffffffU))/float(0x7fffffff);
}
