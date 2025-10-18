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
uniform bool gammaCorrection = false;
uniform float epsilonOcclusion = 1e-2;


#define INFTY_W 100000.0
/* Relative importance of resolution penalty */
#define BETA 	1e-1
/* Relative importance of edges penalty */
#define BETA_UV 0.0

// Textures.
// To support both the regular version (using texture arrays) and the streaming version (using 2D RTs),
// we wrap the texture accesses in two helpers that hide the difference.

layout(binding=1) uniform sampler2DArray input_rgbs;
layout(binding=2) uniform sampler2DArray input_depths;
layout(binding=3) uniform sampler2DArray input_masks;


float getMask(vec3 xy_camid){
	return texture(input_masks, xy_camid).r;
}


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

  vec4  color0 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4  color1 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4  color2 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4  color3 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4 masks = vec4(1.0);
  for(int i = 0; i < NUM_CAMS; i++){
	if(i>=camsCount){
		continue;
	}
	if(cameras[i].selected == 0){
		continue;
	}

	vec3 uvd = project(point.xyz, cameras[i].vp);
	vec2 ndc = abs(2.0*uvd.xy-1.0);

	if (frustumTest(point.xyz, ndc, i)){
		vec3 xy_camid = vec3(uvd.xy,i);
		
		float inputDepth = texture(input_depths, xy_camid).r;

		if (occ_test){
			if(abs(uvd.z-inputDepth) >= epsilonOcclusion) {	  
				continue;
			}
		}

		

		float masked = 1.0;
		if(doMasking){        
			masked = getMask(xy_camid);
             
            if( invert_mask ){
                masked = 1.0 - masked;
            }
            
            if( is_binary_mask ){
                if( masked < 0.5) {
                    continue;
                }
			}
		}
		
		float penaltyValue = 0;

		
		// classic ulr
		vec3 v1 = (point.xyz - cameras[i].pos);
		vec3 v2 = (point.xyz - ncam_pos);
		float dist_i2p 	= length(v1);
		float dist_n2p 	= length(v2);

		float penalty_ang = float(occ_test) * max(0.0001, acos(dot(v1,v2)/(dist_i2p*dist_n2p)));

		float penalty_res = max(0.0001, (dist_i2p - dist_n2p)/dist_i2p );
		 
		vec2 fc = vec2(1.0) - smoothstep(vec2(0.7), vec2(1.0), abs(2.0*uvd.xy-1.0));
    	float penalty_uv = 1.0 - fc.x * fc.y; 

		penaltyValue = penalty_ang + BETA*penalty_res + BETA_UV*penalty_uv;
		


		vec4 color = vec4(xy_camid, penaltyValue);
		if(flipRGBs){
			color.y = 1.0 - color.y;
		}
		 
		// compare with best four candiates and insert at the
		// appropriate rank
		if (color.w<color3.w) {    // better than fourth best candidate
			if (color.w<color2.w) {    // better than third best candidate
				color3 = color2;
				if (color.w<color1.w) {    // better than second best candidate
					color2 = color1;
					if (color.w<color0.w) {    // better than best candidate
						color1 = color0;
						color0 = color;
						masks.x = masked;
					} else {
						color1 = color;
						masks.y = masked;
					}
				} else {
					color2 = color;
					masks.z = masked;
				}
			} else {
				color3 = color;
				masks.w = masked;
			}
		}
	 }  
   }
   


	if(color0.w == INFTY_W){
		discard;
	}

	float thresh = 1.0000001 * color3.w;
    color0.w = max(0, 1.0 - color0.w/thresh);
    color1.w = max(0, 1.0 - color1.w/thresh);
    color2.w = max(0, 1.0 - color2.w/thresh);
    color3.w = 1.0 - 1.0/1.0000001;

    // ignore any candidate which is uninit
	if (color0.w == INFTY_W) color0.w = 0;
    if (color1.w == INFTY_W) color1.w = 0;
    if (color2.w == INFTY_W) color2.w = 0;
    //if (color3.w == INFTY_W) color3.w = 0; uneeded, color3.w = 1.0 - 1.0/1.0000001
	
	// Support output weights as random colors for debug.
	if(showWeights){
		color0.rgb = getRandomColor(int(color0.z));
		color1.rgb = getRandomColor(int(color1.z));
		color2.rgb = getRandomColor(int(color2.z));
		color3.rgb = getRandomColor(int(color3.z));
	} else {
		// Read from textures and apply masking.
		color0.rgb = masks.x*texture(input_rgbs, color0.rgb).rgb;
		color1.rgb = masks.y*texture(input_rgbs, color1.rgb).rgb;
		color2.rgb = masks.z*texture(input_rgbs, color2.rgb).rgb;
		color3.rgb = masks.w*texture(input_rgbs, color3.rgb).rgb;
	}

    // blending
    out_color.w = 1.0;
    out_color.xyz = (color0.w*color0.xyz +
             color1.w*color1.xyz +
             color2.w*color2.xyz +
             color3.w*color3.xyz
            ) / (color0.w + color1.w + color2.w + color3.w);
    gl_FragDepth = point.w;

	if(gammaCorrection){
		out_color.xyz = pow(out_color.xyz, vec3(1.0/2.2));
	}
	
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
	// Color 0 is black, so we shift everything.
	x = x+1;
	uint n = baseHash(uint(x));
	uvec3 rz = uvec3(n, n*16807U, n*48271U);
	return vec3(rz & uvec3(0x7fffffffU))/float(0x7fffffff);
}
