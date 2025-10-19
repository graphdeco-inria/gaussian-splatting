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

layout(location=0) out vec4 out_color0;
layout(location=1) out vec4 out_color1;
layout(location=2) out vec4 out_color2;
layout(location=3) out vec4 out_color3;

layout(binding=0) uniform sampler2D image;    // input image with camera-proxy distance in alpha
layout(binding=1) uniform sampler2D proxy;    // intersection point with proxy with depth in alpha
layout(binding=2) uniform sampler2D texture0; // best candidate for each pixel
layout(binding=3) uniform sampler2D texture1; // second best candidate for each pixel
layout(binding=4) uniform sampler2D texture2; // third best candidate for each pixel
layout(binding=5) uniform sampler2D texture3; // fourth best candidate for each pixel
layout(binding=6) uniform sampler2D mask; // masking texture.

uniform mat4 iCamProj;     // input camera projection
uniform vec3 iCamPos;      // input camera position
uniform vec3 iCamDir;      // novel camera projection
uniform vec3 nCamPos;      // novel camera position
uniform bool occlTest;	// do occlusion test
uniform bool doMasking;	// do masking

// vertex coordinates of the 2D screen size quad,
// used for computing texture coordinates
in vec3 vertex_coord;

#define EPSILON 1e-2
#define BETA 	1e-1  	/* Relative importance of resolution penalty */

vec3 project(vec3 point, mat4 proj) {
  vec4 p1 = proj * vec4(point, 1.0);
  vec3 p2 = (p1.xyz/p1.w);
  return (p2.xyz*vec3(0.5) + vec3(0.5));
}

bool frustumTest(vec3 p, vec2 uv) {
  vec3 d1 = iCamDir;
  vec3 d2 = p - iCamPos;
  bool r = dot(d1,d2)>0.0 && uv.x<1.0 && uv.x>0.0 && uv.y<1.0 && uv.y>0.0;
  return r;
}

void main(void) {

  float penalty_res = 0;  		/* Resolution penalty */
  float penalty_ang = 0;			/* Angular penalty */

  vec2 texcoord = (vertex_coord.xy + vec2(1.0)) / 2.0;

  vec4  point  = texture(proxy,    texcoord);
  vec4  color0 = texture(texture0, texcoord);
  vec4  color1 = texture(texture1, texcoord);
  vec4  color2 = texture(texture2, texcoord);
  vec4  color3 = texture(texture3, texcoord);
  
  vec3 uvd = project(point.xyz, iCamProj);
  vec2 uv = uvd.xy;

  vec4 color = texture(image, uv);

  out_color0 = color0;
  out_color1 = color1;
  out_color2 = color2;
  out_color3 = color3;

  if(doMasking){
    float masked = texture(mask, uv).r;
    if(masked < 0.5){
	 return;
    }
  }

  if (frustumTest(point.xyz, uv))
  {
    float dist_i2p 	= distance(point.xyz, iCamPos);
    float dist_n2p 	= distance(point.xyz, nCamPos);
    penalty_res 	= max(0.0001, (dist_i2p - dist_n2p)/dist_i2p );

    //if (abs(dist-color.w) < EPSILON) {

    vec3 v1 = normalize(point.xyz - iCamPos);
    vec3 v2 = normalize(point.xyz - nCamPos);
    if (occlTest && abs(uvd.z-color.w) < EPSILON) {		/* occlusion test */
      //color.w = max(0.0001, acos(dot(v1,v2)));
      penalty_ang = max(0.0001, acos(dot(v1,v2)));
      } else if( occlTest ) {
        return;;
        //color.w = 5.0 + max(0.001, acos(dot(v1,v2))); /* increase the penalty */
      }
	  if (all(equal(color.xyz, vec3(0,0,0)))){
		return;
		}

      color.w = penalty_ang + BETA*penalty_res;

      // compare with best four candiates and insert at the
      // appropriate rank
      bool done = false;
      if (!done && color.w<color0.w) {    // better than best candidate
        out_color0 = color;
        out_color1 = color0;
        out_color2 = color1;
        out_color3 = color2;
        done = true;
      }
      if (!done && color.w<color1.w) {    // better than second best candidate
        out_color0 = color0;
        out_color1 = color;
        out_color2 = color1;
        out_color3 = color2;
        done = true;
      }
      if (!done && color.w<color2.w) {    // better than third best candidate
        out_color0 = color0;
        out_color1 = color1;
        out_color2 = color;
        out_color3 = color2;
        done = true;
      }
      if (!done && color.w<color3.w) {    // better than fourth best candidate
        out_color0 = color0;
        out_color1 = color1;
        out_color2 = color2;
        out_color3 = color;
        done = true;
      }
    }
  }
