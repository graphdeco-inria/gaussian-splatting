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

layout(location = 0) out vec4 out_color;

layout(binding=0) uniform sampler2D proxy;
layout(binding=1) uniform sampler2DArray soft_visibility_maps;

uniform vec3 ncam_pos;
uniform sampler2D input_rgb[NUM_CAMS];
uniform sampler2D masks[NUM_CAMS];
uniform vec3 icam_pos[NUM_CAMS];
uniform vec3 icam_dir[NUM_CAMS];
uniform mat4 icam_proj[NUM_CAMS];
uniform int selected_cams[NUM_CAMS];
uniform bool occ_test;
uniform bool invert_mask;
uniform bool is_binary_mask;
uniform bool discard_black_pixels;
uniform bool doMasking;
uniform float softVisibilityThreshold;
uniform bool useSoftVisibility;
uniform int camsCount;

in vec2 vertex_coord;


#define INFTY_W 100000.0
#define EPSILON 1e-2
#define BETA 	1e-1  	/* Relative importance of resolution penalty */
#define FOV_BLENDING_BORDER 0.6
#define ENABLE_BORDERS_BLENDING /// \todo TODO SR: investigate effect of this additional blending.

vec3 project(vec3 point, mat4 proj) {
  vec4 p1 = proj * vec4(point, 1.0);
  vec3 p2 = (p1.xyz/p1.w);
  return (p2.xyz*0.5 + 0.5);
}

bool frustumTest(vec3 p, vec2 ndc, int cam_id) {
  vec3 d1 = icam_dir[cam_id];
  vec3 d2 = p - icam_pos[cam_id];
  return !any(greaterThan(ndc, vec2(1.0))) && dot(d1,d2)>0.0;
}

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
	  
	for(int cam_id = 0; cam_id < NUM_CAMS; cam_id++){
		
		if(cam_id >= camsCount){
		  break;
		}
		vec3 uvd = project(point.xyz, icam_proj[cam_id]);
		vec2 ndc = abs(2.0*uvd.xy-1.0);
		
		if (!frustumTest(point.xyz, ndc, cam_id)) {
			continue;
		}
		
		vec4 color = texture(input_rgb[cam_id], uvd.xy);
				
		if(doMasking){	
			float masked = texture(masks[cam_id], uvd.xy).r;
			 
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
			if(abs(uvd.z-color.w) >= EPSILON) {	  
				continue;
			}
		} 
		
		vec3 v1 = (point.xyz - icam_pos[cam_id]);
		float dist_i2p 	= length(v1);
		
		float penalty_ang = float(occ_test) * max(0.0001, acos(dot(v1,v2)/(dist_i2p*dist_n2p)));

		float penalty_res = max(0.0001, (dist_i2p - dist_n2p)/dist_i2p );
		 
		color.w = penalty_ang + BETA*penalty_res;
		  
		if(useSoftVisibility){
			vec4 dist_from_edge = texture(soft_visibility_maps, vec3(uvd.xy, selected_cams[cam_id]));
			float weight_visibility = min(dist_from_edge.x/softVisibilityThreshold,1.0);
			color.w /= (weight_visibility*weight_visibility); 
		}
		
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
