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
uniform float epsilonOcclusion = 1e-2;

in vec2 vertex_coord;


#define INFTY_W 100000.0

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

  vec4  color0 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4  color1 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4  color2 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4  color3 = vec4(0.0,0.0,0.0,INFTY_W);

  // We need to keep the uvs of the selected colors for the fov blending.
  vec4 uvs01 = vec4(0.0,0.0,0.0,0.0);
  vec4 uvs23 = vec4(0.0,0.0,0.0,0.0);

  for(int cam_id = 0; cam_id < NUM_CAMS; cam_id++){
	if(cam_id >= camsCount){
	  break;
    }
	vec3 uvd = project(point.xyz, icam_proj[cam_id]);
	vec2 ndc = abs(2.0*uvd.xy-1.0);

	if (frustumTest(point.xyz, ndc, cam_id))
	{
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
		

		/// \todo Separate uniform and per-pixel branching. TODO SR: test impact.

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
		
		vec3 v1 = (point.xyz - icam_pos[cam_id]);
		vec3 v2 = (point.xyz - ncam_pos);
		float dist_i2p 	= length(v1);
		float dist_n2p 	= length(v2);

		float penalty_ang = float(occ_test) * max(0.0001, acos(dot(v1,v2)/(dist_i2p*dist_n2p)));

		float penalty_res = max(0.0001, (dist_i2p - dist_n2p)/dist_i2p );
		 
		color.w = penalty_ang + BETA*penalty_res;
		  
        if(useSoftVisibility){

            //vec4 dist_from_edge = vec4(2.0,2.0,2.0,2.0); // texture(soft_visibility_maps, vec3(uvd.xy, cam_true_id));
            vec4 dist_from_edge = texture(soft_visibility_maps, vec3(uvd.xy, selected_cams[cam_id]));
            float weight_visibility = min(dist_from_edge.x/softVisibilityThreshold,1.0);
            color.w /= weight_visibility; 

        }
        
        
		// compare with best four candiates and insert at the
		// appropriate rank
		if (color.w<color3.w) {    // better than fourth best candidate
			
			if (color.w<color2.w) {    // better than third best candidate
				color3 = color2;
				uvs23.zw = uvs23.xy;
				
				if (color.w<color1.w) {    // better than second best candidate
					color2 = color1;
					uvs23.xy = uvs01.zw;

					if (color.w<color0.w) {    // better than best candidate
						color1 = color0;
						uvs01.zw = uvs01.xy;

						color0 = color;
						uvs01.xy = ndc;

					} else {
						color1 = color;
						uvs01.zw = ndc;
					}

				} else {
					color2 = color;
					uvs23.xy = ndc;
				}

			} else {
				color3 = color;
				uvs23.zw = ndc;
			}
		}
	 }  
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

    // Blending on the sides of input images.
#ifdef ENABLE_BORDERS_BLENDING
	// Compute the attenuation factors.
	vec4 fcs01 = 1.0 - smoothstep(vec4(FOV_BLENDING_BORDER), vec4(1.0), uvs01);
	vec4 fcs23 = 1.0 - smoothstep(vec4(FOV_BLENDING_BORDER), vec4(1.0), uvs23);
	fcs01.xz *= fcs01.yw;
	fcs23.xz *= fcs23.yw;

	color0.w *= fcs01.x;
	color1.w *= fcs01.z;
	color2.w *= fcs23.x;
	color3.w *= fcs23.z;
#endif

    // blending
    out_color.w = 1.0;
    out_color.xyz = (color0.w*color0.xyz +
             color1.w*color1.xyz +
             color2.w*color2.xyz +
             color3.w*color3.xyz
            ) / (color0.w + color1.w + color2.w + color3.w);
    gl_FragDepth = point.w;
	

}


/* // ATTENTION - non updated code, tentative to use all samples with a different weighting function.
void main(void) {
	vec4 point = texture(proxy, vertex_coord);
	vec3 color = vec3(0.0);
	float mask = 0.0;
	float sumWeight = 0.0;
	vec4 colors[NUM_CAMS];
	bool discarded[NUM_CAMS];

	float maxPenalty = 0.0;
	for(int cam_id = 0; NUM_CAMS < camsCount; cam_id++){
		 vec3 uvd = project(point.xyz, icam_proj[cam_id]);
		 vec2 uv = uvd.xy;

		 discarded[cam_id] = true;
		
		if (frustumTest(point.xyz, uv, cam_id)) // multiply instead
		{
   
			vec4 inputColor = texture(input_rgb[cam_id], uv);
			
			if ( !all(equal(inputColor.xyz, vec3(0,0,0))) && 
				abs(uvd.z-inputColor.w) < epsilonOcclusion) {		
			    vec3 v1 = point.xyz - icam_pos[cam_id];
			    vec3 v2 = point.xyz - ncam_pos;
				float dist_i2p = length(v1);//distance(point.xyz, icam_pos[cam_id]);
				float dist_n2p = length(v2);//distance(point.xyz, ncam_pos);
			    float penalty_ang = max(0.0001, acos(dot(v1,v2)/(dist_i2p*dist_n2p)));
				float penalty_res 	= max(0.0001, (dist_i2p - dist_n2p)/dist_i2p );
				float penalty = penalty_ang + BETA*penalty_res;
				maxPenalty = max(maxPenalty, penalty);
			
				
				colors[cam_id] = vec4(inputColor.rgb, penalty);
				discarded[cam_id] = false;
			} 
		}
	}
	
	for(int i = 0; i < NUM_CAMS; i++){
		if(discarded[i]){
			continue;
		}
		float weight = max(0.0, 1.0 - colors[i].w/(maxPenalty*1.0000001));
		sumWeight += weight;
		color += weight*colors[i].rgb;
	}
	
	out_color.rgb = color / sumWeight;
	out_color.a = 1.0;
}
*/