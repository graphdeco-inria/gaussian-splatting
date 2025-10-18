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
const float PI = 3.1415926535897932384626433832795;

vec2 poissonDisk[64] = vec2[](
vec2(-0.613392, 0.617481),
vec2(0.170019, -0.040254),
vec2(-0.299417, 0.791925),
vec2(0.645680, 0.493210),
vec2(-0.651784, 0.717887),
vec2(0.421003, 0.027070),
vec2(-0.817194, -0.271096),
vec2(-0.705374, -0.668203),
vec2(0.977050, -0.108615),
vec2(0.063326, 0.142369),
vec2(0.203528, 0.214331),
vec2(-0.667531, 0.326090),
vec2(-0.098422, -0.295755),
vec2(-0.885922, 0.215369),
vec2(0.566637, 0.605213),
vec2(0.039766, -0.396100),
vec2(0.751946, 0.453352),
vec2(0.078707, -0.715323),
vec2(-0.075838, -0.529344),
vec2(0.724479, -0.580798),
vec2(0.222999, -0.215125),
vec2(-0.467574, -0.405438),
vec2(-0.248268, -0.814753),
vec2(0.354411, -0.887570),
vec2(0.175817, 0.382366),
vec2(0.487472, -0.063082),
vec2(-0.084078, 0.898312),
vec2(0.488876, -0.783441),
vec2(0.470016, 0.217933),
vec2(-0.696890, -0.549791),
vec2(-0.149693, 0.605762),
vec2(0.034211, 0.979980),
vec2(0.503098, -0.308878),
vec2(-0.016205, -0.872921),
vec2(0.385784, -0.393902),
vec2(-0.146886, -0.859249),
vec2(0.643361, 0.164098),
vec2(0.634388, -0.049471),
vec2(-0.688894, 0.007843),
vec2(0.464034, -0.188818),
vec2(-0.440840, 0.137486),
vec2(0.364483, 0.511704),
vec2(0.034028, 0.325968),
vec2(0.099094, -0.308023),
vec2(0.693960, -0.366253),
vec2(0.678884, -0.204688),
vec2(0.001801, 0.780328),
vec2(0.145177, -0.898984),
vec2(0.062655, -0.611866),
vec2(0.315226, -0.604297),
vec2(-0.780145, 0.486251),
vec2(-0.371868, 0.882138),
vec2(0.200476, 0.494430),
vec2(-0.494552, -0.711051),
vec2(0.612476, 0.705252),
vec2(-0.578845, -0.768792),
vec2(-0.772454, -0.090976),
vec2(0.504440, 0.372295),
vec2(0.155736, 0.065157),
vec2(0.391522, 0.849605),
vec2(-0.620106, -0.328104),
vec2(0.789239, -0.419965),
vec2(-0.545396, 0.538133),
vec2(-0.178564, -0.596057)
);

uniform vec3 lightDir;
uniform float sun_app_radius;
uniform mat4 depthMapMVPinv;
uniform float depthMapRadius;
uniform float biasControl;

layout(binding=0) uniform sampler2D depthMap;

in vec4 depthMapProj;
in vec3 VtoF_normal;
in vec3 VtoF_pos;

out float out_val;

void main(void) {
	
	vec2 texc = (depthMapProj.xy + 1.0) / 2.0;

	float depthImSpace = depthMapProj.z;

	float cosTheta = clamp(dot(VtoF_normal,lightDir),0.0,1.0);
	float bias = biasControl*tan(acos(cosTheta));
	bias = clamp(bias, 0.0, 5*biasControl);

	int textureWidth = textureSize(depthMap,0).x;

	// Compute the size of the shadow transition

	// The 2 account for the fact that we are treating the radius.

	int r_blocker = int(ceil(0.5*tan(sun_app_radius*PI/180.0)*textureWidth));
	float mean_blocker_val = 0.0;
	float blocker_num_val = 0.0;
	float sum_weight = 0.0;

	for(int i = 0; i<7 ; i++){

		float theta_rot=i;
		mat2 rotation_poisson =mat2(cos(theta_rot), sin(theta_rot), -sin(theta_rot), cos(theta_rot));

		for(int k = 0; k <64 ; k++){

			float pixDist = length(r_blocker*poissonDisk[k]/textureWidth);
			if(pixDist<=r_blocker){
				float depthMapVal = texture(depthMap, texc + r_blocker*rotation_poisson*poissonDisk[k]/textureWidth).x;

				float bias_with_dist = bias*(pixDist+1.0);

				if( depthImSpace-bias_with_dist > depthMapVal ){

					vec3 blocker_pos=(depthMapMVPinv*vec4(depthMapProj.x,depthMapProj.y,depthMapVal,1.0)).xyz;

					float weight = 0.01+exp(-pow(pixDist/(0.02*r_blocker),2.0));
					mean_blocker_val += weight*length(VtoF_pos-blocker_pos);
					sum_weight += weight;
					blocker_num_val += 1.0;
			    }
			}		
		}
	}

	if( dot(VtoF_normal,lightDir) <= -0.017 ){
		out_val = 0.0;
	}
	else if(blocker_num_val < 1.0){
		out_val = 1.0;
	}
	else{

		float d_blocker_receiver = mean_blocker_val/sum_weight;

		float wShadow=2*d_blocker_receiver*tan(sun_app_radius*PI/180.0);
		//float angleCompensation= dot(normalize(VtoF_normal),normalize(lightDir));
		//float wShadowAngle = wShadow/angleCompensation;
		float ratioShadowTexture = wShadow/(2.0*depthMapRadius);
		float pixelShadowTexture = ratioShadowTexture*textureWidth;

		const int r = int(ceil(pixelShadowTexture));

		float num_val = 0.0;
		float sum_val = 0.0;

		for(int i = 0; i<14 ; i++){

			float theta_rot=float(i);
			mat2 rotation_poisson =mat2(cos(theta_rot), sin(theta_rot), -sin(theta_rot), cos(theta_rot));

			for(int k = 0; k <64 ; k++){

				float depthMapVal = texture(depthMap, texc + r*rotation_poisson*poissonDisk[k]/textureWidth).x;
				float pixDist = length(r*poissonDisk[k]/textureWidth);
				float bias_with_dist = bias*(pixDist+1.0);

				if( depthImSpace-bias_with_dist > depthMapVal ){
			    	sum_val += 0.0;
			    }
			    else{
			    	sum_val += 1.0;
			    }
			    num_val +=1.0;
			}
		}

		out_val = sum_val/num_val;
	}

}

