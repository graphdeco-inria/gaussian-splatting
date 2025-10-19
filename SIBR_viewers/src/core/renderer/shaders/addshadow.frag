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

#define SHADOWPOWER_METHOD_ESTIM 			1
#define SHADOWPOWER_METHOD_3D 				2
#define SHADOWPOWER_METHOD_ESTIM_NONLINEAR 	3

#define SHADOWPOWER_METHOD SHADOWPOWER_METHOD_ESTIM

layout(binding = 0) uniform sampler2D tex;
layout(binding = 1) uniform sampler2D firstPassRT; /// \todo TODO: use ping pong buffering to update it with the last add

uniform mat4 	in_inv_proj;
uniform vec2  	in_image_size;

layout(location= 0) out vec4 out_color;

in vec2 tex_coord;



//const float blurSize = 1.0 / 1000.0;

const float constFgAdditionalOffset = -0.02; // note it should not be linear because of near/far+proj
const float constSmoothShadowDistAtten = 8.0;

vec3 unproject(vec3 xyd, mat4 inv_proj) {
    vec4 pxl = vec4(xyd,1.0)*vec4(2.0)-vec4(1.0); // [0,1] -> [-1,1]
    vec4 obj = inv_proj * pxl;                    // unproject
    return (obj.xyz/obj.w);
}

float 	smoothShadowDist( float dist01 )
{
	float v = 1.0 - max(0.0, min(1.0, dist01) );
	return exp(log(v)*constSmoothShadowDistAtten);
}

void main(void) {

//===========================================================================//	
//                                                                           //
//							   Adding Shadow                                 //
//                                                                           //
//===========================================================================//	
{	
	const float scanSizeX = (1.0 / in_image_size.x)*4.0; /// \todo TODO: should be split into H and W and use image size
	const float scanSizeY = (1.0 / in_image_size.y)*4.0; /// \todo TODO: should be split into H and W and use image size
	
	vec4 bg = texture(firstPassRT,tex_coord); 	// background color
	vec4 fg = texture(tex,tex_coord);			// foreground color (object to add)
	float bgDepth = bg.a;
	float outDepth = bgDepth;
	float fgDepth = fg.a+constFgAdditionalOffset;

	// By default set output values using bg
    out_color = vec4(bg.rgb, 1.0);
	//out_color = vec4(bg.a, 0.0, 0.0, 1.0);
	//gl_FragDepth = bgDepth;
    
	bool fgIsEmpty = (fg.r == 0 && fg.g == 0 && fg.b == 0);
	/// gl_FragDepth = 0;
	
    if (fgIsEmpty == false)
	{
		out_color = vec4(fg.rgb, (bgDepth <= fgDepth)? 0.0 : 1.0);
		//fgDepth = bgDepth;
		outDepth = (bgDepth <= fgDepth)? bgDepth : fgDepth;
		//gl_FragDepth = fgDepth;
		/// out_color = vec4(fg.a, 0.0, 0.0, 1.0);
	}
	else
	{		
		// Scan for non-empty pixels for determining the power
		// of the shadow.
		// 'non-empty' pixels are FULL black pixels
		
		const int 	scanItCount = 8;
		const int 	maxScanablePixels = (scanItCount*2 + 1)*(scanItCount*2 + 1);
		const float maxAxisX = (scanItCount*2 + 1)*scanSizeX;
		const float maxAxisY = (scanItCount*2 + 1)*scanSizeY;
		const float maxScanDist = maxAxisX*maxAxisX + maxAxisY*maxAxisY; 
		
		float nearestDist = maxScanDist*2.0;//maxScanDist + 1.0;
		float nearestXs = 1.0;
		float nearestYs = 1.0;
		float nearestDepth = 0.0;
		float averageBgDepth = 0.0;
		float nonEmptyPixelFound = 0;
		for (int x = -scanItCount; x <= scanItCount; x++)
		{
			for (int y = -scanItCount; y <= scanItCount; y++)
			{
				float xs = x*scanSizeX;
				float ys = y*scanSizeY;
				float dist = (xs*xs + ys*ys);
				vec4 color = texture(tex, vec2(tex_coord.x+xs, tex_coord.y+ys));
				float sampleDepth =  color.a+constFgAdditionalOffset;
				
				averageBgDepth += sampleDepth;
				
				if ( (color.r == 0 && color.g == 0 && color.b == 0) == false
//// [A] this one will cause you trouble with object in front of your shadow caster
//				)
//// [B] this one will prevent you from casting shadow in front of your object
////     but visually, you can make it fly a bit and it look like the under is in front
			&& sampleDepth <= bgDepth )
//// [C] this one is the best in fidelity but requires a call to texture(...)
//				&& sampleDepth <= texture(firstPassRT, vec2(tex_coord.x+xs, tex_coord.y+ys)).a)
				{
					if (dist < nearestDist)
					{
						nearestXs = xs;
						nearestYs = ys;
						// nearestDist = dist;
						// nearestDepth = sampleDepth;
					}

					
					//nearestXs = min(xs, nearestXs);
					//nearestYs = min(ys, nearestYs); // note that stored nearestXs/Ys might stores an unexisting coordinate pair
					nearestDist = min(nearestDist, dist);
					nearestDepth += sampleDepth;//max(nearestDepth, sampleDepth);
					++nonEmptyPixelFound;
					
					averageBgDepth -= sampleDepth; // cancel this in this case
				}
			}
		}
		nearestDepth = nearestDepth/float(nonEmptyPixelFound);
		averageBgDepth = averageBgDepth/float(maxScanablePixels-nonEmptyPixelFound);
		
		//if (nearestDist > 0)
		{
			
			// Compute the shadow power
			float shadowPower = 1.0;
			
			float ratioNonEmptyPixelFound = float(nonEmptyPixelFound) / float(maxScanablePixels);
			// influence of the caster size
			shadowPower *= smoothstep(0.0, 0.5, ratioNonEmptyPixelFound);
			
			//if (nonEmptyPixelFound < 100)
			//	shadowPower = 0.0;
			
			//shadowPower *= max(0.0, min(1.0, nonEmptyPixelFound/minPixelCaster ));
			
			// Dev Note for improving things
			// There are two way to implement the influence of the shadow caster/receiver distance.
			// [3d solution]
			// - one is to unproject both points and measure their distance in 3d world unit
			// [estim solution]
			// - another one is roughly estimate the effect

#if SHADOWPOWER_METHOD == SHADOWPOWER_METHOD_ESTIM_NONLINEAR			
			// [estim solution]
			// influence of the distance to the object that cast this shadow (slightly improve but not enough)
			//shadowPower *= smoothShadowDist(nearestDist/maxScanDist);
			// influence of the depth distance
			const float maxDiffDepth = 0.04;//0.15; // because it's nonlinear, it will react differently depending on Z (and your dataset clipping planes's near/far)
			float diffDepth =  bgDepth-nearestDepth;
			diffDepth = diffDepth / maxDiffDepth;
			float depthFactor = clamp(diffDepth, 0.0, 1.0);
			shadowPower *= (1.0 - depthFactor);
#endif			
			
#if SHADOWPOWER_METHOD == SHADOWPOWER_METHOD_ESTIM			
			// [estim solution]
			// influence of the distance to the object that cast this shadow (slightly improve but not enough)
			//shadowPower *= smoothShadowDist(nearestDist/maxScanDist);
			// influence of the depth distance
			const float maxDist = 0.4; // in world unit
			vec3 bg2dPos = vec3(tex_coord.xy, bgDepth);
			vec3 bg3dPos = unproject(bg2dPos, in_inv_proj);
			vec3 caster2dPos = vec3(vec2(tex_coord.x+nearestXs, tex_coord.y+nearestYs), nearestDepth);
			vec3 caster3dPos = unproject(caster2dPos, in_inv_proj);
			float line = abs(caster3dPos.z-bg3dPos.z);
			float factorDist = 1.0 - max(0.0, min(1.0, line/maxDist) );
			shadowPower *= factorDist;
#endif			
			
#if SHADOWPOWER_METHOD == SHADOWPOWER_METHOD_3D
			// [3d solution]			
			// influence of the shadow receiver distance
			const float maxDist = 0.25; // in world unit
			const float maxDistSqr = maxDist*maxDist;
			vec3 bg2dPos = vec3(tex_coord.xy, bgDepth);
			vec3 bg3dPos = unproject(bg2dPos, in_inv_proj);
			vec3 caster2dPos = vec3(vec2(tex_coord.x+nearestXs, tex_coord.y+nearestYs), nearestDepth);
			vec3 caster3dPos = unproject(caster2dPos, in_inv_proj);
			vec3 line = caster3dPos-bg3dPos;
			float distSqr = line.x*line.x + line.y*line.y + line.z*line.z;
			float factorDist = 1.0 - max(0.0, min(1.0, distSqr/maxDistSqr) );
			shadowPower *= factorDist;
#endif			
			
		
			
			out_color = vec4(0, 0, 0, shadowPower / 1.5);
			
			// //out_color = vec4(fgZ/10.0, 0, 0, 1.0);
			// float nearPlane = 3.23569;
			// float farPlane = 17.1543;
			// //bgZ = (bg2dPos.z * bg2dPos.w - nearPlane) / (farPlane - nearPlane);
			// out_color = vec4(bg2dPos.z/farPlane, 0, 0, 1.0);
			//out_color = vec4(bgDepth, 0, 0, 1.0);
			//gl_FragDepth = nearestDepth;
			
			
			//gl_FragDepth = bgDepth + constFgAdditionalOffset - 1;
			float newDepth = bgDepth;// + constFgAdditionalOffset;
			outDepth = (newDepth <= bgDepth)? newDepth : bgDepth;
		}
		
		// if (nearestDist < 1.0)
		// {
			// out_color = vec4(nearestDepth, 0.0, 0.0, 1.0);
			// gl_FragDepth = 0;
		// }		
	}
	
	// Simulate BLEND function (GL_ONE_MINUS_SRC_ALPHA)
	//out_color = vec4(bg.xyz + out_color.xyz/out_color.a, gl_FragDepth);
	out_color = vec4(bg.xyz*(1.0 - out_color.a) + out_color.xyz*out_color.a,  outDepth);
	//gl_FragDepth = outDepth;
	//out_color = bg;
}

}
