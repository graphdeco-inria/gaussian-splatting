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

layout(location=0) out vec4 out_color;

layout(binding=0) uniform sampler2D proxy;  // intersection point with proxy with depth in alpha
layout(binding=1) uniform sampler2D ulr0;   // best candidate for each pixel
layout(binding=2) uniform sampler2D ulr1;   // second best candidate for each pixel
layout(binding=3) uniform sampler2D ulr2;   // third best candidate for each pixel
layout(binding=4) uniform sampler2D ulr3;   // fourth best candidate for each pixel

// vertex coordinates of the 2D screen size quad,
// used for computing texture coordinates
in vec3 vertex_coord;

void main(void) {
    vec2 texcoord = (vertex_coord.xy + vec2(1.0)) / 2.0;

    vec4  point  = texture(proxy, texcoord);
    vec4  color0 = texture(ulr0,  texcoord);
    vec4  color1 = texture(ulr1,  texcoord);
    vec4  color2 = texture(ulr2,  texcoord);
    vec4  color3 = texture(ulr3,  texcoord);

    float thresh = 1.0000001 * color3.w;
    color0.w = max(0, 1.0 - color0.w/thresh);
    color1.w = max(0, 1.0 - color1.w/thresh);
    color2.w = max(0, 1.0 - color2.w/thresh);
    color3.w = max(0, 1.0 - color3.w/thresh);

    // ignore any candidate which is black
    if (all(equal(color0.xyz, vec3(0,0,0)))) color0.w = 0;
    if (all(equal(color1.xyz, vec3(0,0,0)))) color1.w = 0;
    if (all(equal(color2.xyz, vec3(0,0,0)))) color2.w = 0;
    if (all(equal(color3.xyz, vec3(0,0,0)))) color3.w = 0;

    // only blend the best two candidates
    /*color2.w = 0.00001;
    color3.w = 0.00001;*/

    // blending
    out_color.xyz = (color0.xyz*color0.w +
            color1.xyz*color1.w +
            color2.xyz*color2.w +
            color3.xyz*color3.w
            ) / (color0.w + color1.w + color2.w + color3.w);
    out_color.w = 1.0;
    gl_FragDepth = point.w;

    // discard if there was no intersection with the proxy
    if (point.w <= 0) {
        discard;
    }
}
