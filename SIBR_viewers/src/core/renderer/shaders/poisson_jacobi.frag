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

uniform vec4 weights;
uniform float scale;

layout(binding = 0) uniform sampler2D curr_tex;
layout(location= 0) out vec4 out_color;

void main(void) {
    ivec2 coord = ivec2(gl_FragCoord.xy*scale);

    float w_center = weights.x;
    float w_edge   = weights.y;
    float w_corner = weights.z;
    float w_c_x_inv= weights.w;

    // Solve the Laplace equation, same as Poisson equation with
    // divergence equal to 0
    out_color =
        -w_c_x_inv * (
                (texelFetch(curr_tex, coord, 0) * w_center +
                 (texelFetch(curr_tex, coord+ivec2( 0, 1), 0) +
                  texelFetch(curr_tex, coord+ivec2( 0,-1), 0) +
                  texelFetch(curr_tex, coord+ivec2( 1, 0), 0) +
                  texelFetch(curr_tex, coord+ivec2(-1, 0), 0)) * w_edge +
                 (texelFetch(curr_tex, coord+ivec2(-1,-1), 0) +
                  texelFetch(curr_tex, coord+ivec2(-1, 1), 0) +
                  texelFetch(curr_tex, coord+ivec2( 1,-1), 0) +
                  texelFetch(curr_tex, coord+ivec2( 1, 1), 0)) * w_corner
                )
            );
}
