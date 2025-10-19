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

uniform float scale;

layout(binding = 0) uniform sampler2D coarse;
layout(binding = 1) uniform sampler2D constraint;
layout(location= 0) out vec4 out_color;

in vec4 texcoord;


void main(void) {
    //  sample color from lower multigrid level by scaling texture coordinates
    vec4 color = texture(coarse,vec2(gl_FragCoord.xy/scale)/textureSize(coarse,0).xy,0);

    //  sample Dirichlet constraint without texture filtering because pixel without
    //  Dirichlet constraint are black and texture filtering may break this check
    vec4 cons  = texelFetch(constraint,ivec2(gl_FragCoord),0);

    //  write color of lower level to output except holes pixels in the constraint
    if (any(greaterThan(cons.rgb,vec3(0.01))))
        out_color = cons;
    else
        out_color = color;
}
