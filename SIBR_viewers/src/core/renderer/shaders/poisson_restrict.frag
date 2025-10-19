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

layout(binding = 0) uniform sampler2D cons;
layout(location= 0) out vec4 out_constraint;

void main(void) {
    mat3 f = mat3(
            0.25, 0.50, 0.25,
            0.50, 1.00, 0.50,
            0.25, 0.50, 0.25);

    vec4 constr = vec4(0);

    float sum = 0;

    ivec2 coord = ivec2(gl_FragCoord.xy*scale);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            vec4 c  = texelFetch(cons,coord+ivec2(i-1,j-1),0);
            float a = float(any(greaterThan(c.rgb, vec3(0.01))));
            constr += f[i][j] * a * c;
            sum    += f[i][j] * a;
        }
    }
    out_constraint = constr / sum;
}
