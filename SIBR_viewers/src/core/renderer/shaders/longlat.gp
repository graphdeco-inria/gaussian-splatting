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


#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices=6) out;

layout(location = 1) in vec3 colors_tes[];
layout(location = 2) in vec2 coordsTex_tes[];
layout(location = 3) in vec3 normals_tes[];

uniform vec3 pos;                           

const float PI = 3.1415926535897932384626433832795;   

layout(location = 0) out vec4 position;
layout(location = 1) out vec3 colors_gs;
layout(location = 2) out vec2 coordsTex_gs;
layout(location = 3) out vec3 normals_gs;

void main()
{	
int i,j;
vec3 toPoint[3];
vec3 d[3];
float lat[3];
float longt[3];


for(i=0; i<3; i++)
{
  toPoint[i] = gl_in[i].gl_Position.xyz-pos;
  d[i] = normalize(toPoint[i]);                                  
  lat[i] = d[i].z;
  longt[i] = atan(d[i].y,d[i].x);
  if(longt[i]<0)
    longt[i] += 2.0f*PI;
}


if((abs(longt[1]-longt[2])<PI && abs(longt[0]-longt[2])<PI && abs(longt[1]-longt[0])<PI)){


  float fact=100.0f;

  for(i=0; i<3; i++)
  {
  gl_Position = vec4(longt[i]/PI-1.0,-lat[i],length(toPoint[i])/fact,1.0f);
  position=gl_Position;
  colors_gs = colors_tes[i];
  coordsTex_gs = coordsTex_tes[i];
  normals_gs = normals_tes[i];
  EmitVertex();
  }
  EndPrimitive();

}
else{
for(i=0; i<3; i++){
  if(abs(longt[i]-longt[(i+1)%3])>PI && abs(longt[i]-longt[(i+2)%3])>PI){

    float longt_0[3]=longt;
    float longt_1[3]=longt;

    if(longt[i]>PI){
      longt_0[i]=longt[i]-2.0f*PI;
      longt_1[(i+1)%3]=longt[(i+1)%3]+2.0f*PI;
      longt_1[(i+2)%3]=longt[(i+2)%3]+2.0f*PI;
    }
    else{
      longt_0[i]=longt[i]+2.0f*PI;
      longt_1[(i+1)%3]=longt[(i+1)%3]-2.0f*PI;
      longt_1[(i+2)%3]=longt[(i+2)%3]-2.0f*PI;
    }

    for(j=0; j<3; j++)
    {
      gl_Position = vec4(longt_0[j]/PI-1.0,-lat[j],length(toPoint[j])/100.0f,1.0f);
      position=gl_Position;
      colors_gs = colors_tes[i];
      coordsTex_gs = coordsTex_tes[i];
      normals_gs = normals_tes[i];
      EmitVertex();
    }
    EndPrimitive();

    for(j=0; j<3; j++)
    {
      gl_Position = vec4(longt_1[j]/PI-1.0,-lat[j],length(toPoint[j])/100.0f,1.0f);
      position=gl_Position;
      colors_gs = colors_tes[i];
      coordsTex_gs = coordsTex_tes[i];
      normals_gs = normals_tes[i];
      EmitVertex();
    }
    EndPrimitive();
    break;
  }
}

  
}

}
