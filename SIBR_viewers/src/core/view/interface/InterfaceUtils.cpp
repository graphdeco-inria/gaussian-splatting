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


#include "InterfaceUtils.h"

#include <core/graphics/Mesh.hpp>
#include <core/graphics/Window.hpp>

namespace sibr {

	const std::string InterfaceUtilities::translationScalingVertexShader =
		"#version 420															\n"
		"layout(location = 0) in vec3 in_vertex;								\n"
		"uniform vec2 translation;												\n"
		"uniform vec2 scaling;													\n"
		"void main(void) {														\n"
		"	gl_Position = vec4(scaling*in_vertex.xy+translation,0.0, 1.0);		\n"
		"}																		\n";

	const std::string InterfaceUtilities::colorAlphaFragmentShader =
		"#version 420														\n"
		"uniform vec3 color;												\n"
		"uniform float alpha;												\n"
		"out vec4 out_color;												\n"
		"void main(void) {													\n"
		"		out_color = vec4(color,alpha);								\n"
		"}																	\n";

	const std::string InterfaceUtilities::meshVertexShader =
		"#version 420															\n"
		"layout(location = 0) in vec3 in_vertex;								\n"
		"uniform mat4 mvp;												\n"
		"void main(void) {														\n"
		"	gl_Position = mvp*vec4(in_vertex, 1.0);						\n"
		"}																		\n";

	const std::string InterfaceUtilities::multiViewVertexShader =
		"#version 420										\n"
		"layout(location = 0) in vec3 in_vertex;			\n"
		"out vec2 uv_coord;									\n"
		"uniform vec2 zoomTL;								\n"
		"uniform vec2 zoomBR;								\n"
		"void main(void) {									\n"
		"	uv_coord = 0.5*in_vertex.xy + vec2(0.5);		\n"
		"	uv_coord = zoomTL + (zoomBR-zoomTL)*uv_coord;	\n"
		"	uv_coord.y = 1.0 - uv_coord.y;					\n"
		"	gl_Position = vec4(in_vertex.xy,0.0, 1.0);		\n"
		"}													\n";


	const std::string InterfaceUtilities::multiViewFragmentShader =
		"#version 420														\n"
		"layout(binding = 0) uniform sampler2DArray texArray;				\n"
		"uniform int numImgs;												\n"
		"uniform vec2 grid;													\n"
		"in vec2 uv_coord;													\n"
		"out vec4 out_color;												\n"
		"void main(void) {													\n"
		"	vec2 uvs = uv_coord;											\n"
		"	uvs =  grid*uvs;												\n"
		"  if( uvs.x < 0 || uvs.y < 0 ) { discard; } 						\n"
		"   vec2 fracs = fract(uvs); 										\n"
		"   vec2 mods = uvs - fracs; 										\n"
		"   int n = int(mods.x + grid.x*mods.y); 							\n"
		" if ( n< 0 || n > numImgs || mods.x >= grid.x || mods.y >= (float(numImgs)/grid.x) + 1) { discard; } else { \n"
		"	out_color = texture(texArray,vec3(fracs.x,fracs.y,n));	}		\n"
		"	//out_color = vec4(n/64.0,0.0,0.0,1.0); }						\n"
		"	//out_color = vec4(uv_coord.x,uv_coord.y,0.0,1.0);	}			\n"
		"}																	\n";

	//sibr::GLShader InterfaceUtilities::baseShader;

	//sibr::GLParameter InterfaceUtilities::colorGL;
	//sibr::GLParameter InterfaceUtilities::alphaGL;
	//sibr::GLParameter InterfaceUtilities::scalingGL;
	//sibr::GLParameter InterfaceUtilities::translationGL;

	//sibr::GLShader InterfaceUtilities::multiViewShader;

	//sibr::GLParameter InterfaceUtilities::numImgsGL;
	//sibr::GLParameter InterfaceUtilities::gridGL;
	//sibr::GLParameter InterfaceUtilities::multiViewTopLeftGL;
	//sibr::GLParameter InterfaceUtilities::multiViewBottomRightGL;

	//const InterfaceUtilities::GLinitializer InterfaceUtilities::init;

	InterfaceUtilities::InterfaceUtilities()
	{
		
	}

	void InterfaceUtilities::initAllShaders()
	{
		initBaseShader();
		initMultiViewShader();
		initMeshViewShader();
		CHECK_GL_ERROR;

		std::cout << " all shaders compiled" << std::endl;
	}

	void InterfaceUtilities::freeAllShaders()
	{
		CHECK_GL_ERROR;
		baseShader.terminate();
		CHECK_GL_ERROR;
		multiViewShader.terminate();
		CHECK_GL_ERROR;
	}

	void InterfaceUtilities::initBaseShader()
	{
		baseShader.init("InterfaceUtilitiesBaseShader", translationScalingVertexShader, colorAlphaFragmentShader);
		colorGL.init(baseShader, "color");
		alphaGL.init(baseShader, "alpha");
		scalingGL.init(baseShader, "scaling");
		translationGL.init(baseShader, "translation");
	}

	void InterfaceUtilities::initMultiViewShader()
	{
		multiViewShader.init("InterfaceUtilitiesMultiViewShader", multiViewVertexShader, multiViewFragmentShader);
		multiViewTopLeftGL.init(multiViewShader, "zoomTL"); 
		multiViewBottomRightGL.init(multiViewShader, "zoomBR");
		numImgsGL.init(multiViewShader, "numImgs");
		gridGL.init(multiViewShader, "grid");
	}

	void InterfaceUtilities::initMeshViewShader()
	{
		meshViewShader.init("InterfaceUtilitiesMeshViewShader", meshVertexShader, colorAlphaFragmentShader);
		mvp.init(meshViewShader,"mvp");
		colorMeshGL.init(meshViewShader,"color");
		alphaMeshGL.init(meshViewShader,"alpha");
	}

	void InterfaceUtilities::rectangle(const sibr::Vector3f & color, const UV11 & tl, const UV11 & br, bool fill, float alpha)
	{

		static sibr::Mesh::Ptr rectangleMesh;
		static int lastContextId = -1;

		if (lastContextId != sibr::Window::contextId) {
			rectangleMesh = std::make_shared<sibr::Mesh>(true);
		};
		
	
		baseShader.begin();

		scalingGL.set(sibr::Vector2f(1.0f,1.0f));
		translationGL.set(sibr::Vector2f(0, 0));
		colorGL.set(color);

		rectangleMesh->vertices({
			{ tl.x(), tl.y() , 0 },
			{ tl.x(), br.y() , 0 },
			{ br.x(), br.y() , 0 },
			{ br.x(), tl.y() , 0 }
		});

		if (fill) {
			rectangleMesh->triangles({
				{ 0,1,2 },
				{ 0,2,3 }
			});

			alphaGL.set(alpha);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glBlendEquation(GL_FUNC_ADD);
			rectangleMesh->render(false, false);
		}
		
		rectangleMesh->triangles({
			{ 0,0,1 },{ 1,1,2 },{ 2,2,3 },{ 3,3,0 }
		});

		alphaGL.set(1.0f);
		rectangleMesh->render(false, false, sibr::Mesh::LineRenderMode);
		
		baseShader.end();
	}

	void InterfaceUtilities::rectanglePixels(const sibr::Vector3f & color, const sibr::Vector2f & center, const sibr::Vector2f & diagonal, bool fill, float alpha, const sibr::Vector2f & winSize)
	{
		UV01 centerUV = UV01::from(center.cwiseQuotient(winSize));
		UV01 tl = UV01::from(centerUV - 0.5f*diagonal.cwiseQuotient(winSize));
		UV01 br = UV01::from(centerUV + 0.5f*diagonal.cwiseQuotient(winSize));
		rectangle(color, tl, br, fill, alpha);
	}

	void InterfaceUtilities::circle(const sibr::Vector3f & color, const UV11 & center, float radius, bool fill, float alpha, const sibr::Vector2f & scaling, int precision)
	{

		static int n;
		static sibr::Mesh::Ptr circleMesh;
		static sibr::Mesh::Ptr circleFilledMesh;
		static sibr::Mesh::Triangles circleTriangles;
		static sibr::Mesh::Triangles circleFillTriangles;
		static int lastContextId = -1;

		bool updateMeshes = (lastContextId != sibr::Window::contextId) || (n != precision);
		if (updateMeshes) {
			lastContextId = sibr::Window::contextId;
			n = precision;
			circleTriangles.resize(n);
			circleFillTriangles.resize(n);
			for (int i = 0; i < n; ++i) {
				int next = (i + 1) % n;
				circleTriangles[i] = sibr::Vector3u(i, i, next);
				circleFillTriangles[i] = sibr::Vector3u(i, next, n);
			}

			sibr::Mesh::Vertices vertices(n + 1);
			double base_angle = 2.0*M_PI / (double)n;
			float rho = 0.5f*radius*(float)(1.0 + cos(0.5*base_angle));

			for (int i = 0; i < n; ++i) {
				double angle = i*base_angle;
				vertices[i] = sibr::Vector3f((float)cos(angle), (float)sin(angle), (float)0.0);
			}
			vertices[n] = sibr::Vector3f(0, 0, 0);

			circleMesh = std::make_shared<sibr::Mesh>(true);
			circleFilledMesh = std::make_shared<sibr::Mesh>(true);
			circleMesh->vertices(vertices);
			circleFilledMesh->vertices(vertices);
			circleMesh->triangles(circleTriangles);
			circleFilledMesh->triangles(circleFillTriangles);
		}

		baseShader.begin();

		translationGL.set(sibr::Vector2f(0, 0));

		colorGL.set(color);
		scalingGL.set(scaling);
		translationGL.set(center);

		if (fill) {
			alphaGL.set(alpha);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glBlendEquation(GL_FUNC_ADD);
			circleFilledMesh->render(false, false);
		}

		alphaGL.set(1.0f);
		circleMesh->render(false, false, sibr::Mesh::LineRenderMode);

		baseShader.end();
	}

	void InterfaceUtilities::circlePixels(const sibr::Vector3f & color, const sibr::Vector2f & center, float radius, bool fill, float alpha, const sibr::Vector2f & winSize, int precision)
	{
		UV10 centerUV = UV10::from(center.cwiseQuotient(winSize));
		sibr::Vector2f scaling = radius*sibr::Vector2f(1, 1).cwiseQuotient(winSize);

		circle(color, centerUV, 1.0f, fill, alpha, scaling, precision);
	}

	void InterfaceUtilities::linePixels(const sibr::Vector3f & color, const sibr::Vector2f & ptA, const sibr::Vector2f & ptB, const sibr::Vector2f & winSize)
	{
		UV11 uvA = UV01::from(ptA.cwiseQuotient(winSize));
		UV11 uvB = UV01::from(ptB.cwiseQuotient(winSize));
			
		sibr::Mesh line(true);
		line.vertices({
			{ uvA.x(), uvA.y(), 0.0f },
			{ uvB.x(), uvB.y(), 0.0f }
		});
		line.triangles({
			sibr::Vector3u(0,0,1)
		});

		baseShader.begin();

		scalingGL.set(sibr::Vector2f(1.0f, 1.0f));
		translationGL.set(sibr::Vector2f(0, 0));
		colorGL.set(color);
		alphaGL.set(1.0f);

		line.render(false, false, sibr::Mesh::LineRenderMode);

		baseShader.end();
	}




} //namespace sibr