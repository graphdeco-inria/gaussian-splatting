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


#include "Intersector2D.h"
#include <core/graphics/Shader.hpp>
#include <core/graphics/RenderTarget.hpp>
#include <core/graphics/Mesh.hpp>

namespace sibr {


	float Intersector2D::sign(sibr::Vector2f p1, sibr::Vector2f p2, sibr::Vector2f p3)
	{
		return (p1.x() - p3.x()) * (p2.y() - p3.y()) - (p2.x() - p3.x()) * (p1.y() - p3.y());
	}

	bool Intersector2D::PointInTriangle(sibr::Vector2f pt, sibr::Vector2f v1, sibr::Vector2f v2, sibr::Vector2f v3)
	{
		bool b1, b2, b3;

		b1 = sign(pt, v1, v2) < 0.0f;
		b2 = sign(pt, v2, v3) < 0.0f;
		b3 = sign(pt, v3, v1) < 0.0f;

		return ((b1 == b2) && (b2 == b3));
	}

	//Segment are a->b and c->d
	bool Intersector2D::LineLineIntersect(sibr::Vector2f a, sibr::Vector2f b, sibr::Vector2f c, sibr::Vector2f d)
	{
		float den = ((d.y() - c.y())*(b.x() - a.x()) - (d.x() - c.x())*(b.y() - a.y()));
		float num1 = ((d.x() - c.x())*(a.y() - c.y()) - (d.y() - c.y())*(a.x() - c.x()));
		float num2 = ((b.x() - a.x())*(a.y() - c.y()) - (b.y() - a.y())*(a.x() - c.x()));
		float u1 = num1 / den;
		float u2 = num2 / den;

		if (den == 0 && num1 == 0 && num2 == 0)
			/* The two lines are coincidents */
			return false;
		if (den == 0)
			/* The two lines are parallel */
			return false;
		if (u1 < 0 || u1 > 1 || u2 < 0 || u2 > 1)
			/* Lines do not collide */
			return false;
		/* Lines DO collide */
		return true;
	}

	bool Intersector2D::TriTriIntersect(sibr::Vector2f t0_0, sibr::Vector2f t0_1, sibr::Vector2f t0_2,
		sibr::Vector2f t1_0, sibr::Vector2f t1_1, sibr::Vector2f t1_2) {

		//Test if lines intersects
		if (LineLineIntersect(t0_0, t0_1, t1_0, t1_1)) { return true; };
		if (LineLineIntersect(t0_0, t0_1, t1_0, t1_2)) { return true; };
		if (LineLineIntersect(t0_0, t0_1, t1_1, t1_2)) { return true; };
		if (LineLineIntersect(t0_0, t0_2, t1_0, t1_1)) { return true; };
		if (LineLineIntersect(t0_0, t0_2, t1_0, t1_2)) { return true; };
		if (LineLineIntersect(t0_0, t0_2, t1_1, t1_2)) { return true; };
		if (LineLineIntersect(t0_1, t0_2, t1_0, t1_1)) { return true; };
		if (LineLineIntersect(t0_1, t0_2, t1_0, t1_2)) { return true; };
		if (LineLineIntersect(t0_1, t0_2, t1_1, t1_2)) { return true; };


		//Test if one point in triangle :
		if (PointInTriangle(t0_0, t1_0, t1_1, t1_2) ||
			PointInTriangle(t0_1, t1_0, t1_1, t1_2) ||
			PointInTriangle(t0_2, t1_0, t1_1, t1_2) ||
			PointInTriangle(t1_0, t0_0, t0_1, t0_2) ||
			PointInTriangle(t1_1, t0_0, t0_1, t0_2) ||
			PointInTriangle(t1_2, t0_0, t0_1, t0_2)) {
			return true;
		}

		return false;

	}

	bool Intersector2D::QuadQuadIntersect(sibr::Vector2f q0_0, sibr::Vector2f q0_1, sibr::Vector2f q0_2, sibr::Vector2f q0_3,
		sibr::Vector2f q1_0, sibr::Vector2f q1_1, sibr::Vector2f q1_2, sibr::Vector2f q1_3)
	{
		if (TriTriIntersect(
			q0_0, q0_1, q0_3,
			q1_0, q1_1, q1_3)) {
			return true;
		}
		if (TriTriIntersect(
			q0_0, q0_1, q0_3,
			q1_1, q1_2, q1_3)) {
			return true;
		}
		if (TriTriIntersect(
			q0_1, q0_2, q0_3,
			q1_0, q1_1, q1_3)) {
			return true;
		}
		if (TriTriIntersect(
			q0_1, q0_2, q0_3,
			q1_1, q1_2, q1_3)) {
			return true;
		}

		return false;
	}

	std::vector<std::vector<bool>> Intersector2D::frustrumQuadsIntersect(std::vector<quad> & quads, const std::vector<InputCamera::Ptr> & cams)
	{
		std::clock_t previous;
		double duration;
		previous = std::clock();

		std::vector<std::vector<bool>> result(cams.size(), std::vector<bool>(quads.size(), false));

		sibr::GLShader				shader;
		sibr::GLParameter			shader_proj;


		std::string vertexShader =
			"#version 420\n"
			"uniform mat4 MVP;\n"
			"layout(location = 0) in vec3 in_vertex;\n"
			"void main(void) {\n"
			"	gl_Position = MVP * vec4(in_vertex, 1.0);\n"
			"}\n";


		std::string fragmentShader =
			"#version 420\n"
			"out float out_color;\n"
			"void main(void) {\n"
			"		out_color = 1.0;\n"
			"}\n";

		shader.init("quadShader", vertexShader, fragmentShader);
		shader_proj.init(shader, "MVP");

		std::shared_ptr<sibr::RenderTargetLum> rtLum;

		for (int c = 0; c < cams.size(); c++) {
			const sibr::InputCamera & cam = *cams[c];

			float ratio = (float)cam.h() / (float)cam.w();
			int w = std::min(400, (int)cam.w());
			int h = int(w*ratio);

			rtLum.reset(new sibr::RenderTargetLum(w, h));

			for (int q = 0; q < quads.size(); q++) {

				quad & quad = quads[q];

				sibr::ImageL8 imLum;

				std::shared_ptr<sibr::Mesh> quadMesh = std::shared_ptr<sibr::Mesh>(new sibr::Mesh(true));

				std::vector<sibr::Vector3f> vertexBuffer;
				vertexBuffer.push_back(quad.q1);
				vertexBuffer.push_back(quad.q2);
				vertexBuffer.push_back(quad.q3);
				vertexBuffer.push_back(quad.q4);

				int indices[12] = { 0, 1, 2, 0, 2, 3, 1, 2, 3, 0, 1, 3 }; //triangle added, can be optimized if quad ensured with good order
				std::vector<uint> indicesBuffer(&indices[0], &indices[0] + 12);

				quadMesh->vertices(vertexBuffer);
				quadMesh->triangles(indicesBuffer);

				glViewport(0, 0, w, h);
				rtLum->bind();
				glClearColor(0.0, 0.0, 0.0, 0.0);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				shader.begin();
				shader_proj.set(cam.viewproj());

				quadMesh->render(false, false, sibr::Mesh::RenderMode::FillRenderMode);

				shader.end();

				rtLum->readBack(imLum);

				bool nonBlack = false;
				bool breakLoop = false;
				for (int j = 0; j < (int)(rtLum->h()); j++) {
					for (int i = 0; i < (int)(rtLum->w()); i++) {
						if (imLum(i, j).x() != 0) {
							nonBlack = true;
							breakLoop = true;
							break;
						}
					}
					if (breakLoop)
						break;
				}

				//std::cout << "result " << q << " : " << nonBlack << std::endl;
				if (nonBlack)
					result[c][q] = true;

			}
		}

		duration = (std::clock() - previous) / (double)CLOCKS_PER_SEC;
		std::cout << "render : " << duration << std::endl;
		previous = std::clock();

		return result;
	}

}
