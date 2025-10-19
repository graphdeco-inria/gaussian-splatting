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


#include "MeshViewer.h"

#include <core/graphics/Window.hpp>
#include <core/assets/InputCamera.hpp>
#include <core/graphics/Mesh.hpp>
#include <core/raycaster/Raycaster.hpp>
#include <core/view/InteractiveCameraHandler.hpp>

const std::string sibr::MeshRenderer::meshVertexShader =
"#version 420										\n"
"uniform mat4 MVP;									\n"
"layout(location = 0) in vec3 in_vertex;			\n"
"layout(location = 1) in vec3 in_color;				\n"
"layout(location = 3) in vec3 in_normal;			\n"
"out vec3 color;									\n"
"out vec3 normal;									\n"
"out vec3 vertex;									\n"
"void main(void) {									\n"
"	color = in_color;								\n"
"	normal = in_normal;								\n"
"	vertex = in_vertex;								\n"
"	gl_Position = MVP * vec4(in_vertex,1.0);		\n"
"}													\n";

const std::string sibr::MeshRenderer::meshFragmentShader =
"#version 420										\n"
"uniform vec3 light_pos;							\n"
"uniform vec3 forcedColor = vec3(0.7f,0.7f,0.7f);	\n"
"in vec3 color;										\n"
"in vec3 normal;									\n"
"in vec3 vertex;									\n"
"out vec4 out_color;								\n"
"void main(void) {									\n"
"	float kd = 0.3;							 		\n"
"	float ks = 0.2;							 		\n"
"	vec3 L = normalize(light_pos - vertex);							 		\n"
"	vec3 N = normalize(normal);							 			\n"
"	vec3 R = 2.0*dot(L,N)*N - N;							 		\n"
"	vec3 V = L;		//light pos = eye					 			\n"
"	vec3 diffuse = max(0.0, dot(L,N))*vec3(1, 1, 1);				\n"
"	vec3 specular = max(0.0, dot(R,V))*vec3(1, 1, 1);				\n"
"	out_color = vec4((1.0 - kd -ks)*forcedColor + kd*diffuse + ks*specular , 1.0);	 	\n"
"}																					\n";

sibr::MeshRenderer::MeshRenderer()
{
	initShaders();

	resetLinesAndPoints();
	
}

void sibr::MeshRenderer::render(const sibr::Camera & eye)
{
	glLineWidth(1.0f);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	for (auto & meshParam : listMeshes) {
		if (meshParam.mode == sibr::Mesh::LineRenderMode) {
			shaderLines.begin();
			mvpLines.set(eye.viewproj());
			lineColor.set(meshParam.color);
			meshParam.mesh->render(meshParam.depthTest, false, sibr::Mesh::LineRenderMode);
			shaderLines.end();
		} else {
			shaderMesh.begin();
			const sibr::Vector3f lightPos = eye.position();
			light_pos.set(lightPos);
			mvpMesh.set(eye.viewproj());
			forcedColor.set(meshParam.color);
			meshParam.mesh->render(meshParam.depthTest, meshParam.backFaceCulling, meshParam.mode);
			shaderMesh.end();
		}
	}


	if (lines.dirty) {
		updateMeshLines();
	}
	shaderLines.begin();
	mvpLines.set(eye.viewproj());
	lines.mesh->render(lines.depthTest, false, sibr::Mesh::LineRenderMode);
	shaderLines.end();

	float radiusW = 10.0f;
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glPointSize(radiusW);

	if (points.dirty) {
		updateMeshPoints();
	}
	shaderPoints.begin();
	mvpPoints.set(eye.viewproj());
	radiusScreen.set(radiusW);
	points.mesh->render(points.depthTest, false, sibr::Mesh::PointRenderMode);

	if (specialPoints.get() != nullptr) {
		specialPoints->render(false, false, sibr::Mesh::PointRenderMode);
	}

	shaderPoints.end();


}

void sibr::MeshRenderer::addMesh(std::shared_ptr<sibr::Mesh> meshPtr, sibr::Mesh::RenderMode mode)
{
	MeshParams mesh;
	mesh.mesh = meshPtr;
	mesh.mode = mode;
	listMeshes.push_back(mesh);
}

void sibr::MeshRenderer::addLines(const std::vector<sibr::Vector3f>& listPoints, const sibr::Vector3f & color)
{
	int nLines = (int)listPoints.size() / 2;
	for (int l = 0; l < nLines; ++l) {
		lines.points.push_back(listPoints[2 * l]);
		lines.points.push_back(listPoints[2 * l + 1]);
		lines.colors.push_back(color);
	}

	lines.dirty = true;
}

void sibr::MeshRenderer::addPoint(const sibr::Vector3f & point, const sibr::Vector3f & color)
{
	points.points.push_back(point);
	points.colors.push_back(color);
	points.dirty = true;
}

void sibr::MeshRenderer::addPoints(const std::vector<sibr::Vector3f>& list_points, const sibr::Vector3f & color)
{
	std::vector<sibr::Vector3f> colors(list_points.size(), color);
	
	points.points.reserve(points.points.size() + list_points.size());
	points.points.insert(points.points.end(), list_points.begin(), list_points.end());

	points.colors.reserve(points.colors.size() + colors.size());
	points.colors.insert(points.colors.end(), colors.begin(), colors.end());
	points.dirty = true;
}

void sibr::MeshRenderer::cleanPoints()
{
	points.points.resize(0);
	points.colors.resize(0);
	points.dirty = true;
}

void sibr::MeshRenderer::cleanLines()
{
	lines.points.resize(0);
	lines.colors.resize(0);
	lines.dirty = true;
}

void sibr::MeshRenderer::resetLinesAndPoints()
{
	lines.mesh = std::shared_ptr<sibr::Mesh>(new sibr::Mesh());
	points.mesh = std::shared_ptr<sibr::Mesh>(new sibr::Mesh());
	cleanLines();
	cleanPoints();
}

void sibr::MeshRenderer::resetMeshes()
{
	listMeshes.resize(0);
}

void sibr::MeshRenderer::initShaders()
{

	shaderMesh.init("meshShader", meshVertexShader, meshFragmentShader);
	mvpMesh.init(shaderMesh, "MVP");
	light_pos.init(shaderMesh, "light_pos");
	forcedColor.init(shaderMesh, "forcedColor");

	std::string lineVertexShader =
		"#version 420										\n"
		"uniform mat4 MVP;									\n"
		"layout(location = 0) in vec3 in_vertex;			\n"
		"layout(location = 1) in vec3 in_color;			\n"
		"out vec3 color_vert;								\n"
		"void main(void) {									\n"
		"	gl_Position = MVP * vec4(in_vertex,1.0);		\n"
		"	color_vert = in_color;							\n"
		"}													\n";

	std::string lineFragmentShader =
		"#version 420										\n"
		"in vec3 color_vert;								\n"
		"uniform vec3 color;								\n"
		"out vec4 out_color;								\n"
		"void main(void) {									\n"
		"	out_color = vec4( color_vert, 1.0 );	 				\n"
		"}													\n";

	shaderLines.init("LineShader", lineVertexShader, lineFragmentShader);
	mvpLines.init(shaderLines, "MVP");
	lineColor.init(shaderLines, "color");

	std::string pointVertexShader =
		"#version 420										\n"
		"uniform mat4 MVP;									\n"
		"uniform float radiusScreen;									\n"
		"layout(location = 0) in vec3 in_vertex;			\n"
		"layout(location = 1) in vec3 in_color;				\n"
		"out vec3 color_vert;								\n"
		"void main(void) {									\n"
		"	gl_Position = MVP * vec4(in_vertex,1.0);		\n"
		"	gl_PointSize = radiusScreen;							\n"
		"	color_vert = in_color;							\n"
		"}													\n";

	std::string pointFragmentShader =
		"#version 420										\n"
		"in vec3 color_vert;								\n"
		"out vec4 out_color;								\n"
		"void main(void) {									\n"
		"	out_color = vec4( color_vert, 1.0 );	 		\n"
		"}													\n";

	shaderPoints.init("PointShader", pointVertexShader, pointFragmentShader);
	mvpPoints.init(shaderPoints, "MVP");
	radiusScreen.init(shaderPoints, "radiusScreen");
}

void sibr::MeshRenderer::updateMeshPoints(void)
{
	std::vector<float> vertexBuffer;
	for (int vertex_id = 0; vertex_id<(int)points.points.size(); vertex_id++) {
		for (int c = 0; c<3; ++c) {
			vertexBuffer.push_back(points.points.at(vertex_id)[c]);
		}
	}

	points.mesh->vertices(vertexBuffer);
	points.mesh->colors(points.colors);

	points.dirty = false;
}

void sibr::MeshRenderer::updateMeshLines(void)
{
	std::vector<float> vertexBuffer;
	std::vector<uint> indicesBuffer(3 * (lines.points.size() / 2));
	std::vector<sibr::Vector3f> colors(lines.points.size());
	for (int vertex_id = 0; vertex_id<(int)lines.points.size(); vertex_id += 2) {
		for (int c = 0; c<3; ++c) {
			vertexBuffer.push_back(lines.points.at(vertex_id)[c]);
		}
		for (int c = 0; c<3; ++c) {
			vertexBuffer.push_back(lines.points.at(vertex_id + 1)[c]);
		}

		indicesBuffer.at(3 * (vertex_id / 2)) = vertex_id;
		indicesBuffer.at(3 * (vertex_id / 2) + 1) = vertex_id;
		indicesBuffer.at(3 * (vertex_id / 2) + 2) = vertex_id + 1;

		colors.at(vertex_id) = lines.colors.at(vertex_id / 2);
		colors.at(vertex_id + 1) = lines.colors.at(vertex_id / 2);
	}

	lines.mesh->vertices(vertexBuffer);
	lines.mesh->colors(colors);
	lines.mesh->triangles(indicesBuffer);

	lines.dirty = false;
}

sibr::MeshViewer::MeshViewer()
{
	renderer = std::make_shared<MeshRenderer>();
	interactCam = std::make_shared<sibr::InteractiveCameraHandler>(true);
	interactCam->setFPSCameraSpeed(1);
	interactCam->switchMode(sibr::InteractiveCameraHandler::InteractionMode::TRACKBALL);
	inChargeOfWindow = false;
	fpsCounter.init(sibr::Vector2f(10, 10));
}

sibr::MeshViewer::MeshViewer(const sibr::Vector2i & screenRes, const sibr::Mesh & mesh, bool launchRenderingLoop)
{
	window.reset(new Window(screenRes[0], screenRes[1], "MeshViewer" ));
	renderer = std::make_shared<MeshRenderer>();
	interactCam = std::make_shared<sibr::InteractiveCameraHandler>(new sibr::InteractiveCameraHandler());
	interactCam->setFPSCameraSpeed(1);
	interactCam->switchMode(sibr::InteractiveCameraHandler::InteractionMode::TRACKBALL);
	inChargeOfWindow = true;

	setMainMesh(mesh);

	if (launchRenderingLoop) {
		renderLoop();
	}
}

void sibr::MeshViewer::setMainMesh(const sibr::Mesh & mesh, sibr::Mesh::RenderMode mode, bool updateCam, bool setupRaycaster)
{
	setMainMesh(*window, mesh, mode, updateCam, setupRaycaster);
}

void sibr::MeshViewer::setMainMesh(sibr::Window & win, const sibr::Mesh & mesh, sibr::Mesh::RenderMode mode, bool updateCam, bool setupRaycaster)
{
	sibr::Mesh::Ptr meshGL = std::make_shared<sibr::Mesh>(true);
	meshGL->vertices(mesh.vertices());
	meshGL->triangles(mesh.triangles());
	if (mesh.hasNormals()) {
		meshGL->normals(mesh.normals());
	}

	renderer->resetMeshes();
	renderer->addMesh(meshGL, mode);

	if (updateCam) {
		interactCam->setup(meshGL, win.viewport());
		interactCam->getTrackball().fromMesh(*meshGL, win.viewport());
	}

	if (setupRaycaster) {
		raycaster = std::make_shared<sibr::Raycaster>();
		raycaster->init();
		raycaster->addMesh(*meshGL);
	}

	float radius;
	sibr::Vector3f pos;
	meshGL->getBoundingSphere(pos, radius);
	interactCam->setFPSCameraSpeed(radius/10.0f);

}

void sibr::MeshViewer::render()
{
	if (window.get()) {
		render(window->viewport(), interactCam->getCamera());
		window->swapBuffer();
	}
}

void sibr::MeshViewer::renderLoop(sibr::Window & window)
{
	bool doLoop = true;

	while (doLoop && window.isOpened() ) {
		sibr::Input::poll();

		if (sibr::Input::global().key().isPressed(sibr::Key::Escape)) {
			doLoop = false;
		}

		interactCam->update(sibr::Input::global(), 1 / 60.0f, window.viewport());
		
		window.viewport().bind();
		window.viewport().clear(sibr::Vector3f(0.9f, 0.9f, 0.9f));
		renderer->render(interactCam->getCamera());
		interactCam->onRender(window.viewport());

		window.swapBuffer();
	}

	
}

void sibr::MeshViewer::render(const sibr::Viewport & viewport, const sibr::Camera & eye )
{
	viewport.bind();
	viewport.clear(sibr::Vector3f(0.9f, 0.9f, 0.9f));
	renderer->render(eye);
	interactCam->onRender(viewport);
	fpsCounter.update(true);
}

void sibr::MeshViewer::render(const sibr::Viewport & viewport)
{
	render(viewport, interactCam->getCamera());
}

void sibr::MeshViewer::render(const sibr::Camera & eye)
{
	if (window.get()) {
		render(window->viewport(), eye);
		window->swapBuffer();
	}
}

void sibr::MeshViewer::renderLoop(std::shared_ptr<sibr::Window> otherWindow)
{

	if (!otherWindow.get() && !window->isOpened()) {
		return;
	}
	if (otherWindow.get() && !window.get() ) {
		window = otherWindow;
	}

	while (window->isOpened()) {
		sibr::Input::poll();

		if (sibr::Input::global().key().isPressed(sibr::Key::Escape)) {
			window->close();
		}

		interactCam->update(sibr::Input::global(), 1 / 60.0f, window->viewport());
		render();
	}

	reset();
}

void sibr::MeshViewer::renderLoop(const std::function<void(MeshViewer*)> & f, bool customRendering, bool doReset)
{
	bool doRender = true;
	while (doRender && window->isOpened()) {
		sibr::Input::poll();
		input = sibr::Input::global();
		if (input.key().isPressed(sibr::Key::Escape)) {
			doRender = false; 
			if (inChargeOfWindow) {
				window->close();
			}
		}

		interactCam->update(input,1/60.0f, window->viewport());

		f(this);

		if (!customRendering) {
			render();
		}
	}
	if(doReset) {
		reset();
	}
}

void sibr::MeshViewer::reset()
{
	if (inChargeOfWindow) {
		interactCam.reset();
		renderer.reset();
		raycaster.reset();
		window.reset();
	}
	
}

void sibr::MeshViewer::demo()
{
	sibr::Mesh::Ptr meshPtr = sibr::Mesh::getTestCube();

	sibr::MeshViewer meshViewer(sibr::Vector2i(1600, 1200), *meshPtr);

	meshViewer.renderer->addPoints(meshPtr->vertices(), sibr::Vector3f(0, 1, 0));
	
	for (const auto & tri : meshPtr->triangles()) {
		for (int k = 0; k < 3; ++k) {
			meshViewer.renderer->addLines( 
				{ meshPtr->vertices()[tri[k]], meshPtr->vertices()[tri[(k + 1) % 3]] },
				sibr::Vector3f(1, 0, 0));
		}
	}

	meshViewer.renderLoop();
}
