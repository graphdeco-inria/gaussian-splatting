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


#include "TrackBall.h"
#include <boost/filesystem.hpp>
#include "core/graphics/Input.hpp"
#include "core/graphics/Viewport.hpp"
#include "core/raycaster/CameraRaycaster.hpp" 
#include "core/graphics/Window.hpp"
#include "core/graphics/Mesh.hpp"

namespace sibr {

	float TrackBall::ratioTrackBall2D = 0.75f;

	TrackBall::TrackBall(bool _verbose) : hasBeenInitialized(false), shadersCompiled(false), state(TrackBallState::IDLE), verbose(_verbose),
		fixedCamera(InputCamera()), tempCamera(InputCamera())
	{
		drawThis = true;
	}

	void TrackBall::update(const sibr::Input& input, const float deltaTime, const Viewport& viewport) {
		update(input, viewport, std::shared_ptr<Raycaster>());
	}

	const InputCamera & TrackBall::getCamera(void) const
	{
		if (!hasBeenInitialized) {
			SIBR_ERR << " TrackBall : camera not initialized before use" << std::endl
				<< "\t you should use either fromMesh(), fromCamera() or load() " << std::endl;
		}
		if (state == TrackBallState::IDLE) {
			return fixedCamera;
		}
		else {
			return tempCamera;
		}
	}

	void TrackBall::onRender(const sibr::Viewport& viewport) {
		if (!drawThis) { return; }

		if (!shadersCompiled) {
			initTrackBallShader();
		}

		if (state == TrackBallState::IDLE) { return; }

		// Save current blending state and function.
		GLboolean blendState;
		glGetBooleanv(GL_BLEND, &blendState);
		GLint blendSrc, blendDst;
		glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrc);
		glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDst);

		// Enable basic blending.
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);

		// Render.
		viewport.bind();
		trackBallShader.begin();
		ratioTrackBall2Dgpu.set(ratioTrackBall2D);
		trackBallStateGPU.set((int)state);
		quadMesh->render(false, false, Mesh::RenderMode::FillRenderMode);
		trackBallShader.end();

		// Restore blend state.
		if (!blendState) {
			glDisable(GL_BLEND);
		}
		glBlendFunc(blendSrc, blendDst);
	}

	void TrackBall::saveVectorInFile(std::ofstream & s, const Vector3f & v) const {
		s << v.x() << " " << v.y() << " " << v.z() << std::endl;
	}

	void TrackBall::setCameraAttributes(const Viewport & viewport)
	{
		fixedCamera.size((int)viewport.finalWidth(), (int)viewport.finalHeight());
		fixedCamera.aspect(viewport.finalWidth() / viewport.finalHeight());
	}

	void TrackBall::updateTrackBallCameraSize(const Viewport & viewport)
	{
		sibr::Vector2i viewPortSize = viewport.finalSize().cast<int>();
		fixedCamera.size(viewPortSize[0], viewPortSize[1]);
	}

	bool TrackBall::load(std::string & filePath, const Viewport & viewport)
	{
		std::ifstream file(filePath.c_str());
		if (file.is_open()) {
			float a, b, c, fov, zNear, zFar;
			file >> a >> b >> c;
			Vector3f tbCenter(a, b, c);
			file >> a >> b >> c;
			Vector3f eye(a, b, c);
			file >> a >> b >> c;
			Vector3f up(a, b, c);
			file >> fov >> zNear >> zFar;

			tempCenter = fixedCenter = tbCenter;

			fixedCamera.setLookAt(eye, fixedCenter, up);
			fixedCamera.fovy(fov);
			fixedCamera.znear(zNear);
			fixedCamera.zfar(zFar);
			setCameraAttributes(viewport);
			tempCamera = fixedCamera;

			hasBeenInitialized = true;
			printMessage(" n trackBall loaded " + filePath);
			return true;
		}
		else {
			printMessage(" could not open trackBall" + filePath);
			return false;
		}
	}

	void TrackBall::save(std::string & filePath) const
	{
		if (boost::filesystem::exists(filePath)) {
			char c;
			SIBR_LOG << " a track ball already exists, override ? y/n ... " << std::flush;
			std::cin >> c;
			if (c != 'y') {
				std::cout << " not saved ! " << std::endl;
				return;
			}
		}
		std::ofstream file(filePath.c_str());
		if (file.is_open()) {
			saveVectorInFile(file, fixedCenter);
			saveVectorInFile(file, fixedCamera.position());
			saveVectorInFile(file, fixedCamera.up());
			file << fixedCamera.fovy() << " " << fixedCamera.znear() << " " << fixedCamera.zfar() << std::endl;
			SIBR_LOG << " TrackBall saved at " << filePath << std::endl;
		}
		else {
			SIBR_LOG << " Could not save trackBall" << std::endl;
		}
	}

	void TrackBall::fromCamera(const InputCamera & cam, const Viewport & viewport, const float & radius)
	{
		fixedCamera = cam;

		if (fixedCamera.zfar() == 0 || fixedCamera.znear() == 0) {
			InputCamera defaultCam = InputCamera();
			fixedCamera.znear(defaultCam.znear());
			fixedCamera.zfar(defaultCam.zfar());
		}

		setCameraAttributes(viewport);
		tempCamera = fixedCamera;
		tempCenter = fixedCenter = cam.position() + cam.dir().normalized() * radius;

		hasBeenInitialized = true;
	}

	bool TrackBall::fromMesh(const Mesh & mesh, const Viewport & viewport)
	{
		return fromBoundingBox(mesh.getBoundingBox(), viewport);
	}

	bool TrackBall::fromBoundingBox(const Eigen::AlignedBox<float, 3> & box, const Viewport & viewport)
	{

		if (box.isEmpty() || (box.diagonal().array() == 0.0f).any()) {
			SIBR_LOG << " [WARNING] TrackBall::fromMesh : cannot create camera from flat mesh " << std::endl;
			return false;
		}
		else {
			tempCenter = fixedCenter = box.center();
			Vector3f eye = fixedCenter + box.diagonal();
			Vector3f up(0, 1, 0);

			fixedCamera.setLookAt(eye, fixedCenter, up);

			fixedCamera.zfar(2.0f*box.diagonal().norm());
			setCameraAttributes(viewport);
			tempCamera = fixedCamera;
			hasBeenInitialized = true;
			printMessage(" TrackBall::fromMesh : camera created ");
			return true;
		}
	}

	void TrackBall::update(const Input & input, const Viewport & viewport, std::shared_ptr<Raycaster> raycaster)
	{
		if( !hasBeenInitialized || input.empty()) { return; }

		updateTrackBallCameraSize(viewport);

		updateTrackBallStatus(input, viewport);

		updateTrackBallCamera(input, viewport, raycaster);

		updateFromKeyboard(input);
	}

	void TrackBall::updateAspectWithViewport(const Viewport & viewport)
	{
		fixedCamera.size(static_cast<uint>(viewport.finalWidth()), static_cast<uint>(viewport.finalHeight()));
		fixedCamera.aspect(viewport.finalHeight() / viewport.finalWidth());
	}

	void TrackBall::updateTrackBallStatus(const Input & input, const Viewport & viewport)
	{
		currentPoint2D = input.mousePosition();

		if (input.key().isActivatedOnly(Key::T) && input.key().isActivatedOnly(Key::V)) {
			verbose = !verbose;
			if (verbose) {
				printMessage("trackBall is now verbose ");
			}
			else {
				SIBR_LOG << " TrackBall not verbose anymore " << std::endl;
			}
		}
		if (input.key().isActivated(Key::LeftControl)) {
			state = TrackBallState::IDLE;
		}
		else if (input.mouseButton().isPressed(Mouse::Right)) {
			lastPoint2D = currentPoint2D;
			tempCamera = fixedCamera;
			tempCenter = fixedCenter;
			if (isInTrackBall2dRegion(lastPoint2D, viewport)) {
				state = TrackBallState::TRANSLATION_PLANE;
			}
			else {
				state = TrackBallState::TRANSLATION_Z;
			}
		}
		else if (input.mouseButton().isPressed(Mouse::Left)) {
			lastPoint2D = currentPoint2D;
			tempCamera = fixedCamera;
			if (isInTrackBall2dRegion(lastPoint2D, viewport)) {
				state = TrackBallState::ROTATION_SPHERE;
			}
			else {
				state = TrackBallState::ROTATION_ROLL;
			}
		}
		else if (input.mouseButton().isReleased(Mouse::Right) || input.mouseButton().isReleased(Mouse::Left)) {
			if (state != TrackBallState::IDLE) {
				state = TrackBallState::IDLE;
				fixedCamera = tempCamera;
				fixedCenter = tempCenter;

			}
		}
	}

	void TrackBall::updateTrackBallCamera(const Input & input, const Viewport & viewport, std::shared_ptr<Raycaster> raycaster)
	{
		if (state == TrackBallState::ROTATION_SPHERE) {
			updateRotationSphere(input, viewport);
		}
		else if (state == TrackBallState::ROTATION_ROLL) {
			updateRotationRoll(input, viewport);
		}
		else if (state == TrackBallState::TRANSLATION_PLANE) {
			updateTranslationPlane(input, viewport, raycaster);
		}
		else if (state == TrackBallState::TRANSLATION_Z) {
			updateTranslationZ(input, viewport);
		}
		else if (state == TrackBallState::IDLE) {
			if (input.key().isActivated(Key::LeftControl)) {
				updateBallCenter(input, raycaster);
			}
			else if (input.mouseScroll() != 0) {
				updateZnearZFar(input);
				updateRadius(input);
			}
		}
	}

	void TrackBall::updateBallCenter(const Input & input, std::shared_ptr<Raycaster> raycaster)
	{

		if (raycaster.get() == nullptr || !input.mouseButton().isPressed(Mouse::Left)) {
			return;
		}

		sibr::Vector3f worldPos, dir;
		if(fixedCamera.ortho())
		{
			sibr::Vector2i clickPos = input.mousePosition();
			worldPos = fixedCamera.position() +
									(2.0f*clickPos.x() / (float)fixedCamera.w() - 1.0f)*fixedCamera.orthoRight()*fixedCamera.right()
									+ (2.0f*((float)fixedCamera.h() - 1 - clickPos.y()) / (float)fixedCamera.h() - 1.0f)*fixedCamera.orthoTop()*fixedCamera.up();
			dir = fixedCamera.dir();

		}
		else {
			dir = CameraRaycaster::computeRayDir(fixedCamera, input.mousePosition().cast<float>()).normalized();
			worldPos = fixedCamera.position();
		}
		RayHit hit = raycaster->intersect(Ray(worldPos, dir));

		if (hit.hitSomething()) {
			printMessage(" TrackBall::updateBallCenter : updating center from mesh ");
			Vector3f intersection(worldPos + hit.dist()*dir.normalized());
			fixedCenter = tempCenter = intersection;
			fixedCamera.setLookAt(worldPos, fixedCenter, fixedCamera.up());
		}
		else {
			printMessage(" TrackBall::updateBallCenter : could not intersect mesh ");
		}

	}

	void TrackBall::updateRotationSphere(const Input & input, const Viewport & viewport)
	{
		if (!isInTrackBall2dRegion(input.mousePosition(), viewport) || input.mousePosition() == lastPoint2D) { return; }
		Vector3f lastPointSphere(mapToSphere(lastPoint2D, viewport));
		Vector3f newPointSphere(mapToSphere(input.mousePosition(), viewport));
		Vector3f rotationAxisScreenSpace((lastPointSphere.cross(newPointSphere)).normalized());
		Vector4f axis;
		axis << rotationAxisScreenSpace, 0.0f;
		Vector3f rotationAxisWorldSpace((fixedCamera.view().inverse()* axis).xyz());

		float angleCos = newPointSphere.dot(lastPointSphere);
		if (std::abs(angleCos) < 1.0f) {
			float rotationAngle = -2.0f * acos(angleCos);
			Eigen::Quaternionf rot(Eigen::AngleAxisf(rotationAngle, rotationAxisWorldSpace));

			float radius = (fixedCamera.position() - fixedCenter).norm();
			Vector3f oldEye = -fixedCamera.dir().normalized();
			Vector3f newEye = fixedCenter + radius * (rot*oldEye);
			tempCamera.setLookAt(newEye, fixedCenter, fixedCamera.up());
		
		}
	}

	void TrackBall::updateRotationRoll(const Input & input, const Viewport & viewport)
	{
		if (isInTrackBall2dRegion(input.mousePosition(), viewport)) { return; }

		Vector2f viewportCenter(0.5f*(viewport.finalLeft() + viewport.finalRight()), 0.5f*(viewport.finalTop() + viewport.finalBottom()));
		float clockwise = (areClockWise(viewportCenter, lastPoint2D.cast<float>(), input.mousePosition().cast<float>()) ? -1.0f : 1.0f);
		float diagonal = std::sqrt((float)(viewport.finalWidth()*viewport.finalWidth() + viewport.finalHeight()*viewport.finalHeight()));
		float rollAngle = clockwise * (float)M_PI * (float)(lastPoint2D - input.mousePosition()).norm() / diagonal;

		Eigen::Quaternionf rot(Eigen::AngleAxisf(rollAngle, -fixedCamera.dir().normalized()));
		Vector3f newUp = rot * fixedCamera.up().normalized();

		tempCamera.setLookAt(fixedCamera.position(), fixedCenter, newUp);
	}

	void TrackBall::updateTranslationPlane(const Input & input, const Viewport & viewport, std::shared_ptr<Raycaster> raycaster)
	{
		if (!isInTrackBall2dRegion(input.mousePosition(), viewport)) { return; }

		if (input.mouseButton().isPressed(Mouse::Right)) {

			sibr::Vector3f worldPos, dir;
			if(fixedCamera.ortho())
			{
				sibr::Vector2i clickPos = input.mousePosition();
				worldPos = fixedCamera.position() +
										(2.0f*clickPos.x() / (float)fixedCamera.w() - 1.0f)*fixedCamera.orthoRight()*fixedCamera.right()
										+ (2.0f*((float)fixedCamera.h() - 1 - clickPos.y()) / (float)fixedCamera.h() - 1.0f)*fixedCamera.orthoTop()*fixedCamera.up();
				dir = fixedCamera.dir();
			
			}
			else {
				dir = CameraRaycaster::computeRayDir(fixedCamera, input.mousePosition().cast<float>()).normalized();
				worldPos = fixedCamera.position();
			}

			Vector3f pointOnPlane = fixedCenter;
			if (raycaster.get() != nullptr) {
				RayHit hit = raycaster->intersect(Ray(worldPos, dir));
				if (hit.hitSomething()) {
					pointOnPlane = worldPos + hit.dist()*dir;
				}
			}
			trackballPlane = Eigen::Hyperplane<float, 3>(fixedCamera.dir().normalized(), pointOnPlane);
		}

		Vector3f clicked3DPosition(mapTo3Dplane(lastPoint2D));
		Vector3f current3DPosition(mapTo3Dplane(input.mousePosition()));
		Vector3f shift3D = clicked3DPosition - current3DPosition;

		tempCenter = fixedCenter + shift3D / zoom;
		tempCamera.setLookAt(fixedCamera.position() + shift3D, tempCenter, fixedCamera.up());
	}

	void TrackBall::updateTranslationZ(const Input & input, const Viewport & viewport)
	{
		if (isInTrackBall2dRegion(input.mousePosition(), viewport)) { return; }
		Vector3f zAxis = -fixedCamera.dir().normalized();

		Vector2i shift2D(input.mousePosition() - lastPoint2D);
		Vector2f shift2Df(shift2D.cast<float>().array() / Vector2f(viewport.finalWidth(), viewport.finalHeight()).array());

		int whichDir = (std::abs(shift2D.x()) > std::abs(shift2D.y()) ? 0 : 1);

		float shift = 4.0f*(fixedCenter - fixedCamera.position()).norm()*(whichDir == 0 ? -1.0f : 1.0f)*shift2Df[whichDir];
		Vector3f shift3D = shift * zAxis;
		tempCenter = fixedCenter + shift3D / zoom;
		tempCamera.setLookAt(fixedCamera.position() + shift3D, tempCenter, fixedCamera.up());
	}

	void TrackBall::updateFromKeyboard(const Input & input)
	{
		float angle = 0.005f;
		float angleChange = 0.0f;
		enum Change { NONE, X, Y, Z };
		Change change = NONE;

		if (input.key().isActivated(sibr::Key::KPNum6)) {
			angleChange = +angle;
			change = Y;
		}
		if (input.key().isActivated(sibr::Key::KPNum4)) {
			angleChange = -angle;
			change = Y;
		}
		if (input.key().isActivated(sibr::Key::KPNum8)) {
			angleChange = -angle;
			change = X;
		}
		if (input.key().isActivated(sibr::Key::KPNum2)) {
			angleChange = +angle;
			change = X;
		}
		if (input.key().isActivated(sibr::Key::KPNum7)) {
			angleChange = -angle;
			change = Z;
		}
		if (input.key().isActivated(sibr::Key::KPNum9)) {
			angleChange = +angle;
			change = Z;
		}
		if (change != NONE) {
			Vector3f zAxis = -fixedCamera.dir().normalized();
			Vector3f yAxis = fixedCamera.up().normalized();
			Vector3f xAxis = fixedCamera.right().normalized();

			Vector3f rotAxis = (change == Z ? zAxis : change == Y ? yAxis : xAxis);

			Eigen::Quaternionf rot(Eigen::AngleAxisf(angleChange, rotAxis));
			sibr::Vector3f newEye = fixedCamera.position();
			sibr::Vector3f newUp = yAxis;
			if (change == Z) {
				newUp = rot * newUp;
			}
			else {
				newEye = rot * (newEye - fixedCenter) + fixedCenter;
			}

			fixedCamera.setLookAt(newEye, fixedCenter, newUp);
		}

	}

	void TrackBall::updateRadius(const Input & input)
	{
		if(input.key().getNumActivated() != 0){ return; }
		if (!fixedCamera.ortho()) {
			float zoomIn = (input.mouseScroll() > 0 ? -1.0f : 1.0f);
			float radius = (fixedCamera.position() - fixedCenter).norm();
			Vector3f oldEye = -fixedCamera.dir().normalized();
			radius = radius * pow(1.25f, zoomIn);
			Vector3f newEye = fixedCenter + radius * oldEye;
			fixedCamera.setLookAt(newEye, fixedCenter, fixedCamera.up());
		}
		else
		{
			float zoomIn = (input.mouseScroll() > 0.0f ? -1.0f : 1.0f);
			fixedCamera.orthoRight(fixedCamera.orthoRight() * pow(1.25f, zoomIn));
			fixedCamera.orthoTop(fixedCamera.orthoTop() * pow(1.25f, zoomIn));
			zoom /= pow(1.25f, zoomIn);
		}
	}

	void TrackBall::updateZnearZFar(const Input & input)
	{
		float direction = (input.mouseScroll() > 0 ? 1.0f : -1.0f);

		if (input.key().isActivatedOnly(Key::Z)) {
			fixedCamera.zfar(fixedCamera.zfar()* pow(1.25f, direction));
			printMessage(" zFar : " + std::to_string(fixedCamera.zfar()));
		}
		else if (input.key().isActivatedOnly(sibr::Key::Z) && input.key().isActivatedOnly(Key::LeftShift)) {
			fixedCamera.znear(fixedCamera.znear()* pow(1.25f, direction));
			printMessage(" zNear : " + std::to_string(fixedCamera.znear()));
		}
		tempCamera = fixedCamera;
	}

	bool TrackBall::isInTrackBall2dRegion(const Vector2i & pos2D, const Viewport & viewport) const
	{
		float pos_x = (lastPoint2D.x()) / viewport.finalWidth();
		float pos_y = (lastPoint2D.y()) / viewport.finalHeight();
		float min_ratio = 0.5f * (1.0f - TrackBall::ratioTrackBall2D);
		float max_ratio = 0.5f * (1.0f + TrackBall::ratioTrackBall2D);
		return pos_x >= min_ratio && pos_x <= max_ratio && pos_y >= min_ratio && pos_y <= max_ratio;
	}

	Vector3f TrackBall::mapToSphere(const Vector2i & pos2D, const Viewport & viewport) const
	{
		
		int xMin = (int)0;
		int xMax = (int)(viewport.finalRight() - viewport.finalLeft());
		int yMin = (int)0;
		int yMax = (int)(viewport.finalBottom() - viewport.finalTop());

		Vector2i clampPos = pos2D.cwiseMin(Vector2i(xMax, yMax)).cwiseMax(Vector2i(xMin, yMin));

		double x = clampPos.x() / (double)viewport.finalWidth() - 0.5;
		double y = 0.5 - clampPos.y() / (double)viewport.finalHeight();

		double sinx = sin(M_PI * x * 0.5);
		double siny = sin(M_PI * y * 0.5);
		double sinx2siny2 = sinx * sinx + siny * siny;

		return Vector3d(sinx, siny, sinx2siny2 < 1.0 ? sqrt(1.0 - sinx2siny2) : 0.0).cast<float>();
	}

	Vector3f TrackBall::mapTo3Dplane(const Vector2i & pos2D) const
	{
		sibr::Vector3f worldPos, dir;
		if(fixedCamera.ortho())
		{
			worldPos = fixedCamera.position() +
									(2.0f*pos2D.x() / (float)fixedCamera.w() - 1.0f)*fixedCamera.orthoRight()*fixedCamera.right()
									+ (2.0f*((float)fixedCamera.h() - 1 - pos2D.y()) / (float)fixedCamera.h() - 1.0f)*fixedCamera.orthoTop()*fixedCamera.up();
			dir = fixedCamera.dir();
			
		}
		else {
			dir = CameraRaycaster::computeRayDir(fixedCamera, pos2D.cast<float>()).normalized();
			worldPos = fixedCamera.position();
		}

		Eigen::ParametrizedLine<float, 3> line(worldPos, dir);
		return line.intersectionPoint(trackballPlane);
	}

	bool TrackBall::areClockWise(const Vector2f & a, const Vector2f & b, const Vector2f & c) const
	{
		Vector2f u((b - a).normalized());
		Vector2f v((c - b).normalized());
		Vector2f uOrtho(u.y(), -u.x());
		return v.dot(uOrtho) >= 0;
	}


	void TrackBall::initTrackBallShader(void)
	{
		quadMesh = std::shared_ptr<Mesh>(new Mesh(true));

		int corners[4][2] = { {-1,-1}, {-1,1}, {1,-1}, {1,1} };

		std::vector<float> vertexBuffer;
		for (int i = 0; i < 4; i++) {
			Vector3f corner((float)corners[i][0], (float)corners[i][1], 0.0f);
			for (int c = 0; c < 3; c++) {
				vertexBuffer.push_back(corner[c]);
			}
		}

		int indices[6] = { 0, 1, 3, 0, 2, 3 };
		std::vector<uint> indicesBuffer(&indices[0], &indices[0] + 6);

		quadMesh->vertices(vertexBuffer);
		quadMesh->triangles(indicesBuffer);

		std::string trackBallVertexShader =
			"#version 420										\n"
			"layout(location = 0) in vec3 in_vertex;			\n"
			"out vec2 uv_coord;									\n"
			"void main(void) {									\n"
			"	uv_coord = in_vertex.xy;						\n"
			"	gl_Position = vec4(in_vertex.xy,0.0, 1.0);		\n"
			"}													\n";

		std::string trackBallFragmentShader =
			"#version 420														\n"
			"uniform float ratio;												\n"
			"uniform int mState;												\n"
			"in vec2 uv_coord;													\n"
			"out vec4 out_color;												\n"
			"void main(void) {													\n"
			"	float minB = -ratio;	 										\n"
			"	float maxB = +ratio;											\n"
			"	float x = uv_coord.x;		 									\n"
			"	float y = uv_coord.y;		 									\n"
			"	bool fragOutside = ( x<minB || x>maxB || y<minB || y>maxB );	\n"
			"	if( mState == 1 ){												\n" //plane transl	
			"		vec2 d = abs(uv_coord ) - vec2(ratio,ratio);				\n"
			"		float v = min(max(d.x,d.y),0.0) + length(max(d,0.0));		\n"
			"		float a =  0.2 * exp( - 5000.0 *v*v );						\n"
			"		out_color = vec4(1.0,0.0,0.0,a);							\n"
			"	} else if (  mState == 2 && fragOutside  ){						\n" //zoom transl
			"		out_color = vec4(0.0,1.0,0.0,0.1);							\n"
			"	} else if (  mState == 3 ){										\n" //sphere rot	
			"		float d =  x*x + y*y - ratio*ratio;							\n"
			"		float a =  0.2 * exp( - 5000.0 *d*d );						\n"
			"		out_color = vec4(1.0,0.0,0.0,a);							\n"
			"	} else if (  mState == 4  ){									\n" //roll rot	
			"		float d =  x*x + y*y - 0.5*(ratio+1.0)*ratio*ratio;			\n"
			"		float a =  0.2 * exp( - 5000.0 *d*d );						\n"
			"		out_color = vec4(0.0,1.0,0.0,a);							\n"
			"	} else {														\n"
			"		out_color = vec4(0.0,0.0,0.0,0.0);							\n"
			"	}																\n"
			"}																	\n";

		trackBallShader.init("trackBallShader", trackBallVertexShader, trackBallFragmentShader);

		ratioTrackBall2Dgpu.init(trackBallShader, "ratio");
		trackBallStateGPU.init(trackBallShader, "mState");

		shadersCompiled = true;
	}

	void TrackBall::printMessage(const std::string & msg) const
	{
		if (verbose) {
			std::cout << msg << std::endl;
		}
	}

}