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


#include "FPSCamera.hpp"
#include <boost/filesystem.hpp>
#include "core/graphics/Input.hpp"
#include "core/graphics/Viewport.hpp"
#include "core/graphics/Window.hpp"
#include "core/view/UIShortcuts.hpp"
#include "core/graphics/GUI.hpp"


# define IBRVIEW_CAMSPEED 1.f

namespace sibr {

	FPSCamera::FPSCamera(void) : _hasBeenInitialized(false) 
	{ 
		UIShortcuts::global().add("[FPS camera] j", "rotate camera -Y (look left)");
		UIShortcuts::global().add("[FPS camera] l", "rotate camera +Y (look right)");
		UIShortcuts::global().add("[FPS camera] i", "rotate camera +X (look up)");
		UIShortcuts::global().add("[FPS camera] k", "rotate camera -X (look down)");
		UIShortcuts::global().add("[FPS camera] u", "rotate camera +Z ");
		UIShortcuts::global().add("[FPS camera] o", "rotate camera -Z ");
		UIShortcuts::global().add("[FPS camera] w", "move camera -Z (move forward)");
		UIShortcuts::global().add("[FPS camera] s", "move camera +Z (move backward)");
		UIShortcuts::global().add("[FPS camera] a", "move camera -X (strafe left)");
		UIShortcuts::global().add("[FPS camera] d", "move camera +X (strafe right)");
		UIShortcuts::global().add("[FPS camera] q", "move camera -Y (move down)");
		UIShortcuts::global().add("[FPS camera] e", "move camera +Y (move up)");
	/*
		_speedFpsCam = 1.0f;
		_speedRotFpsCam = 1.0f;
		_useAcceleration = true; */
		_speedFpsCam = 0.3f;
		_speedRotFpsCam = 1.0f;
		_useAcceleration = false; 
	}

	void FPSCamera::fromCamera( const sibr::InputCamera & cam)
	{
		_currentCamera = cam;
		_hasBeenInitialized = true;
	}

	void FPSCamera::update(const sibr::Input & input, float deltaTime) {
	
		if (!_hasBeenInitialized) { return; }
		// Read input and update camera.
		moveUsingWASD(input, deltaTime);
		moveUsingMousePan(input, deltaTime);
	}

	void FPSCamera::snap(const std::vector<InputCamera::Ptr> & cams){
		sibr::Vector3f sumDir(0.f, 0.f, 0.f);
		sibr::Vector3f sumUp(0.f, 0.f, 0.f);
		for (const auto& cam: cams)
		{
			float dist = 1.0f/std::max(1e-6f,distance(_currentCamera.position(), cam->position()));
			sumDir += dist * cam->dir();
			sumUp  += dist * cam->up();
		}
		Matrix4f m = lookAt(Vector3f(0, 0, 0), sumDir, sumUp);
		_currentCamera.rotation(quatFromMatrix(m));
	}

	void FPSCamera::update(const sibr::Input & input, const float deltaTime, const Viewport & viewport)
	{
		update(input, deltaTime);
	}

	const sibr::InputCamera & FPSCamera::getCamera( void ) const
	{
		if( !_hasBeenInitialized ){
			SIBR_ERR << " FPS Camera : camera not initialized before use" << std::endl
				<< "\t you should use either fromMesh(), fromCamera() or load() " << std::endl;
		}
		return _currentCamera;
	}

	void FPSCamera::setSpeed(const float speed, const float angular) {
		_speedFpsCam = speed;
		if(angular != 0.0f) {
			_speedRotFpsCam = angular;
		}
	}

	void FPSCamera::setGoalAltitude(const float& goalAltitude) {
		_goalAltitude = goalAltitude;
	}

	void FPSCamera::onGUI(const std::string& suffix) {
		if(ImGui::Begin(suffix.c_str())) {
			ImGui::PushScaledItemWidth(130);
			ImGui::Checkbox("Acceleration", &_useAcceleration);
			ImGui::SameLine();
			if(!_useAcceleration) {
				ImGui::InputFloat("Speed", &_speedFpsCam, 0.1f, 0.5f);
				ImGui::SameLine();
			}
			ImGui::InputFloat("Rot. speed", &_speedRotFpsCam, 0.1f, 0.5f);
			ImGui::PopItemWidth();
		}
		ImGui::End();
	}


	void FPSCamera::moveUsingWASD(const sibr::Input& input, float deltaTime)
	{


		if (input.key().isActivated(sibr::Key::LeftControl)) { return; }

		float camSpeed = 2.f * deltaTime		* IBRVIEW_CAMSPEED;
		if (_currentCamera.ortho()) {
			camSpeed *= 5.0f;
		}
		float camRotSpeed = 30.f * deltaTime	* IBRVIEW_CAMSPEED;
		//float camSpeed = 0.1f;
		//float camRotSpeed = 1.f;

		sibr::Vector3f move(0, 0, 0);

		move.x() -= input.key().isActivated(sibr::Key::A) ? camSpeed : 0.f;
		move.x() += input.key().isActivated(sibr::Key::D) ? camSpeed : 0.f;
		move.z() -= input.key().isActivated(sibr::Key::W) ? camSpeed : 0.f;
		move.z() += input.key().isActivated(sibr::Key::S) ? camSpeed : 0.f;
		move.y() -= input.key().isActivated(sibr::Key::Q) ? camSpeed : 0.f;
		move.y() += input.key().isActivated(sibr::Key::E) ? camSpeed : 0.f;

		// If the acceleration effect is enabled, we alter the speed along a move.
		if(_useAcceleration) {
			if (move.isNull() == true) {
				_speedFpsCam = 1.f;
			} else {
				_speedFpsCam *= 1.02f;
			}
		}


		sibr::Vector3f pivot(0, 0, 0);

		camRotSpeed *= _speedRotFpsCam;
		pivot[1] += input.key().isActivated(sibr::Key::J) ? camRotSpeed : 0.f;
		pivot[1] -= input.key().isActivated(sibr::Key::L) ? camRotSpeed : 0.f;
		pivot[0] -= input.key().isActivated(sibr::Key::K) ? camRotSpeed : 0.f;
		pivot[0] += input.key().isActivated(sibr::Key::I) ? camRotSpeed : 0.f;
		pivot[2] -= input.key().isActivated(sibr::Key::O) ? camRotSpeed : 0.f;
		pivot[2] += input.key().isActivated(sibr::Key::U) ? camRotSpeed : 0.f;

		if (_currentCamera.ortho()) {
			if (input.key().isActivated(sibr::Key::Z)) {
				_currentCamera.orthoRight(_currentCamera.orthoRight()/1.1f);
				_currentCamera.orthoTop(_currentCamera.orthoTop()/1.1f);
				_speedRotFpsCam /= 1.1f;
			}
			else if (input.key().isActivated(sibr::Key::X)) {
				_currentCamera.orthoRight(_currentCamera.orthoRight()*1.1f);
				_currentCamera.orthoTop(_currentCamera.orthoTop()*1.1f);
				_speedRotFpsCam *= 1.1f;
			}
		}

		// Try to keep the same altitude as cameras around.
		if (_goalAltitude != -1) {
			sibr::Vector3f worldUp(0., 0., 1.);
			const sibr::Vector3f custom_forward = _currentCamera.right().cross(worldUp);
			const sibr::Vector3f translation_right = (_speedFpsCam * move.x()) * _currentCamera.right();

			sibr::Vector3f translation = _speedFpsCam * (move.z() * custom_forward) + translation_right;
			//const float altitudeDiff = _goalAltitude - _currentCamera.position().z();
			translation[2] = _goalAltitude - _currentCamera.position().z();

			_currentCamera.translate(translation);
		}
		else {
			_currentCamera.translate(move * _speedFpsCam, _currentCamera.transform());
		}

		_currentCamera.rotate(pivot, _currentCamera.transform());
	}

	void FPSCamera::moveUsingMousePan( const sibr::Input& input, float deltaTime )
	{
		
		float speed = 0.05f*deltaTime;
		sibr::Vector3f move(
			input.mouseButton().isActivated(sibr::Mouse::Left)? input.mouseDeltaPosition().x()*speed : 0.f,
			input.mouseButton().isActivated(sibr::Mouse::Right)? input.mouseDeltaPosition().y()*speed : 0.f,
			input.mouseButton().isActivated(sibr::Mouse::Middle)? input.mouseDeltaPosition().y()*speed : 0.f
			);
		_currentCamera.translate(move, _currentCamera.transform());

	}
}
