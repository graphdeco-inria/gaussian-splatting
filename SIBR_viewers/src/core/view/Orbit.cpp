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


#include "Orbit.hpp"
#include <boost/filesystem.hpp>
#include "core/graphics/Input.hpp"
#include "core/graphics/Viewport.hpp"
#include "core/raycaster/CameraRaycaster.hpp" 
#include "core/graphics/Window.hpp"
#include "core/graphics/Mesh.hpp"
#include "core/view/UIShortcuts.hpp"

# define SIBR_ORBIT_INTERPOLATE_FRAMES 900

namespace sibr {

	Orbit::Orbit(void) : _hasBeenInitialized(false), _orbitPointClicked(false) {
		UIShortcuts::global().add("[Orbit cam] alt+click", "Select new orbit center.");
		UIShortcuts::global().add("[Orbit cam] 4", "move left");
		UIShortcuts::global().add("[Orbit cam] 6", "move right");
		UIShortcuts::global().add("[Orbit cam] 8", "move down");
		UIShortcuts::global().add("[Orbit cam] 2", "move up");
		UIShortcuts::global().add("[Orbit cam] 7", "rotate left ");
		UIShortcuts::global().add("[Orbit cam] 9", "rotate right ");
		UIShortcuts::global().add("[Orbit cam] 1", "get closer");
		UIShortcuts::global().add("[Orbit cam] 3", "get further");
		UIShortcuts::global().add("[Orbit cam] 5", "flip up vector (look upside down)");
		UIShortcuts::global().add("[Orbit cam] alt+1-9", "automatic move");
		UIShortcuts::global().add("[Orbit cam] 0", "stop automatic move, restore previous cam");
		UIShortcuts::global().add("[Orbit cam] .", "stop automatic move, keep current cam");
	}

	void Orbit::update(const sibr::Input & input, const std::shared_ptr<sibr::Raycaster> raycaster) {
	
		if (!_hasBeenInitialized) { return; }

		if (raycaster != nullptr && input.mouseButton().isReleased(sibr::Mouse::Left)
			&& input.key().isActivated(sibr::Key::LeftAlt)) {
			updateOrbitParameters(input, raycaster);
		}

		const float sensibility = 64.0f;

		if (input.key().isActivated(sibr::Key::LeftAlt)) {
			//orbit.factor = 0;
			if (input.key().isReleased(sibr::Key::KPNum4) ||
				input.key().isReleased(sibr::Key::Num4) ||
				input.key().isReleased(sibr::Key::F)  // for laptops
				) {
				_orbit.status = OrbitParameters::FORWARD_X;
				_orbit.direction = OrbitParameters::ACW;
			}
			else if (input.key().isReleased(sibr::Key::KPNum6) ||
				input.key().isReleased(sibr::Key::Num6)) {
				_orbit.status = OrbitParameters::FORWARD_X;
				_orbit.direction = OrbitParameters::CW;
			}
			else if (input.key().isReleased(sibr::Key::KPNum2) ||
				input.key().isReleased(sibr::Key::Num2)) {
				_orbit.status = OrbitParameters::FORWARD_Y;
				_orbit.direction = OrbitParameters::ACW;
			}
			else if (input.key().isReleased(sibr::Key::KPNum8) ||
				input.key().isReleased(sibr::Key::Num8)) {
				_orbit.status = OrbitParameters::FORWARD_Y;
				_orbit.direction = OrbitParameters::CW;
			}
			else if (input.key().isReleased(sibr::Key::KPNum7) ||
				input.key().isReleased(sibr::Key::Num7)) {
				_orbit.status = OrbitParameters::FORWARD_Z;
				_orbit.direction = OrbitParameters::ACW;
			}
			else if (input.key().isReleased(sibr::Key::KPNum9) ||
				input.key().isReleased(sibr::Key::Num9)) {
				_orbit.status = OrbitParameters::FORWARD_Z;
				_orbit.direction = OrbitParameters::CW;
			}
		}
		else if ((input.key().isReleased(sibr::Key::KPNum0) || input.key().isReleased(sibr::Key::Num0))
			&& _orbit.status != OrbitParameters::STATIC) {
			_orbit.status = OrbitParameters::STATIC;
		}
		else if (input.key().isReleased(sibr::Key::KPDecimal)
			&& _orbit.status != OrbitParameters::STATIC) {
			_orbit.keepCamera = true;
		}
		else if (input.key().isActivated(sibr::Key::KPNum4) ||
			input.key().isActivated(sibr::Key::Num4)) {
			_orbit.theta = -(float)M_2_PI / sensibility;
		}
		else if (input.key().isActivated(sibr::Key::KPNum6) ||
			input.key().isActivated(sibr::Key::Num6)) {
			_orbit.theta = (float)M_2_PI / sensibility;
		}
		else if (input.key().isActivated(sibr::Key::KPNum2) ||
			input.key().isActivated(sibr::Key::Num2)) {
			_orbit.phi = -(float)M_2_PI / sensibility;
		}
		else if (input.key().isActivated(sibr::Key::KPNum8) ||
			input.key().isActivated(sibr::Key::Num8)) {
			_orbit.phi = (float)M_2_PI / sensibility;
		}
		else if (input.key().isActivated(sibr::Key::KPNum7) ||
			input.key().isActivated(sibr::Key::Num7)) {
			_orbit.roll = -(float)M_2_PI / sensibility;
		}
		else if (input.key().isActivated(sibr::Key::KPNum9) ||
			input.key().isActivated(sibr::Key::Num9)) {
			_orbit.roll = (float)M_2_PI / sensibility;
		}
		else if (input.key().isActivated(sibr::Key::KPNum1) ||
			input.key().isActivated(sibr::Key::Num1)) {
			_orbit.radius *= 0.98f;
		}
		else if (input.key().isActivated(sibr::Key::KPNum3) ||
			input.key().isActivated(sibr::Key::Num3)) {
			_orbit.radius *= 1.02f;
		}
		else if (input.key().isReleased(sibr::Key::KPNum5) ||
			input.key().isReleased(sibr::Key::Num5)) {
			if (_orbit.status == OrbitParameters::STATIC) {
				_orbit.flip();
				std::cout << "\t orbit flip ! " << std::endl;
			}
			else {
				if (_orbit.direction == OrbitParameters::CW) {
					_orbit.direction = OrbitParameters::ACW;
					std::cout << "\t orbit anti clockwise  " << std::endl;
				}
				else {
					_orbit.direction = OrbitParameters::CW;
					std::cout << "\t orbit clockwise  " << std::endl;
				}
			}
		}

		interpolateOrbit();
	}

	void Orbit::interpolateOrbit() {
		using namespace Eigen;

		float k = (_orbit.factor) / (float)(SIBR_ORBIT_INTERPOLATE_FRAMES);
		bool keepCam = _orbit.keepCamera;

		float theta = (_orbit.status == OrbitParameters::FORWARD_X ? (float)(SIBR_2PI  * k) : _orbit.theta);
		float phi = (_orbit.status == OrbitParameters::FORWARD_Y ? (float)(SIBR_2PI  * k) : _orbit.phi);
		float roll = (_orbit.status == OrbitParameters::FORWARD_Z ? (float)(SIBR_2PI  * k) : _orbit.roll);

		sibr::Vector3f dir = -(_orbit.zAxis);
		sibr::Quaternionf qRoll(AngleAxisf(roll, _orbit.zAxis));
		sibr::Quaternionf qTheta(AngleAxisf(theta, _orbit.yAxis));
		sibr::Quaternionf qPhi(AngleAxisf(phi, _orbit.xAxis));

		sibr::Vector3f center = _orbit.center;
		sibr::Vector3f Eye = center + _orbit.radius*((qTheta*qPhi)*(dir));
		sibr::Vector3f up(qRoll*_orbit.yAxis);

		sibr::Camera n(_orbit.initialCamera);
		n.setLookAt(Eye, center, up);
		n.aspect(_orbit.initialCamera.aspect());
	

		if (_orbit.status == OrbitParameters::STATIC || keepCam) {
			sibr::Quaternionf qTot = qTheta*qPhi*qRoll;
			_orbit.xAxis = qTot*_orbit.xAxis;
			_orbit.yAxis = qTot*_orbit.yAxis;
			_orbit.zAxis = qTot*_orbit.zAxis;

			_orbit.theta = 0;
			_orbit.phi = 0;
			_orbit.roll = 0;
		}
		else {
			_orbit.factor += _orbit.direction;
		}

		if (keepCam) {
			_orbit.status = OrbitParameters::STATIC;
			_orbit.keepCamera = false;
		}

		_currentCamera = sibr::InputCamera(n, _currentCamera.w(), _currentCamera.h());
	}

	void Orbit::updateOrbitParameters(const sibr::Input& input, std::shared_ptr<sibr::Raycaster> raycaster)
	{
	
		// Clicked pixel (might need to check against viewport ?)
		const float px = (float)input.mousePosition().x();
		const float py = (float)input.mousePosition().y();

		sibr::Vector3f dx;
		sibr::Vector3f dy;
		sibr::Vector3f upLeftOffset;

		sibr::CameraRaycaster::computePixelDerivatives(_currentCamera, dx, dy, upLeftOffset);
		const sibr::Vector3f worldPos = px*dx + py*dy + upLeftOffset;
		
		// Cast a ray.
		if (raycaster != nullptr) {
			sibr::Vector3f dir =  worldPos - _currentCamera.position();
			//sibr::Vector3f dir = sibr::CameraRaycaster::computeRayDir(_currentCamera, input.mousePosition().cast<float>()).normalized();
			sibr::RayHit hit = raycaster->intersect(sibr::Ray(_currentCamera.position(), dir));
	
			// If hit at the proxy surface, compute the corresponding worls position, save it.
			if (hit.hitSomething()) {
				_orbit.center = _currentCamera.position() + hit.dist()*dir.normalized();

				// \todo TODO: SR reimplement the fitting of planes by either passing cameras all the way down, or something else.
				//_orbit.planePointCams = computeFittingPlaneCameras(_orbit.center);
				_orbit.yAxis = _currentCamera.up(); // _orbit.planePointCams.xyz();

				//cheap trick to solve the ambiguity of the up direction
				if (_orbit.yAxis.dot(_currentCamera.up()) < 0) {
					_orbit.yAxis = -_orbit.yAxis;
				}

				_orbit.zAxis = dir.normalized();
				_orbit.xAxis = _orbit.yAxis.cross(_orbit.zAxis);
				_orbit.radius = (_orbit.initialCamera.position() - _orbit.center).norm();
				_orbit.initialCamera = _currentCamera;

				_orbitPointClicked = true;
			}

		}
		
	}

	void Orbit::update(const sibr::Input & input, const float deltaTime, const Viewport & viewport)
	{
		update(input);
	}

	const sibr::InputCamera & Orbit::getCamera( void ) const
	{
		if( !_hasBeenInitialized ){
			SIBR_ERR << " Orbit : camera not initialized before use" << std::endl
				<< "\t you should use either fromMesh(), fromCamera() or load() " << std::endl;
		}
		return _currentCamera;

	}

	void Orbit::fromCamera( const sibr::InputCamera & cam,  const std::shared_ptr<sibr::Raycaster> raycaster )
	{
		_orbit.initialCamera = cam;
		_currentCamera = cam;
		_hasBeenInitialized = true;

		// If no point has already been selected by the user, we simply pick it automatically by intersecting cam dir and the mesh.
		if (!_orbitPointClicked) {
			// We need to transfer the camera parameters to the orbit.
			updateOrbitParametersCentered(raycaster);
		}
	}

	void	Orbit::updateOrbitParametersCentered(const std::shared_ptr<sibr::Raycaster> raycaster)
	{
		if (raycaster != nullptr) {
			sibr::RayHit hit = raycaster->intersect(sibr::Ray(_currentCamera.position(), _currentCamera.dir()));
			// If hit at the proxy surface, compute the corresponding world position, save it.
			if (hit.hitSomething()) {
				sibr::Vector3f intersection(_currentCamera.position() + hit.dist()* _currentCamera.dir().normalized());
				_orbit.center = intersection;
				_orbit.yAxis = _currentCamera.up();
				_orbit.zAxis = _currentCamera.dir();
				_orbit.xAxis = _currentCamera.right();
				_orbit.radius = (_currentCamera.position() - _orbit.center).norm();
				_orbit.initialCamera = _currentCamera;
				//orbitPointClicked -->; don't set it, the center is picked automatically.
			}
		}
		
	}

	sibr::Vector4f Orbit::computeFittingPlaneCameras(sibr::Vector3f & clickedPoint, const std::vector<InputCamera::Ptr> & cams)
	{
		using namespace Eigen;

		std::vector<sibr::Vector3f> positions(cams.size());

		for (int i = 0; i<(int)cams.size(); ++i) {
			positions.at(i) = cams.at(i)->position();
		}
		positions.push_back(clickedPoint);

		std::vector<sibr::Vector3f> colors(positions.size(), sibr::Vector3f(1, 0, 0));

		MatrixXf data(3, positions.size());
		int posId = 0;
		for (auto & pos : positions) {
			data(0, posId) = pos.x();
			data(1, posId) = pos.y();
			data(2, posId) = pos.z();
			++posId;
		}

		sibr::Vector3f center = data.rowwise().mean();
		Eigen::MatrixXf dataCentered = data.colwise() - center;

		JacobiSVD<MatrixXf> svd(dataCentered, ComputeFullU | ComputeThinV);

		//the normal to the fitting plane is the eigenvector associated to the smallest eigenvalue (i.e. the direction in which the variance of all points is the smallest)
		sibr::Vector3f  normal = svd.matrixU().col(2);
		normal.normalize();
		sibr::Vector3f n(normal);

		//the fitting plane contains the mean point
		float d = -center.dot(normal);

		
		std::cout << " \t  plane ( clicked point + input cams ) : " << n.x() << "*x + " << n.y() << "*y + " << n.z() << "*z + " << d << std::endl;

		return sibr::Vector4f(n.x(), n.y(), n.z(), d);
	}
}