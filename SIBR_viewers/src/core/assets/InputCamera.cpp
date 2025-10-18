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


#include "core/assets/ActiveImageFile.hpp"
#include "core/assets/InputCamera.hpp"
#include <boost/algorithm/string.hpp>
#include <map>
#include "core/system/String.hpp"
#include "picojson/picojson.hpp"


// Colmap binary stuff
#include "colmapheader.h"
typedef uint32_t image_t;
typedef uint32_t camera_t;
typedef uint64_t point3D_t;
typedef uint32_t point2D_t;

#define SIBR_INPUTCAMERA_BINARYFILE_VERSION 10
#define IBRVIEW_TOPVIEW_SAVEVERSION "version002"
#define FOCAL_X_UNDEFINED -1

namespace sibr
{
	InputCamera::InputCamera(float f, float k1, float k2, int w, int h, int id) :
		_focal(f), _k1(k1), _k2(k2), _w(w), _h(h), _id(id), _active(true), _name(""), _focalx(FOCAL_X_UNDEFINED)
	{
		// Update fov and aspect ratio.
		float fov = 2.0f * atan(0.5f * h / f);
		float aspect = float(w) / float(h);

		Camera::aspect(aspect);
		Camera::fovy(fov);

		_id = id;
	}

	InputCamera::InputCamera(float fy, float fx, float k1, float k2, int w, int h, int id) :
		_focal(fy), _k1(k1), _k2(k2), _w(w), _h(h), _id(id), _active(true), _name(""), _focalx(fx)
	{
		// Update fov and aspect ratio.
		float fovY = 2.0f * atan(0.5f * h / fy);
		float fovX = 2.0f * atan(0.5f * w / fx);

		Camera::aspect(tan(fovX / 2) / tan(fovY / 2));
		Camera::fovy(fovY);

		_id = id;
	}


	InputCamera::InputCamera(int id, int w, int h, sibr::Matrix4f m, bool active) :
		_active(active)
	{
		Vector3f t;
		float r[9];

		for (int i = 0; i < 9; i++)  r[i] = m(3 + i);
		for (int i = 0; i < 3; i++)  t[i] = m(12 + i);

		_w = w;
		_h = h;

		_focal = m(0);
		_focalx = FOCAL_X_UNDEFINED;
		_k1 = m(1);
		_k2 = m(2);

		float fov = 2.0f * atan(0.5f * h / m(0));
		float aspect = float(w) / float(h);

		sibr::Matrix3f		matRotation;
		matRotation <<
			r[0], r[1], r[2],
			r[3], r[4], r[5],
			r[6], r[7], r[8]
			;

		Camera::aspect(aspect);
		Camera::fovy(fov);

		// http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
		// Do pos = -R' * t
		const sibr::Matrix3f orientation = matRotation.transpose();
		sibr::Vector3f position = -orientation * t;
		Camera::position(position);
		Camera::rotation(Quaternionf(orientation));


		Camera::principalPoint({ 0.5f, 0.5f });


		_id = id;
		_name = "";
	}



	InputCamera::InputCamera(int id, int w, int h, sibr::Vector3f& position, sibr::Matrix3f& orientation, float focal, float k1, float k2, bool active) :
		_active(active)
	{


		_w = w;
		_h = h;

		_focal = focal;
		_focalx = FOCAL_X_UNDEFINED;
		_k1 = k1;
		_k2 = k2;

		float fov = 2.0f * atan(0.5f * h / _focal);
		float aspect = float(w) / float(h);



		Camera::aspect(aspect);
		Camera::fovy(fov);

		// http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
		// Do pos = -R' * t

		Camera::position(position);
		Camera::rotation(Quaternionf(orientation));

		_id = id;
		_name = "";
	}

	InputCamera::InputCamera(const Camera& c, int w, int h) : Camera(c) {
		_focal = 1.0f / (tan(0.5f * fovy()) * 2.0f / float(h));
		_focalx = FOCAL_X_UNDEFINED;
		_k1 = _k2 = 0;
		_w = w;
		_h = h;
		_id = 0;
		_name = "";
		_active = true;
		aspect(float(_w) / float(_h));
	}

	// ------------------------------------------------------------------------

	void InputCamera::size(uint w, uint h) { _w = w; _h = h; }
	uint InputCamera::w(void)  const { return _w; }
	uint InputCamera::h(void)  const { return _h; }
	bool InputCamera::isActive(void)  const { return _active; }

	/* compatibility for preprocess (depth) */


	Vector3f InputCamera::projectScreen(const Vector3f& pt) const {
		Vector3f proj_pt = project(pt);
		Vector3f screen_pt((proj_pt[0] + 1.f) * _w / 2.0f, (1.f - proj_pt[1]) * _h / 2.0f, proj_pt[2] * 0.5f + 0.5f);

		return screen_pt;
	}

	float InputCamera::focal() const { return _focal; };
	float InputCamera::focalx() const { return _focalx; };
	float InputCamera::k1() const { return _k1; };
	float InputCamera::k2() const { return _k2; };

	InputCamera InputCamera::resizedH(int h) const {

		int w = int(_aspect * h);

		float sibr_focal = h * _focal / _h;
		float k1 = _k1;
		float k2 = _k2;
		int id = _id;

		sibr::Matrix4f m;

		sibr::InputCamera cam(sibr_focal, k1, k2, w, h, id);

		cam.rotation(rotation());
		cam.position(position());

		cam.znear(znear());
		cam.zfar(zfar());
		cam.name(name());

		return cam;
	}

	InputCamera InputCamera::resizedW(int w) const {

		int h = int(float(w) / _aspect);

		float sibr_focal = h * _focal / _h;
		float k1 = _k1;
		float k2 = _k2;
		int id = _id;

		sibr::Matrix4f m;

		sibr::InputCamera cam(sibr_focal, k1, k2, w, h, id);

		cam.rotation(rotation());
		cam.position(position());

		cam.znear(znear());
		cam.zfar(zfar());
		cam.name(name());

		return cam;
	}



	std::vector<InputCamera::Ptr> InputCamera::load(const std::string& datasetPath, float zNear, float zFar, const std::string& bundleName, const std::string& listName)
	{
		const std::string bundlerFile = datasetPath + "/cameras/" + bundleName;
		const std::string listFile = datasetPath + "/images/" + listName;
		const std::string clipFile = datasetPath + "/clipping_planes.txt";

		// Loading clipping planes if they are available.
		SIBR_LOG << "Loading clipping planes from " << clipFile << std::endl;

		struct Z {
			Z() {}
			Z(float f, float n) : far(f), near(n) {}
			float far;
			float near;
		};
		std::vector<Z> nearsFars;

		{ // Load znear & zfar for unprojecting depth samples

			float z;
			std::ifstream zfile(clipFile);
			// During preprocessing clipping planes are not yet defined
			// the preprocess utility "depth" defines this
			// SIBR_ASSERT(zfile.is_open());
			if (zfile.is_open()) {
				int num_z_values = 0;
				while (zfile >> z) {
					if (num_z_values % 2 == 0) {
						nearsFars.push_back(Z());
						nearsFars[nearsFars.size() - 1].near = z;
					}
					else {
						nearsFars[nearsFars.size() - 1].far = z;
					}
					++num_z_values;
				}

				if (num_z_values > 0 && num_z_values % 2 != 0) {
					nearsFars.resize(nearsFars.size() - 1);
				}

				if (nearsFars.size() == 0) {
					SIBR_WRG << " Could not extract at leat 2 clipping planes from '" << clipFile << "' ." << std::endl;
				}
			}
			else {
				SIBR_WRG << "Cannot open '" << clipFile << "' (not clipping plane loaded)." << std::endl;
			}

		}

		// Load cameras and images infos.
		SIBR_LOG << "Loading input cameras." << std::endl;
		auto cameras = InputCamera::loadBundle(bundlerFile, zNear, zFar, listFile);

		if (!nearsFars.empty()) {
			for (int cid = 0; cid < cameras.size(); ++cid) {
				const int zid = std::min(cid, int(nearsFars.size()) - 1);
				cameras[cid]->znear(nearsFars[zid].near);
				cameras[cid]->zfar(nearsFars[zid].far);
			}
		}

		// Load active images
		ActiveImageFile activeImageFile;
		activeImageFile.setNumImages((int)cameras.size());
		// load active image file and set (in)active
		if (activeImageFile.load(datasetPath + "/active_images.txt", false)) {
			for (int i = 0; i < (int)cameras.size(); i++) {
				if (!activeImageFile.active()[i])
					cameras[i]->setActive(false);
			}
		}

		// Load excluded images
		ActiveImageFile excludeImageFile;
		excludeImageFile.setNumImages((int)cameras.size());
		// load exclude image file and set *in*active
		if (excludeImageFile.load(datasetPath + "/exclude_images.txt", false)) {
			for (int i = 0; i < (int)cameras.size(); i++) {
				// Attn (GD): invert the meaning of active for exclude:
				// only file numbers explicitly in exclude_images are set
				// to active, and these are the only ones we set to *inactive*
				// should really create a separate class or have a flag "invert"
				if (excludeImageFile.active()[i])
					cameras[i]->setActive(false);
			}
		}
		return cameras;
	}

	std::vector<InputCamera::Ptr> InputCamera::loadNVM(const std::string& nvmPath, float zNear, float zFar, std::vector<sibr::Vector2u> wh)
	{
		std::ifstream in(nvmPath);
		std::vector<InputCamera::Ptr> cameras;

		if (in.is_open())
		{
			int rotation_parameter_num = 4;
			bool format_r9t = false;
			std::string token;
			if (in.peek() == 'N')
			{
				in >> token; //file header
				if (strstr(token.c_str(), "R9T"))
				{
					rotation_parameter_num = 9;    //rotation as 3x3 matrix
					format_r9t = true;
				}
			}

			int ncam = 0, npoint = 0, nproj = 0;
			// read # of cameras
			in >> ncam;  if (ncam <= 1) return std::vector<InputCamera::Ptr>();

			//read the camera parameters

			std::function<Eigen::Matrix3f(const double[9])> matrix = [](const double q[9])
			{

				Eigen::Matrix3f m;
				double qq = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
				double qw, qx, qy, qz;
				if (qq > 0)
				{
					qw = q[0] / qq;
					qx = q[1] / qq;
					qy = q[2] / qq;
					qz = q[3] / qq;
				}
				else
				{
					qw = 1;
					qx = qy = qz = 0;
				}
				m(0, 0) = float(qw * qw + qx * qx - qz * qz - qy * qy);
				m(0, 1) = float(2 * qx * qy - 2 * qz * qw);
				m(0, 2) = float(2 * qy * qw + 2 * qz * qx);
				m(1, 0) = float(2 * qx * qy + 2 * qw * qz);
				m(1, 1) = float(qy * qy + qw * qw - qz * qz - qx * qx);
				m(1, 2) = float(2 * qz * qy - 2 * qx * qw);
				m(2, 0) = float(2 * qx * qz - 2 * qy * qw);
				m(2, 1) = float(2 * qy * qz + 2 * qw * qx);
				m(2, 2) = float(qz * qz + qw * qw - qy * qy - qx * qx);

				return m;
			};

			for (int i = 0; i < ncam; ++i)
			{
				double f, q[9], c[3], d[2];
				in >> token >> f;
				for (int j = 0; j < rotation_parameter_num; ++j) in >> q[j];
				in >> c[0] >> c[1] >> c[2] >> d[0] >> d[1];

				std::string     image_path = sibr::parentDirectory(nvmPath) + "/" + token;
				sibr::Vector2i	resolution = sibr::IImage::imageResolution(image_path);

				if (resolution.x() < 0 || resolution.y() < 0)
				{
					std::cerr << "Could not get resolution for input image: " << image_path << std::endl;
					return std::vector<InputCamera::Ptr>();
				}

				int wIm = 1, hIm = 1;
				if (ncam == wh.size()) {
					wIm = wh[i].x();
					hIm = wh[i].y();
				}
				else {
					wIm = resolution.x();
					hIm = resolution.y();
				}

				//camera_data[i].SetFocalLength(f);
				cameras.emplace_back(new InputCamera((float)f, (float)d[0], (float)d[1], wIm, hIm, i));

				float fov = 2.0f * atan(0.5f * hIm / (float)f);
				float aspect = float(wIm) / float(hIm);
				cameras[i]->aspect(aspect);
				cameras[i]->fovy(fov);

				//translation
				Vector3f posCam((float)c[0], (float)c[1], (float)c[2]);

				if (format_r9t)
				{

					std::cout << " WARNING THIS PART OF THE CODE WAS NEVER TESTED. IT IS SUPPOSED NOT TO WORK PROPERLY" << std::endl;
					Eigen::Matrix3f		matRotation;
					matRotation <<
						float(q[0]), float(q[1]), float(q[2]),
						float(q[3]), float(q[4]), float(q[5]),
						float(q[6]), float(q[7]), float(q[8])
						;
					matRotation.transposeInPlace();


					cameras[i]->position(posCam);
					cameras[i]->rotation(Quaternionf(matRotation));

				}
				else
				{

					Eigen::Matrix3f converter;
					converter <<
						1, 0, 0,
						0, -1, 0,
						0, 0, -1;
					//older format for compability
					Quaternionf quat((float)q[0], (float)q[1], (float)q[2], (float)q[3]);
					Eigen::Matrix3f	matRotation = converter.transpose() * quat.toRotationMatrix();
					matRotation.transposeInPlace();

					cameras[i]->position(posCam);
					cameras[i]->rotation(Quaternionf(matRotation));

				}
				//camera_data[i].SetNormalizedMeasurementDistortion(d[0]);
				cameras[i]->name(token);
			}
			std::cout << ncam << " cameras; " << npoint << " 3D points; " << nproj << " projections\n";
		}
		else {
			SIBR_WRG << "Cannot open '" << nvmPath << std::endl;
		}

		return cameras;
	}

	std::vector<InputCamera::Ptr> InputCamera::loadLookat(const std::string& lookatPath, const std::vector<sibr::Vector2u>& wh, float znear, float zfar)
	{

		std::ifstream in(lookatPath);
		std::vector<InputCamera::Ptr> cameras;

		if (in.is_open())
		{
			int i = 0;
			for (std::string line; safeGetline(in, line); i++)
			{
				int w = 1024, h = 768;
				if (wh.size() > 0) {
					int whI = std::min(i, (int)wh.size() - 1);
					w = wh[whI].x();
					h = wh[whI].y();
				}
				else {
					std::cout << "Warning default image size of 1024*768 is supposed for camera" << std::endl;
				}

				bool use_fovx = false;
				std::string camName = line.substr(0, line.find(" "));
				size_t originPos = line.find("-D origin=") + 10;
				size_t targetPos = line.find("-D target=") + 10;
				size_t upPos = line.find("-D up=") + 6;
				size_t fovPos = line.find("-D fovy=") + 8;
				int delta_fov = 9;
				if (fovPos < 8) {
					std::cout << "Warning: Fovy not found, backing to Fovx mode" << std::endl;
					fovPos = line.find("-D fov=") + 7;
					use_fovx = true;
					delta_fov = 8;
				}
				size_t clipPos = line.find("-D clip=") + 8;
				size_t aspectPos = line.find("-D aspect=") + 10;
				size_t endPos = line.size();

				std::string originStr = line.substr(originPos, targetPos - originPos - 11);
				std::string targetStr = line.substr(targetPos, upPos - targetPos - 7);
				std::string upStr = line.substr(upPos, fovPos - upPos - delta_fov);
				std::string fovStr = line.substr(fovPos, clipPos - fovPos - 9);
				std::string clipStr = line.substr(clipPos, endPos - clipPos);

				std::vector<std::string> vecVal;
				boost::split(vecVal, originStr, [](char c) {return c == ','; });
				Vector3f Eye(std::strtof(vecVal[0].c_str(), 0), std::strtof(vecVal[1].c_str(), 0), std::strtof(vecVal[2].c_str(), 0));
				boost::split(vecVal, targetStr, [](char c) {return c == ','; });
				Vector3f At(std::strtof(vecVal[0].c_str(), 0), std::strtof(vecVal[1].c_str(), 0), std::strtof(vecVal[2].c_str(), 0));

				boost::split(vecVal, upStr, [](char c) {return c == ','; });
				Vector3f Up(std::strtof(vecVal[0].c_str(), 0), std::strtof(vecVal[1].c_str(), 0), std::strtof(vecVal[2].c_str(), 0));

				float fov = std::strtof(fovStr.c_str(), 0);

				boost::split(vecVal, clipStr, [](char c) {return c == ','; });
				Vector2f clip(std::strtof(vecVal[0].c_str(), 0), std::strtof(vecVal[1].c_str(), 0));

				Vector3f zAxis((Eye - At).normalized());
				Vector3f xAxis((Up.cross(zAxis)).normalized());
				Vector3f yAxis(zAxis.cross(xAxis));

				Vector3f transl(-Eye.dot(xAxis), -Eye.dot(yAxis), -Eye.dot(zAxis));

				Eigen::Matrix3f rotation;
				rotation << xAxis, yAxis, zAxis;
				rotation.transposeInPlace();

				Eigen::Matrix4f mLook;
				mLook.setZero();
				mLook.block<3, 3>(0, 0) = rotation;
				mLook.block<3, 1>(0, 3) = transl;
				mLook(3, 3) = 1;

				float fovRad = fov * float(M_PI) / 180;
				float sibr_focal = 0.5f * h / tan(fovRad / 2.0f); //Lookat file contain the vertical field of view now
				if (use_fovx) {
					sibr_focal = 0.5f * w / tan(fovRad / 2.0f); //Lookat file contain the vertical field of view now
				}

				Eigen::Matrix4f r(mLook);
				/*float m[15] = {
					sibr_focal,0,0,
					r(0,0),r(0,1),r(0,2),
					r(1,0),r(1,1),r(1,2),
					r(2,0),r(2,1),r(2,2),
					r(0,3),r(1,3),r(2,3)
				};
				*/
				Eigen::Matrix4f m;
				m(0) = sibr_focal;  m(1) = 0; m(2) = 0;
				m(3) = r(0, 0); m(4) = r(0, 1); m(5) = r(0, 2);
				m(6) = r(1, 0); m(7) = r(1, 1); m(8) = r(1, 2);
				m(9) = r(2, 0); m(10) = r(2, 1); m(11) = r(2, 2);
				m(12) = r(0, 3); m(13) = r(1, 3); m(14) = r(2, 3);


				bool isActive = true;

				cameras.emplace_back(new InputCamera((int)cameras.size(), w, h, m, isActive));

				if (znear > 0) {
					cameras[i]->znear(znear);
				}
				else {
					cameras[i]->znear(clip.x());
				}
				if (zfar > 0) {
					cameras[i]->zfar(zfar);
				}
				else {
					cameras[i]->zfar(clip.y());
				}
				cameras[i]->name(camName);
			}

		}
		else {
			SIBR_WRG << "Cannot open '" << lookatPath << std::endl;
		}

		return cameras;

	}

	std::string InputCamera::lookatString() const {
		std::string infos = std::string(" -D origin=") +
			std::to_string(position()[0]) +
			std::string(",") +
			std::to_string(position()[1]) +
			std::string(",") +
			std::to_string(position()[2]) +
			std::string(" -D target=") +
			std::to_string(position()[0] +
				dir()[0]) +
			std::string(",") +
			std::to_string(position()[1] +
				dir()[1]) +
			std::string(",") +
			std::to_string(position()[2] +
				dir()[2]) +
			std::string(" -D up=") +
			std::to_string(up()[0]) +
			std::string(",") +
			std::to_string(up()[1]) +
			std::string(",") +
			std::to_string(up()[2]) +
			std::string(" -D fovy=") +
			std::to_string(180 * fovy() / M_PI) +
			std::string(" -D clip=") +
			std::to_string(znear()) +
			std::string(",") +
			std::to_string(zfar()) +
			std::string("\n");
		return infos;
	}

	void InputCamera::saveAsLookat(const std::vector<InputCamera::Ptr>& cams, const std::string& fileName) {

		std::ofstream fileRender(fileName, std::ios::out | std::ios::trunc);
		for (const auto& cam : cams) {

			fileRender << cam->name() << cam->lookatString();
		}

		fileRender.close();
	}

	void InputCamera::saveImageSizes(const std::vector<InputCamera::Ptr>& cams, const std::string& fileName) {

		std::ofstream fileRender(fileName, std::ios::out | std::ios::trunc);
		for (const auto& cam : cams) {

			fileRender << cam->w() << "x" << cam->h() << "\n";
		}

		fileRender.close();
	}




	std::vector<InputCamera::Ptr> InputCamera::loadColmap(const std::string& colmapSparsePath, const float zNear, const float zFar, const int fovXfovYFlag)
	{
		const std::string camerasListing = colmapSparsePath + "/cameras.txt";
		const std::string imagesListing = colmapSparsePath + "/images.txt";

		const std::string camerasListing2 = colmapSparsePath + "/cameras.txt2";
		const std::string imagesListing2 = colmapSparsePath + "/images.txt2";

		std::ifstream camerasFile(camerasListing);
		std::ifstream imagesFile(imagesListing);
		std::ofstream camerasFile2(camerasListing2);
		std::ofstream imagesFile2(imagesListing2);
		if (!camerasFile.is_open()) {
			SIBR_ERR << "Unable to load camera colmap file" << std::endl;
		}
		if (!imagesFile.is_open()) {
			SIBR_WRG << "Unable to load images colmap file" << std::endl;
		}

		std::vector<InputCamera::Ptr> cameras;

		std::string line;

		struct CameraParametersColmap {
			size_t id;
			size_t width;
			size_t height;
			float  fx;
			float  fy;
			float  dx;
			float  dy;
		};

		std::map<size_t, CameraParametersColmap> cameraParameters;

		std::map<int, std::vector<std::string>> camidtokens;

		while (safeGetline(camerasFile, line)) {
			if (line.empty() || line[0] == '#') {
				continue;
			}

			std::vector<std::string> tokens = sibr::split(line, ' ');
			if (tokens.size() < 8) {
				SIBR_WRG << "Unknown line." << std::endl;
				continue;
			}
			if (tokens[1] != "PINHOLE" && tokens[1] != "OPENCV") {
				SIBR_WRG << "Unknown camera type." << std::endl;
				continue;
			}

			CameraParametersColmap params;
			params.id = std::stol(tokens[0]);
			params.width = std::stol(tokens[2]);
			params.height = std::stol(tokens[3]);
			params.fx = std::stof(tokens[4]);
			params.fy = std::stof(tokens[5]);
			params.dx = std::stof(tokens[6]);
			params.dy = std::stof(tokens[7]);

			cameraParameters[params.id] = params;

			camidtokens[params.id] = tokens;
		}

		// Now load the individual images and their extrinsic parameters
		sibr::Matrix3f converter;
		converter << 1, 0, 0,
			0, -1, 0,
			0, 0, -1;

		int camid = 0;
		int valid = 0;
		while (safeGetline(imagesFile, line)) {
			if (line.empty() || line[0] == '#') {
				continue;
			}
			std::vector<std::string> tokens = sibr::split(line, ' ');
			if (tokens.size() < 10) {
				SIBR_WRG << "Unknown line." << std::endl;
				continue;
			}

			uint		cId = std::stoi(tokens[0]) - 1;
			float       qw = std::stof(tokens[1]);
			float       qx = std::stof(tokens[2]);
			float       qy = std::stof(tokens[3]);
			float       qz = std::stof(tokens[4]);
			float       tx = std::stof(tokens[5]);
			float       ty = std::stof(tokens[6]);
			float       tz = std::stof(tokens[7]);
			size_t      id = std::stol(tokens[8]);

			std::string imageName = tokens[9];

			if (cameraParameters.find(id) == cameraParameters.end())
			{
				SIBR_ERR << "Could not find intrinsics for image: "
					<< tokens[9] << std::endl;
			}
			const CameraParametersColmap& camParams = cameraParameters[id];

			const sibr::Quaternionf quat(qw, qx, qy, qz);
			const sibr::Matrix3f orientation = quat.toRotationMatrix().transpose() * converter;
			sibr::Vector3f translation(tx, ty, tz);

			sibr::Vector3f position = -(orientation * converter * translation);

			sibr::InputCamera::Ptr camera;
			if (fovXfovYFlag) {
				camera = std::make_shared<InputCamera>(InputCamera(camParams.fy, camParams.fx, 0.0f, 0.0f, int(camParams.width), int(camParams.height), int(cId)));
			}
			else {
				camera = std::make_shared<InputCamera>(InputCamera(camParams.fy, 0.0f, 0.0f, int(camParams.width), int(camParams.height), int(cId)));
			}

			camera->name(imageName);
			camera->position(position);
			camera->rotation(sibr::Quaternionf(orientation));
			camera->znear(zNear);
			camera->zfar(zFar);

			if (camera->position().x() < 0)
			{
				camerasFile2 << ++valid;
				for (int i = 1; i < camidtokens[id].size(); i++)
					camerasFile2 << " " << camidtokens[id][i];
				camerasFile2 << "\n\n";

				imagesFile2<< valid;
				for (int i = 1; i < tokens.size() - 1; i++)
					imagesFile2 << " " << tokens[i];
				imagesFile2 << " " << valid << std::endl;
				imagesFile2 << "\n\n";
			}

			cameras.push_back(camera);

			++camid;
			// Skip the observations.
			safeGetline(imagesFile, line);
		}


		return cameras;
	}

	std::vector<InputCamera::Ptr> InputCamera::loadBundle(const std::string& bundlerPath, float zNear, float zFar, const std::string& listImagePath, bool path)
	{
		SIBR_LOG << "Loading input cameras." << std::endl;

		// check bundler file
		std::ifstream bundle_file(bundlerPath);
		if (!bundle_file.is_open()) {
			SIBR_ERR << "Unable to load bundle file at path \"" << bundlerPath << "\"." << std::endl;
			return {};
		}

		const std::string listImages = listImagePath.empty() ? (bundlerPath + "/../list_images.txt") : listImagePath;
		std::ifstream list_images(listImages);
		if (!list_images.is_open()) {
			SIBR_ERR << "Unable to load list_images file at path \"" << listImages << "\"." << std::endl;
			return {};
		}

		// read number of images
		std::string line;
		getline(bundle_file, line);	// ignore first line - contains version
		int numImages = 0;
		bundle_file >> numImages;	// read first value (number of images)
		getline(bundle_file, line);	// ignore the rest of the line

									// Read all filenames
		struct ImgInfos
		{
			std::string name;
			int id;
			int w, h;
		};
		std::vector<ImgInfos>	imgInfos;
		{
			ImgInfos				infos;
			while (true)
			{
				list_images >> infos.name;
				if (infos.name.empty()) break;
				list_images >> infos.w >> infos.h;
				infos.name.erase(infos.name.find_last_of("."), std::string::npos);
				infos.id = atoi(infos.name.c_str());
				imgInfos.push_back(infos);
				infos.name.clear();
			}
		}

		ImgInfos infoPrevImage;
		bool shortListImages = false;
		// check if list images has the same number of cameras as path, else assume we read the dataset list_images.txt
		if (path && imgInfos.size() != numImages)
			shortListImages = true;




		std::vector<InputCamera::Ptr> cameras(numImages);
		//  Parse bundle.out file for camera calibration parameters
		for (int i = 0, infosId = 0; i < numImages; i++) {

			ImgInfos infos;
			std::string camName;

			if (!shortListImages) {
				infoPrevImage = infos = imgInfos[infosId];
				camName = infos.name;
				if (infosId > imgInfos.size())
					break;
			}
			else {
				// hack; use info of last available image, but (always) change name
				if( i < imgInfos.size())
					infoPrevImage = infos = imgInfos[infosId];
				else 
					infos = infoPrevImage;

				std::stringstream ss;
				ss << std::setw(10) << std::setfill('0') << i;
				std::string s = ss.str();
				camName = std::string("path_camera") + s;
			}

			Matrix4f m; // bundler params

			bundle_file >> m(0) >> m(1) >> m(2) >> m(3) >> m(4);
			bundle_file >> m(5) >> m(6) >> m(7) >> m(8) >> m(9);
			bundle_file >> m(10) >> m(11) >> m(12) >> m(13) >> m(14);

			cameras[infosId] = InputCamera::Ptr(new InputCamera(infosId, infos.w, infos.h, m, true));
			cameras[infosId]->name(camName);
			cameras[infosId]->znear(zNear); cameras[infosId]->zfar(zFar);

			++infosId;
		}

		return cameras;
	}

	std::vector<InputCamera::Ptr> InputCamera::loadBundleFRIBR(const std::string& bundlerPath, float zNear, float zFar, const std::string& listImagePath)
	{
		SIBR_LOG << "Loading input cameras." << std::endl;

		// check bundler file
		std::ifstream bundle_file(bundlerPath);
		if (!bundle_file.is_open()) {
			SIBR_ERR << "Unable to load bundle file at path \"" << bundlerPath << "\"." << std::endl;
			return {};
		}


		// read number of images
		std::string line;
		getline(bundle_file, line);	// ignore first line - contains version
		int numImages = 0;
		bundle_file >> numImages;	// read first value (number of images)
		getline(bundle_file, line);	// ignore the rest of the line

		std::vector<InputCamera::Ptr> cameras(numImages);

		Eigen::Matrix3f to_cv, converter;
		to_cv << 1.0f, 0.0f, 0.0f,
			0.0f, -1.0f, 0.0f,
			0.0f, 0.0f, -1.0f;
		converter <<
			1, 0, 0,
			0, -1, 0,
			0, 0, -1;
		//  Parse bundle.out file for camera calibration parameters
		for (int i = 0; i < numImages; i++) {

			float f, k1, k2;
			bundle_file >> f >> k1 >> k2;

			float r00, r01, r02;
			float r10, r11, r12;
			float r20, r21, r22;
			bundle_file >> r00 >> r01 >> r02
				>> r10 >> r11 >> r12
				>> r20 >> r21 >> r22;

			Eigen::Matrix3f rotation;
			rotation(0, 0) = r00;
			rotation(0, 1) = r01;
			rotation(0, 2) = r02;
			rotation(1, 0) = r10;
			rotation(1, 1) = r11;
			rotation(1, 2) = r12;
			rotation(2, 0) = r20;
			rotation(2, 1) = r21;
			rotation(2, 2) = r22;

			sibr::Matrix3f orientation = (to_cv * rotation).transpose();

			float tx, ty, tz;
			bundle_file >> tx >> ty >> tz;
			sibr::Vector3f position = -orientation * (to_cv * Eigen::Vector3f(tx, ty, tz));

			std::stringstream pad_stream;
			pad_stream << std::setfill('0') << std::setw(10) << i - 2 << ".png";
			std::string     image_path = sibr::parentDirectory(bundlerPath) + "/" + listImagePath + pad_stream.str();

			sibr::Vector2u resolution(2, 2);
			sibr::ImageRGB temp;
			if (!temp.load(image_path)) {

				pad_stream.str("");
				pad_stream << std::setfill('0') << std::setw(8) << i << ".jpg";
				image_path = sibr::parentDirectory(bundlerPath) + "/" + listImagePath + pad_stream.str();
				temp.load(image_path);
			}
			resolution = temp.size();

			if (resolution.x() < 0 || resolution.y() < 0)
			{
				std::cerr << "Could not get resolution for calibrated camera: " << image_path << std::endl;
				return {};
			}

			float dx = resolution.x() * 0.5f;
			float dy = resolution.y() * 0.5f;

			orientation = /*converter.transpose() **/ orientation * converter;
			position = /*converter.transpose() **/ position;

			cameras[i] = InputCamera::Ptr(new InputCamera(i, resolution.x(), resolution.y(), position, orientation, f, k1, k2, true));
			cameras[i]->name(pad_stream.str());
			cameras[i]->znear(zNear); cameras[i]->zfar(zFar);

		}

		return cameras;
	}

	std::vector<InputCamera::Ptr> InputCamera::loadMeshroom(const std::string& meshroomSFMPath, const float zNear, const float zFar)
	{

		std::string file_path = meshroomSFMPath + "/cameras.sfm";

		std::ifstream json_file(file_path, std::ios::in);

		if (!json_file)
		{
			std::cerr << "file loading failed: " << file_path << std::endl;
			return std::vector<InputCamera::Ptr>();
		}

		std::vector<InputCamera::Ptr> cameras;

		picojson::value v;
		picojson::set_last_error(std::string());
		std::string err = picojson::parse(v, json_file);
		if (!err.empty()) {
			picojson::set_last_error(err);
			json_file.setstate(std::ios::failbit);
		}

		picojson::array& views = v.get("views").get<picojson::array>();
		picojson::array& intrinsincs = v.get("intrinsics").get<picojson::array>();
		picojson::array& poses = v.get("poses").get<picojson::array>();

		int numCameras = int(poses.size());
		//meras.resize(numCameras);

		sibr::Matrix3f converter;
		converter << 1.0f, 0, 0,
			0, -1, 0,
			0, 0, -1;

		size_t pose_idx, view_idx, intrinsic_idx;
		std::vector<std::string> splitS;


		for (size_t i = 0; i < numCameras; ++i)
		{

			Matrix4f m;
			//std::vector<std::string> splitS;

			pose_idx = i;
			std::string pose_id = poses[pose_idx].get("poseId").get<std::string>();

			for (size_t j = 0; j < views.size(); j++) {
				if (pose_id.compare(views[j].get("poseId").get<std::string>()) == 0) {
					view_idx = j;
					break;
				}
			}

			std::string intrinsics_id = views[view_idx].get("intrinsicId").get<std::string>();

			for (size_t k = 0; k < intrinsincs.size(); k++) {
				if (intrinsics_id.compare(intrinsincs[k].get("intrinsicId").get<std::string>()) == 0) {
					intrinsic_idx = k;
					break;
				}
			}

			m(0) = std::stof(intrinsincs[intrinsic_idx].get("pxFocalLength").get<std::string>());
			float dx = std::stof(intrinsincs[intrinsic_idx].get("principalPoint").get<picojson::array>()[0].get<std::string>());
			float dy = std::stof(intrinsincs[intrinsic_idx].get("principalPoint").get<picojson::array>()[1].get<std::string>());

			//std::stof(intrinsincs[intrinsic_idx].get("distortionParams").get<picojson::array>()[0].get<std::string>());
			m(1) = dx;
			//std::stof(intrinsincs[intrinsic_idx].get("distortionParams").get<picojson::array>()[1].get<std::string>());
			m(2) = dy;

			std::string camName = pose_id + ".exr";
			int width = std::stoi(views[view_idx].get("width").get<std::string>());
			int height = std::stoi(views[view_idx].get("height").get<std::string>());

			uint camId = uint(i);

			picojson::array& center = poses[pose_idx].get("pose").get("transform").get("center").get<picojson::array>();
			picojson::array& rotation = poses[pose_idx].get("pose").get("transform").get("rotation").get<picojson::array>();

			std::vector<Eigen::Vector3f> rows;
			Eigen::Vector3f row;
			Eigen::Vector3f position(std::stof(center[0].get<std::string>()), std::stof(center[1].get<std::string>()), std::stof(center[2].get<std::string>()));
			Eigen::Matrix3f orientation;

			for (int ii = 0; ii < 3; ++ii) {
				for (int jj = 0; jj < 3; ++jj)
					row(jj) = std::stof(rotation[jj + ii * 3].get<std::string>());
				rows.push_back(row);
			}

			orientation.row(0) = rows[0];
			orientation.row(1) = rows[1];
			orientation.row(2) = rows[2];
			orientation = orientation * converter;

			for (int ii = 0; ii < 9; ii++) {
				m(3 + ii) = orientation(ii);
			}

			const sibr::Vector3f finTrans = -orientation.transpose() * position;
			for (int ii = 0; ii < 3; ii++) {
				m(12 + ii) = finTrans[ii];
			}

			sibr::InputCamera::Ptr cam = std::make_shared<InputCamera>(camId, width, height, m, true);
			cam->name(camName);
			cam->znear(zNear);
			cam->zfar(zFar);
			cameras.push_back(cam);

		}
		return cameras;
	}

	Vector3f			InputCamera::unprojectImgSpaceInvertY(const sibr::Vector2i& pixelPos, const float& depth) const
	{
		sibr::Vector2f pos2dGL(2.0f * ((pixelPos.cast<float>() + sibr::Vector2f(0.5, 0.5)).cwiseQuotient(sibr::Vector2f(w(), h()))) - sibr::Vector2f(1, 1));  //to [-1,1]
		pos2dGL.y() = -pos2dGL.y();
		return unproject(sibr::Vector3f(pos2dGL.x(), pos2dGL.y(), depth));
	}

	Vector3f			InputCamera::projectImgSpaceInvertY(const Vector3f& point3d) const
	{
		sibr::Vector3f pos2dGL = project(point3d);
		pos2dGL.y() = -pos2dGL.y();
		sibr::Vector2f pos2dImg = (0.5f * (pos2dGL.xy() + sibr::Vector2f(1, 1))).cwiseProduct(sibr::Vector2f(w(), h()));
		return sibr::Vector3f(pos2dImg.x(), pos2dImg.y(), pos2dGL.z());
	}

	bool				InputCamera::loadFromBinary(const std::string& filename)
	{
		ByteStream	bytes;

		if (bytes.load(filename))
		{
			uint8	version;
			float	focal;
			float	k1;
			float	k2;
			uint16	w;
			uint16	h;
			Vector3f	pos;
			Quaternionf	rot;
			float		fov;
			float		aspect;
			float		znear;
			float		zfar;

			bytes
				>> version;

			if (version != SIBR_INPUTCAMERA_BINARYFILE_VERSION)
			{
				// Maybe the file format has been updated, or your binary file is not about InputCamera...
				SIBR_ERR << "incorrect file format (version number does not correspond)." << std::endl;

				return false;
			}

			bytes
				>> focal >> k1 >> k2 >> w >> h
				>> pos.x() >> pos.y() >> pos.z()
				>> rot.w() >> rot.x() >> rot.y() >> rot.z()
				>> fov >> aspect >> znear >> zfar
				;

			_focal = focal;
			_k1 = k1;
			_k2 = k2;
			_w = (uint)w;
			_h = (uint)h;
			Camera::position(pos);
			Camera::rotation(rot);
			Camera::fovy(fov);
			Camera::aspect(aspect);
			Camera::znear(znear);
			Camera::zfar(zfar);

			return true;
		}
		else
		{
			SIBR_WRG << "cannot open file '" << filename << "'." << std::endl;
		}
		return false;
	}

	void				InputCamera::saveToBinary(const std::string& filename) const
	{
		ByteStream	bytes;

		uint8	version = SIBR_INPUTCAMERA_BINARYFILE_VERSION;
		float	focal = _focal;
		float	k1 = _k1;
		float	k2 = _k2;
		uint16	w = (uint16)_w;
		uint16	h = (uint16)_h;
		Vector3f	pos = position();
		Quaternionf	rot = rotation();
		float		fov = _fov;
		float		aspect = _aspect;
		float		znear = _znear;
		float		zfar = _zfar;

		bytes
			<< version
			<< focal << k1 << k2 << w << h
			<< pos.x() << pos.y() << pos.z()
			<< rot.w() << rot.x() << rot.y() << rot.z()
			<< fov << aspect << znear << zfar
			;

		bytes.saveToFile(filename);
	}

	void InputCamera::readFromFile(std::istream& infile)
	{
		std::string version;
		infile >> version;
		if (version != IBRVIEW_TOPVIEW_SAVEVERSION)
		{
			SIBR_WRG << "Sorry but your TopView camera configuration "
				"is too old (we added new features since!)" << std::endl;
			return;
		}

		Vector3f v;
		infile >> v.x() >> v.y() >> v.z();
		Quaternionf q;
		infile >> q.x() >> q.y() >> q.z() >> q.w();
		set(v, q);
	}

	void InputCamera::writeToFile(std::ostream& outfile) const
	{
		outfile << IBRVIEW_TOPVIEW_SAVEVERSION "\n";
		Vector3f v = transform().position();
		Quaternionf q = transform().rotation();
		outfile << " " << v.x() << " " << v.y() << " " << v.z() << " ";
		outfile << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
	}

	std::string InputCamera::toBundleString(bool negativeZ, bool recomputeFocal) const {

		std::stringstream ss;
		ss << std::setprecision(16);
		float focal;
		if( recomputeFocal )
			focal = 0.5f * h() / tan(fovy() / 2.0f); // We cannot set the focal but we need to compute it
		else
			focal = _focal;

		Eigen::Matrix3f r = transform().rotation().toRotationMatrix();
		sibr::Vector3f t = -transform().rotation().toRotationMatrix().transpose() * position();

		ss << focal << " " << k1() << " " << k2() << "\n"; // The focal is set to zero in the loading module we use fov=2.0f * atan( 0.5f*h / focal) here
		if (!negativeZ) {
			ss << r(0) << " " << r(1) << " " << r(2) << "\n";
			ss << r(3) << " " << r(4) << " " << r(5) << "\n";
			ss << r(6) << " " << r(7) << " " << r(8) << "\n";
			ss << t(0) << " " << t(1) << " " << t(2) << "\n";
		}
		else {
			ss << r(0) << " " << -r(2) << " " << r(1) << "\n";
			ss << r(3) << " " << -r(5) << " " << r(4) << "\n";
			ss << r(6) << " " << -r(8) << " " << r(7) << "\n";
			ss << t(0) << " " << t(1) << " " << t(2) << "\n";
		}

		return ss.str();
	}

	std::vector<sibr::Vector2i> InputCamera::getImageCorners() const
	{
		return { {0,0}, {_w - 1, 0}, {_w - 1,_h - 1}, {0, _h - 1} };
	}

	void InputCamera::saveAsBundle(const std::vector<InputCamera::Ptr>& cams, const std::string& fileName, bool negativeZ, const bool exportImages, bool oldFocal) {

		std::ofstream outputBundleCam;
		outputBundleCam.open(fileName);
		outputBundleCam << "# Bundle file v0.3" << std::endl;
		outputBundleCam << cams.size() << " " << 0 << std::endl;

		for (int c = 0; c < cams.size(); c++) {
			auto& camIm = cams[c];
			outputBundleCam << camIm->toBundleString(negativeZ, oldFocal);
		}

		outputBundleCam.close();

		// Export the images list and empty images (useful for fribr).
		if (exportImages) {
			std::ofstream outList;
			const std::string listpath = fileName + "/../list_images.txt";
			const std::string imagesDir = fileName + "/../visualize/";
			sibr::makeDirectory(imagesDir);

			outList.open(listpath);
			if (outList.good()) {
				for (int i = 0; i < cams.size(); ++i) {
					const sibr::InputCamera::Ptr cam = cams[i];
					const std::string imageName = cam->name().empty() ? sibr::intToString<8>(i) + ".jpg" : cam->name();
					outList << "visualize/" << imageName << " " << cam->w() << " " << cam->h() << std::endl;
					cv::Mat3b dummy(cam->h(), cam->w());
					cv::imwrite(imagesDir + imageName, dummy);
				}
				outList.close();
			}
			else {
				SIBR_WRG << "Unable to export images list to path \"" << listpath << "\"." << std::endl;
			}
		}
	}

	void InputCamera::saveAsLookat(const std::vector<sibr::Camera>& cams, const std::string& fileName)
	{

		std::ofstream file(fileName, std::ios::out | std::ios::trunc);
		if (!file.is_open()) {
			SIBR_WRG << "Unable to save to file at path " << fileName << std::endl;
			return;
		}
		// Get the padding count.
		const int len = int(std::floor(std::log10(cams.size()))) + 1;
		for (size_t cid = 0; cid < cams.size(); ++cid) {
			const auto& cam = cams[cid];
			std::string id = std::to_string(cid);
			const std::string pad = std::string(len - id.size(), '0');

			const sibr::Vector3f& pos = cam.position();
			const sibr::Vector3f& up = cam.up();
			const sibr::Vector3f tgt = cam.position() + cam.dir();


			file << "Cam" << pad << id;
			file << " -D origin=" << pos[0] << "," << pos[1] << "," << pos[2];
			file << " -D target=" << tgt[0] << "," << tgt[1] << "," << tgt[2];
			file << " -D up=" << up[0] << "," << up[1] << "," << up[2];
			file << " -D fovy=" << cam.fovy();
			file << " -D clip=" << cam.znear() << "," << cam.zfar();
			file << "\n";
		}

		file.close();
	}

	std::vector<InputCamera::Ptr> InputCamera::loadColmapBin(const std::string& colmapSparsePath, const float zNear, const float zFar, const int fovXfovYFlag)
	{
		const std::string camerasListing = colmapSparsePath + "/cameras.bin";
		const std::string imagesListing = colmapSparsePath + "/images.bin";


  		std::ifstream camerasFile(camerasListing, std::ios::binary);
		std::ifstream imagesFile(imagesListing, std::ios::binary);

		if (!camerasFile.is_open()) {
			SIBR_ERR << "Unable to load camera colmap file" << camerasListing << std::endl;
		}
		if (!imagesFile.is_open()) {
			SIBR_WRG << "Unable to load images colmap file" << imagesListing << std::endl;
		}

		std::vector<InputCamera::Ptr> cameras;

		std::string line;

		struct CameraParametersColmap {
			size_t id;
			size_t width;
			size_t height;
			float  fx;
			float  fy;
			float  dx;
			float  dy;
		};

		std::map<size_t, CameraParametersColmap> cameraParameters;
  		const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&camerasFile);

  		for (size_t i = 0; i < num_cameras ; ++i) {

			CameraParametersColmap params;

			params.id = ReadBinaryLittleEndian<uint32_t>(&camerasFile);
			int model_id = ReadBinaryLittleEndian<int>(&camerasFile);
			params.width = ReadBinaryLittleEndian<uint64_t>(&camerasFile);
			params.height = ReadBinaryLittleEndian<uint64_t>(&camerasFile);
			std::vector<double> Params(4);

    			ReadBinaryLittleEndian<double>(&camerasFile, &Params);
			params.fx = float(Params[0]);
			params.fy = float(Params[1]);
			params.dx = float(Params[2]);
			params.dy = float(Params[3]);
			cameraParameters[params.id] = params;
		}

		// Now load the individual images and their extrinsic parameters
		sibr::Matrix3f converter;
		converter << 1, 0, 0,
			0, -1, 0,
			0, 0, -1;

  		const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&imagesFile);
		for (size_t i = 0; i < num_reg_images; ++i) {

			uint	    cId = ReadBinaryLittleEndian<image_t>(&imagesFile);
			float       qw = float(ReadBinaryLittleEndian<double>(&imagesFile));
			float       qx = float(ReadBinaryLittleEndian<double>(&imagesFile)) ;
			float       qy = float(ReadBinaryLittleEndian<double>(&imagesFile)) ;
			float       qz = float(ReadBinaryLittleEndian<double>(&imagesFile)) ;
			float       tx = float(ReadBinaryLittleEndian<double>(&imagesFile));
			float       ty = float(ReadBinaryLittleEndian<double>(&imagesFile));
			float       tz = float(ReadBinaryLittleEndian<double>(&imagesFile));
			size_t      id = ReadBinaryLittleEndian<camera_t>(&imagesFile) ;


			if (cameraParameters.find(id) == cameraParameters.end())
			{
				/* code multi camera broken
				SIBR_ERR << "Could not find intrinsics for image: "
					<< id << std::endl;
			*/
				id = 1;
			}
			const CameraParametersColmap& camParams = cameraParameters[id];


			const sibr::Quaternionf quat(qw, qx, qy, qz);
			const sibr::Matrix3f orientation = quat.toRotationMatrix().transpose() * converter;
			sibr::Vector3f translation(tx, ty, tz);

			sibr::Vector3f position = -(orientation * converter * translation);

			sibr::InputCamera::Ptr camera;
			if (fovXfovYFlag) {
				camera = std::make_shared<InputCamera>(InputCamera(camParams.fy, camParams.fx, 0.0f, 0.0f, int(camParams.width), int(camParams.height), int(cId)));
			}
			else {
				camera = std::make_shared<InputCamera>(InputCamera(camParams.fy, 0.0f, 0.0f, int(camParams.width), int(camParams.height), int(cId)));
			}
			std::string image_name;
			char name_char;
			do {
				imagesFile.read(&name_char, 1);
				if (name_char != '\0') {
					image_name += name_char;
				}
			} while (name_char != '\0');

			camera->name(image_name);
			camera->position(position);
			camera->rotation(sibr::Quaternionf(orientation));
			camera->znear(zNear);
			camera->zfar(zFar);
			cameras.push_back(camera);


    		// ignore the 2d points
    		const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&imagesFile);

    			for (size_t j = 0; j < num_points2D; ++j) {
			      const double x = ReadBinaryLittleEndian<double>(&imagesFile);
			      const double y = ReadBinaryLittleEndian<double>(&imagesFile);
				  point3D_t id = ReadBinaryLittleEndian<point3D_t>(&imagesFile);
    			}
		}
		return cameras;
	}

	std::vector<InputCamera::Ptr> InputCamera::loadJSON(const std::string& jsonPath, const float zNear, const float zFar)
	{
		std::ifstream json_file(jsonPath, std::ios::in);

		if (!json_file)
		{
			std::cerr << "file loading failed: " << jsonPath << std::endl;
			return std::vector<InputCamera::Ptr>();
		}

		std::vector<InputCamera::Ptr> cameras;

		picojson::value v;
		picojson::set_last_error(std::string());
		std::string err = picojson::parse(v, json_file);
		if (!err.empty()) {
			picojson::set_last_error(err);
			json_file.setstate(std::ios::failbit);
		}

		picojson::array& frames = v.get<picojson::array>();

		for (size_t i = 0; i < frames.size(); ++i)
		{
			int id = frames[i].get("id").get<double>();
			std::string imgname = frames[i].get("img_name").get<std::string>();
			int width = frames[i].get("width").get<double>();
			int height = frames[i].get("height").get<double>();
			float fy = frames[i].get("fy").get<double>();
			float fx = frames[i].get("fx").get<double>();

			sibr::InputCamera::Ptr camera = std::make_shared<InputCamera>(InputCamera(fy, fx, 0.0f, 0.0f, width, height, id));

			picojson::array& pos = frames[i].get("position").get<picojson::array>();
			sibr::Vector3f position(pos[0].get<double>(), pos[1].get<double>(), pos[2].get<double>());

			//position.x() = 0;
			//position.y() = 0;
			//position.z() = 1;

			picojson::array& rot = frames[i].get("rotation").get<picojson::array>();
			sibr::Matrix3f orientation;
			for (int i = 0; i < 3; i++)
			{
				picojson::array& row = rot[i].get<picojson::array>();
				for (int j = 0; j < 3; j++)
				{
					orientation(i, j) = row[j].get<double>();
				}
			}
			orientation.col(1) = -orientation.col(1);
			orientation.col(2) = -orientation.col(2);
			//orientation = sibr::Matrix3f::Identity();

			camera->name(imgname);
			camera->position(position);
			camera->rotation(sibr::Quaternionf(orientation));
			camera->znear(zNear);
			camera->zfar(zFar);
			cameras.push_back(camera);
		}
		return cameras;
	}

	std::vector<InputCamera::Ptr> InputCamera::loadTransform(const std::string& transformPath, int w, int h, std::string extension, const float zNear, const float zFar, const int offset, const int fovXfovYFlag)
	{
		std::ifstream json_file(transformPath, std::ios::in);

		if (!json_file)
		{
			std::cerr << "file loading failed: " << transformPath << std::endl;
			return std::vector<InputCamera::Ptr>();
		}

		std::vector<InputCamera::Ptr> cameras;

		picojson::value v;
		picojson::set_last_error(std::string());
		std::string err = picojson::parse(v, json_file);
		if (!err.empty()) {
			picojson::set_last_error(err);
			json_file.setstate(std::ios::failbit);
		}

		float fovx = v.get("camera_angle_x").get<double>();
		picojson::array& frames = v.get("frames").get<picojson::array>();

		for (int i = 0; i < frames.size(); i++)
		{
			std::string imgname = frames[i].get("file_path").get<std::string>() + "." + extension;

			auto mat = frames[i].get("transform_matrix").get<picojson::array>();

			Eigen::Matrix4f matrix;
			for (int i = 0; i < 4; i++)
			{
				auto row = mat[i].get<picojson::array>();
				for (int j = 0; j < 4; j++)
				{
					matrix(i, j) = row[j].get<double>();
				}
			}

			Eigen::Matrix3f R = matrix.block<3, 3>(0, 0);
			Eigen::Vector3f T(matrix(0, 3), matrix(1, 3), matrix(2, 3));

			float focalx = 0.5f * w / tan(fovx / 2.0f);
			float focaly = (((float)h)/w) * focalx;

			sibr::InputCamera::Ptr camera;
			if (fovXfovYFlag) {
				camera = std::make_shared<InputCamera>(InputCamera(focaly, focalx, 0.0f, 0.0f, int(w), int(h), i + offset));
			}
			else {
				camera = std::make_shared<InputCamera>(InputCamera(focalx, 0.0f, 0.0f, int(w), int(h), i + offset));
			}

			camera->name(imgname);
			camera->position(T);
			camera->rotation(sibr::Quaternionf(R));
			camera->znear(zNear);
			camera->zfar(zFar);
			cameras.push_back(camera);
		}
		return cameras;
	}


} 
