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


#include <fstream>
#include "core/assets/CameraRecorder.hpp"
#include "core/assets/InputCamera.hpp"
#include <opencv2/imgcodecs.hpp>

namespace sibr
{
	void	CameraRecorder::use(Camera& cam)
	{
		if (_recording) {
			_cameras.push_back(cam);
		} 
		else if (_playing && _pos < _cameras.size() ) {
			const float znear = cam.znear();
			const float zfar = cam.zfar();

			if( !_playNoInterp ) {
				//std::cout << _playing << std::endl;
				// If we reach the last frame of the interpolation b/w two cameras, skip to next camera.
				if (_interp > (1.0f - _speed))
				{
					_interp = 0.0f;
					_pos++;
				}
				// Interpolate between the two closest cameras.
				const float k = std::min(std::max(_interp, 0.0f), 1.0f);
				
				sibr::Camera & camStart = _cameras[std::min(int(_pos), int(_cameras.size()) - 1)];
				sibr::Camera & camNext = _cameras[std::min(int(_pos) + 1, int(_cameras.size())-1)];

				cam = sibr::Camera::interpolate(camStart, camNext, k);

				_interp += _speed;
			}
			else {
				cam = _cameras[_pos];
				_pos++;
				if (_pos == _cameras.size())
					_playNoInterp = false;
				
			}

			// Preserve the znear and zfar.
			cam.znear(znear);
			cam.zfar(zfar);

			if (_saving) {
				std::ostringstream ssZeroPad;
				ssZeroPad << std::setw(8) << std::setfill('0') << (_pos);
				cam.setSavePath(_savingPath + "/" + ssZeroPad.str() + ".png");
				//std::cout << "Saving frame as: " << cam.savePath() << std::endl;
			}
			if (_savingVideo) {
				cam.setDebugVideo(true);
			}
			if (_pos >= _cameras.size())
			{
				stop();
				SIBR_LOG << "[CameraRecorder] - Playback Finished" << std::endl;
			}
		} 
		else {
			//std::cout << _playing << std::endl;
			cam.setSavePath("");
			cam.setDebugVideo(false);
		}
	}

	void	CameraRecorder::playback(void)
	{
		stop();
		_playing = true;
		SIBR_LOG << "[CameraRecorder] - Playing" << std::endl;
	}

	void	CameraRecorder::record(void)
	{
		stop();
		_recording = true;
		SIBR_LOG << "[CameraRecorder] - Recording" << std::endl;
	}

	void sibr::CameraRecorder::saving(std::string savePath)
	{
		_saving = true;
		_savingPath = savePath;
		SIBR_LOG << "[CameraRecorder] - Recording" << std::endl;
	}

	void CameraRecorder::savingVideo(bool saveVideo)
	{
		_savingVideo = saveVideo;
	}

	void sibr::CameraRecorder::stopSaving(void)
	{
		_saving = false;
		_savingPath = "";
	}

	void	CameraRecorder::stop(void)
	{
		_recording = _playing = false;
		_pos = 0;
		_interp = 0.0f;
	}

	void	CameraRecorder::reset(void)
	{
		stop();
		_cameras.clear();
	}

	bool	CameraRecorder::load(const std::string& filename)
	{
		reset();

		sibr::ByteStream stream;

		if (stream.load(filename) == false)
			return false;

		int32 num = 0;

		std::cout << " CameraRecorder::load " << num << std::endl;

		stream >> num;
		while (num > 0)
		{
			Camera cam;
			stream >> cam;
			_cameras.emplace_back(std::move(cam));
			--num;
		}
		SIBR_LOG << "[CameraRecorder] - Loaded from " << filename << std::endl;
		return stream;
	}

	void	CameraRecorder::save(const std::string& filename)
	{
		sibr::ByteStream stream;
		int32 num = (int32)_cameras.size();
		stream << num;
		for (const Camera& cam : _cameras)
			stream << cam;

		stream.saveToFile(filename);
		SIBR_LOG << "[CameraRecorder] - Saved " << num << " cameras to " << filename << std::endl;
	}

	bool CameraRecorder::safeLoad(const std::string & filename, int w, int h)
	{
		Path path = Path(filename);

		if (path.extension().string() == ".out") {
			loadBundle(filename, w, h);
			return true;
		} else if (path.extension().string() == ".path") {
			return load(filename);
		} 
		SIBR_WRG << "Unable to load camera path" << std::endl;
		return false;
	}

	void CameraRecorder::loadBundle(const std::string & filePath, int w, int h)
	{
		const std::string bundlerFile = filePath;
		SIBR_LOG << "Loading bundle path." << std::endl;

		// check bundler file
		std::ifstream bundle_file(bundlerFile);
		SIBR_ASSERT(bundle_file.is_open());

		// read number of images
		std::string line;
		getline(bundle_file, line);	// ignore first line - contains version
		int numImages = 0;
		bundle_file >> numImages;	// read first value (number of images)
		getline(bundle_file, line);	// ignore the rest of the line

		std::vector<InputCamera::Ptr> cameras(numImages);
		//  Parse bundle.out file for camera calibration parameters
		for (int i = 0; i < numImages; i++) {
		
			Matrix4f m;
			bundle_file >> m(0) >> m(1) >> m(2) >> m(3) >> m(4);
			bundle_file >> m(5) >> m(6) >> m(7) >> m(8) >> m(9);
			bundle_file >> m(10) >> m(11) >> m(12) >> m(13) >> m(14);

			cameras[i] = InputCamera::Ptr(new InputCamera(i, w, h, m, true));

			cameras[i]->znear(0.2f); cameras[i]->zfar(250.f);

		}

		for (const InputCamera::Ptr cam : cameras)
		{
			_cameras.push_back(*cam);
		}

	}

	void CameraRecorder::loadColmap(const std::string &filePath, int w, int h)
	{
		SIBR_LOG << "Loading colmap path." << std::endl;
		boost::filesystem::path colmapDir ( filePath );

		SIBR_LOG << "DEBUG: colmap path dir " << colmapDir.parent_path().string() << std::endl;

		std::vector<InputCamera::Ptr> path = InputCamera::loadColmap(colmapDir.parent_path().string(), float(0.01), 1000, false );
		for (const InputCamera::Ptr cam : path)
		{
			_cameras.push_back(*cam);
		}
	}

	void CameraRecorder::loadLookat(const std::string &filePath, int w, int h)
	{
		SIBR_LOG << "Loading lookat path." << std::endl;
		std::vector<InputCamera::Ptr> path = InputCamera::loadLookat(filePath, std::vector<Vector2u>{Vector2u(w, h)});
		for (const InputCamera::Ptr cam : path)
		{
			_cameras.push_back(*cam);
		}
	}

	void CameraRecorder::saveAsBundle(const std::string & filePath, const int height, const int step)
	{

		std::ofstream out(filePath.c_str(), std::ios::binary);
		if (!out.good()) {
			SIBR_LOG << "ERROR: cannot write to the file '" << filePath << "'" << std::endl;
			return;
		}

		if(_cameras.empty()) {
			return;
		}

		const int size = static_cast<int>(_cameras.size() / step);
		
		out << "# Bundle file v0.3\n";
		out << size << " " << 0 << "\n";

		for (int i = 0; i < _cameras.size(); i += step) {

			const sibr::Camera cam = _cameras[i];
			sibr::Quaternionf q = cam.rotation();
			sibr::Matrix3f m1 = q.toRotationMatrix();
			sibr::Vector3f pos = -m1.transpose()  * cam.position();

			float m[15];
			m[0] = 0.5f*height / tan(cam.fovy() / 2.f); m[1] = 0.0f; m[2] = 0.0f;
			m[3] = m1(0, 0), m[4] = m1(1, 0), m[5] = m1(2, 0);
			m[6] = m1(0, 1), m[7] = m1(1, 1), m[8] = m1(2, 1);
			m[9] = m1(0, 2), m[10] = m1(1, 2), m[11] = m1(2, 2);
			m[12] = pos.x(), m[13] = pos.y(), m[14] = pos.z();

			out << m[0] << " " << m[1] << " " << m[2] << std::endl;
			out << m[3] << " " << m[4] << " " << m[5] << std::endl;
			out << m[6] << " " << m[7] << " " << m[8] << std::endl;
			out << m[9] << " " << m[10] << " " << m[11] << std::endl;
			out << m[12] << " " << m[13] << " " << m[14] << std::endl;
		}
		out << std::endl;
		out.close();

		SIBR_LOG << "[CameraRecorder] - Saved " << _cameras.size() << " cameras to " << filePath << " (using fovy " <<_cameras[0].fovy() << ")." << std::endl;
		
	}

	void CameraRecorder::saveAsColmap(const std::string& filePath, const int height, const int width)
	{

		std::string basepath = parentDirectory(filePath);
		std::cout << basepath << std::endl;
		std::string images_filepath = basepath + "/images.txt";
		std::string cameras_filepath = basepath + "/cameras.txt";

		std::ofstream out_images(images_filepath.c_str(), std::ios::binary);
		std::ofstream out_cameras(cameras_filepath.c_str(), std::ios::binary);

		if (!out_images.good()) {
			SIBR_LOG << "ERROR: cannot write to the file '" << filePath << "'" << std::endl;
			return;
		}
		if (!out_cameras.good()) {
			SIBR_LOG << "ERROR: cannot write to the file '" << filePath << "'" << std::endl;
			return;
		}

		if (_cameras.empty()) {
			return;
		}

		const int size = static_cast<int>(_cameras.size());
		
		sibr::Matrix3f converter;
		converter << 1, 0, 0,
			0, -1, 0,
			0, 0, -1;



		out_images << "# Image list with two lines of data per image:" << std::endl;
		out_images << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME" << std::endl;
		out_images << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
		for (int i = 0; i < _cameras.size(); i++) {
			sibr::Matrix3f tmp = _cameras[i].rotation().toRotationMatrix() * converter;
			sibr::Matrix3f Qinv = tmp.transpose();
			sibr::Quaternionf q = quatFromMatrix(Qinv);
			sibr::Vector3f t = converter * _cameras[i].rotation().toRotationMatrix().transpose() * (-_cameras[i].position());
		

			out_images << i << " " << -_cameras[i].rotation().x() << " " << -_cameras[i].rotation().w() << " " << -_cameras[i].rotation().z() << " " << _cameras[i].rotation().y() << " " <<
				_cameras[i].view()(0, 3) << " " << -_cameras[i].view()(1, 3) << " " << -_cameras[i].view()(2, 3) << " " << i << " " << "00000000.png" << std::endl << std::endl;
			
			float focal = 0.5f * height / tan(_cameras[i].fovy() / 2.f);
			//float focal = 1.0f / (tan(0.5f * cam.fovy()) * 2.0f / float(height));
			out_cameras << i << " " << "PINHOLE" << " " << width << " " << height << " " << focal << " " << focal << " " << width / 2.0 << " " << height / 2.0 << std::endl;
		}

		out_images << std::endl;
		out_images.close();
		out_cameras << std::endl;
		out_cameras.close();
		SIBR_LOG << "[CameraRecorder] - Saved " << _cameras.size() << " cameras to " << filePath << " (using fovy " << _cameras[0].fovy() << ")." << std::endl;

	}


	void CameraRecorder::saveAsFRIBRBundle(const std::string & dirPath, const int width, const int height)
	{
		const std::string bundlepath = dirPath + "/path.rd.out";
		const std::string listpath = dirPath + "/list.txt";
		const std::string imagesDir = dirPath + "/visualize/";
		sibr::makeDirectory(dirPath);
		sibr::makeDirectory(imagesDir);
		std::ofstream out(bundlepath);
		std::ofstream outList(listpath);
		if (out.good() && outList.good()) {
			const int size = static_cast<int>(_cameras.size() / 1);
			out << "# Bundle file v0.3\n";
			out << size << " " << 0 << "\n";
			sibr::Matrix3f converter;
			converter << 1, 0, 0,
				0, -1, 0,
				0, 0, -1;
			sibr::Matrix3f from_cv;
			from_cv << 1, 0, 0,
				0, -1, 0,
				0, 0, -1;
			for (int i = 0; i < _cameras.size(); ++i) {

				const sibr::Camera cam = _cameras[i];

				sibr::Matrix3f orientation = cam.rotation().toRotationMatrix();
				sibr::Vector3f position = cam.position();
				sibr::Matrix3f rotation_cv = converter.transpose() * orientation.transpose() * converter;
				sibr::Matrix3f rotation_bundler = from_cv * rotation_cv;
				sibr::Vector3f position_cv = converter.transpose() * position;
				sibr::Vector3f translation_cv = -(rotation_cv * position_cv);
				sibr::Vector3f pos = from_cv * translation_cv;

				sibr::Matrix3f m1 = rotation_bundler.transpose();
				float m[15];
				m[0] = 0.5f*height / tan(cam.fovy() / 2.f); m[1] = 0.0f; m[2] = 0.0f;
				m[3] = m1(0, 0), m[4] = m1(1, 0), m[5] = m1(2, 0);
				m[6] = m1(0, 1), m[7] = m1(1, 1), m[8] = m1(2, 1);
				m[9] = m1(0, 2), m[10] = m1(1, 2), m[11] = m1(2, 2);
				m[12] = pos.x(), m[13] = pos.y(), m[14] = pos.z();
				out << m[0] << " " << m[1] << " " << m[2] << std::endl;
				out << m[3] << " " << m[4] << " " << m[5] << std::endl;
				out << m[6] << " " << m[7] << " " << m[8] << std::endl;
				out << m[9] << " " << m[10] << " " << m[11] << std::endl;
				out << m[12] << " " << m[13] << " " << m[14] << std::endl;

				const std::string imageName = sibr::intToString<8>(i) + ".jpg";
				outList << "visualize/" + imageName << " 0 " << m[0] << std::endl;

				cv::Mat3b dummy(height, width);
				cv::imwrite(imagesDir + imageName, dummy);
			}
			out << std::endl;
			out.close();
			outList.close();

			SIBR_LOG << "[CameraRecorder] - Saved " << _cameras.size() << " cameras to " << dirPath << "." << std::endl;
		}
	}

	void CameraRecorder::saveAsLookAt(const std::string & filePath) const
	{
		InputCamera::saveAsLookat(_cameras, filePath);
	}

	// offline path rendering
	bool CameraRecorder::loadPath(const std::string& pathFileName, int w, int h) {
		_savingPath = parentDirectory(pathFileName);
		if (!fileExists(pathFileName)) {
			SIBR_WRG << "Camera path " << pathFileName << " doesnt exist. " << std::endl;
			return false;
		}
		_ow = w, _oh = h;
		if (boost::filesystem::extension(pathFileName) == ".out")
			loadBundle(pathFileName, w, h);
		else if (boost::filesystem::extension(pathFileName) == ".lookat")
			loadLookat(pathFileName, w, h);
		else if (boost::filesystem::extension(pathFileName) == ".txt")
			loadColmap(pathFileName, w, h);
		else
			load(pathFileName);
		return true;
	}

	void CameraRecorder::recordOfflinePath(const std::string& outPathDir, ViewBase::Ptr view, const std::string& prefix) {
		sibr::ImageRGBA32F::Ptr outImage;
		outImage.reset(new ImageRGBA32F(_ow, _oh));
		std::string outpathd = outPathDir;

		sibr::RenderTargetRGBA32F::Ptr outFrame;
		outFrame.reset(new RenderTargetRGBA32F(_ow, _oh));
		std::string outFileName;

		boost::filesystem::path dstFolder;

		outpathd = outPathDir;

		if (outPathDir == "pathOutput" && _savingPath != "") { // default to path parent, saved by loadPath
			outpathd = _savingPath + "/" + "pathOutput";
			dstFolder = outpathd;
			if (!directoryExists(outpathd) && !boost::filesystem::create_directories(dstFolder))
				SIBR_ERR << "Error creating directory " << dstFolder << std::endl;
		}

        if( prefix != "" ) 
    		dstFolder = outpathd = outpathd + "/" + prefix;
        else 
            dstFolder = outpathd ;

		if (!directoryExists(outpathd) && !boost::filesystem::create_directories(dstFolder))
			SIBR_ERR << "Error creating directory " << dstFolder << std::endl;

		std::cout << "Rendering path with " << _cameras.size() << " cameras to " << outpathd << std::endl;

		for (int i = 0; i < _cameras.size(); ++i) {
			outFrame->clear();
			std::ostringstream ssZeroPad;
			ssZeroPad << std::setw(8) << std::setfill('0') << i;
			outFileName = outpathd + "/" +  ssZeroPad.str() + ".png";
			std::cout << outFileName << " " << std::endl;
			view->onRenderIBR(*outFrame, _cameras[i]);
			outFrame->readBack(*outImage);
			outImage->save(outFileName, false);
		}
		std::cout << std::endl;

		std::cout << "Done rendering path. " << std::endl;

	}

	void CameraRecorder::saveImage(const std::string& outPathDir, const Camera& cam, int w, int h) {
		sibr::ImageRGBA32F::Ptr outImage;
		_ow = w, _oh = h;
		outImage.reset(new ImageRGBA32F(_ow, _oh));
		std::string outpathd = outPathDir;

		sibr::RenderTargetRGBA32F::Ptr outFrame;
		outFrame.reset(new RenderTargetRGBA32F(_ow, _oh));
		std::string outFileName;

		boost::filesystem::path dstFolder;

		outpathd = outPathDir;

		if (outPathDir == "") { // default to path parent, saved by loadPath
			outpathd = _dsPath + "/" + "pathOutput";
			dstFolder = outpathd;
			if (!directoryExists(outpathd) && !boost::filesystem::create_directories(dstFolder))
				SIBR_ERR << "Error creating directory " << dstFolder << std::endl;
		}

		dstFolder = outpathd;

		if (!directoryExists(outpathd) && !boost::filesystem::create_directories(dstFolder))
			SIBR_ERR << "Error creating directory " << dstFolder << std::endl;

		std::cout << "Saving current camera to " << outpathd << std::endl;

		outFrame->clear();
		std::ostringstream ssZeroPad;
		static int i = 0;
		ssZeroPad << std::setw(8) << std::setfill('0') << i++;
		outFileName = outpathd + "/" + ssZeroPad.str() + ".png";
		std::cout << outFileName << " " << std::endl;
		_view->onRenderIBR(*outFrame, cam);
		outFrame->readBack(*outImage);
		outImage->save(outFileName, false);
		std::cout << std::endl;
		std::cout << "Done saving image. " << std::endl;
	}

} // namespace sibr
