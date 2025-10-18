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
#include <memory>
#include <map>
#include <queue>

#include <assimp/Importer.hpp> // C++ importer interface
#include <assimp/scene.h> // Output data structure
#include <assimp/postprocess.h> // Post processing flags
#include <assimp/cexport.h>
#include <assimp/Exporter.hpp>

#include "core/system/ByteStream.hpp"
#include "core/graphics/Mesh.hpp"

#include "boost/filesystem.hpp"
#include "core/system/XMLTree.h"
#include "core/system/Matrix.hpp"
#include <set>
#include <boost/variant/detail/substitute.hpp>
#include "core/assets/colmapheader.h"

namespace sibr
{
	typedef uint32_t image_t;
	typedef uint32_t camera_t;
	typedef uint64_t point3D_t;
	typedef uint32_t point2D_t;

	void ReadPoints3DBinary(const std::string& path, Mesh::Vertices& verts, Mesh::Colors& cols, int& numverts) {
	  std::ifstream file(path, std::ios::binary);
	//  CHECK(file.is_open()) << path;

	  const size_t num_points3D = ReadBinaryLittleEndian<uint64_t>(&file);
	  numverts = num_points3D;
	  std::cerr << "Num 3D pts " << num_points3D << std::endl;
	  for (size_t i = 0; i < num_points3D; ++i) {
	    //class Point3D point3D;

	    const uint64_t point3D_id = ReadBinaryLittleEndian<uint64_t>(&file);
	  //  num_added_points3D_ = std::max(num_added_points3D_, point3D_id);

	//    point3D.XYZ()(0) = ReadBinaryLittleEndian<double>(&file);
	//    point3D.XYZ()(1) = ReadBinaryLittleEndian<double>(&file);
	//    point3D.XYZ()(2) = ReadBinaryLittleEndian<double>(&file);
	//    point3D.Color(0) = ReadBinaryLittleEndian<uint8_t>(&file);
	//    point3D.Color(1) = ReadBinaryLittleEndian<uint8_t>(&file);
	//    point3D.Color(2) = ReadBinaryLittleEndian<uint8_t>(&file);
	//    point3D.SetError(ReadBinaryLittleEndian<double>(&file));

		double x = ReadBinaryLittleEndian<double>(&file);
		double y = ReadBinaryLittleEndian<double>(&file);
		double z = ReadBinaryLittleEndian<double>(&file);
		Vector3f vert(x,y,z);

		verts.push_back(vert);

		float r = float(ReadBinaryLittleEndian<uint8_t>(&file))/255.f;
		float g = float(ReadBinaryLittleEndian<uint8_t>(&file))/255.f;
		float b = float(ReadBinaryLittleEndian<uint8_t>(&file))/255.f;

		Vector3f c(r, g, b);

		cols.push_back(c);
		double err =  ReadBinaryLittleEndian<double>(&file);

	    const size_t track_length = ReadBinaryLittleEndian<uint64_t>(&file);
	    //std::cerr << "Track length " << track_length << std::endl;
		// read and include
	    for (size_t j = 0; j < track_length; ++j) {
	      const image_t image_id = ReadBinaryLittleEndian<image_t>(&file);
	      const point2D_t point2D_idx = ReadBinaryLittleEndian<point2D_t>(&file);
	      //point3D.Track().AddElement(image_id, point2D_idx);
	    }
	    //point3D.Track().Compress();

	    //points3D_.emplace(point3D_id, point3D);
	  }
	}

	void ReadPoints3DText(const std::string& path, Mesh::Vertices& verts, Mesh::Vertices& cols) {
	//  points3D_.clear();
	  std::ifstream file(path);
	//  CHECK(file.is_open()) << path;
	  std::string line;
	  std::string item;
	  while (std::getline(file, line)) {
	    StringTrim(&line);
	    if (line.empty() || line[0] == '#') {
	      continue;
	    }
	    std::stringstream line_stream(line);
	    // ID
	    std::getline(line_stream, item, ' ');
	    const point3D_t point3D_id = std::stoll(item);

	    // Make sure, that we can add new 3D points after reading 3D points
	    // without overwriting existing 3D points.
	    // num_added_points3D_ = std::max(num_added_points3D_, point3D_id);

	    // XYZ
	    std::getline(line_stream, item, ' ');
	    std::cerr << "point3D.XYZ(0) = " << std::stold(item) << std::endl;

	    std::getline(line_stream, item, ' ');
	    std::cerr << "point3D.XYZ(1) = " << std::stold(item) << std::endl;

	    std::getline(line_stream, item, ' ');
	    std::cerr << "point3D.XYZ(2) = " << std::stold(item) << std::endl;

	    // Color
	    std::getline(line_stream, item, ' ');
	    std::cerr << "point3D.Color(0) = " << static_cast<uint8_t>(std::stoi(item)) << std::endl;

	    std::getline(line_stream, item, ' ');
	    std::cerr << "point3D.Color(1) = " << static_cast<uint8_t>(std::stoi(item)) << std::endl;

	    std::getline(line_stream, item, ' ');
	    std::cerr << "point3D.Color(2) = " << static_cast<uint8_t>(std::stoi(item)) << std::endl;

	    // ERROR
	    std::getline(line_stream, item, ' ');
	    std::cerr << "point3D.SetError(" << std::stold(item) << std::endl;

	    // TRACK
	    while (!line_stream.eof()) {
	    //  TrackElement track_el;

	      std::getline(line_stream, item, ' ');
	      StringTrim(&item);
	      if (item.empty()) {
		break;
	      }
	      std::cerr << "track_el.image_id = " << std::stoul(item) << std::endl;

	      std::getline(line_stream, item, ' ');
	      std::cerr << "track_el.point2D_idx = " << std::stoul(item) << std::endl;
	//      point3D.Track().AddElement(track_el);
	    }
	 //   point3D.Track().Compress();
	  //  points3D_.emplace(point3D_id, point3D);
	  }
	}

	Mesh::Mesh(bool withGraphics) : _meshPath("") {
		if (withGraphics) {
			_gl.bufferGL.reset(new MeshBufferGL);
		}
		else {
			_gl.bufferGL = nullptr;
		}
	}

	bool		Mesh::saveToObj(const std::string& filename)  const
	{
		aiScene scene;
		scene.mRootNode = new aiNode();

		scene.mMaterials = new aiMaterial * [1];
		scene.mMaterials[0] = nullptr;
		scene.mNumMaterials = 1;

		scene.mMaterials[0] = new aiMaterial();

		scene.mMeshes = new aiMesh * [1];
		scene.mNumMeshes = 1;

		scene.mMeshes[0] = new aiMesh();
		scene.mMeshes[0]->mMaterialIndex = 0;

		scene.mRootNode->mMeshes = new unsigned int[1];
		scene.mRootNode->mMeshes[0] = 0;
		scene.mRootNode->mNumMeshes = 1;

		auto pMesh = scene.mMeshes[0];

		const auto& vVertices = _vertices;

		pMesh->mVertices = new aiVector3D[vVertices.size()];
		pMesh->mNumVertices = static_cast<unsigned int>(vVertices.size());

		if (hasNormals()) {
			pMesh->mNormals = new aiVector3D[vVertices.size()];
		}
		else {
			pMesh->mNormals = nullptr;
		}

		if (hasTexCoords()) {
			pMesh->mTextureCoords[0] = new aiVector3D[vVertices.size()];
			pMesh->mNumUVComponents[0] = 2;
		}
		else {
			pMesh->mTextureCoords[0] = nullptr;
			pMesh->mNumUVComponents[0] = 0;
		}

		int j = 0;
		for (auto itr = vVertices.begin(); itr != vVertices.end(); ++itr)
		{
			pMesh->mVertices[itr - vVertices.begin()] = aiVector3D(vVertices[j].x(), vVertices[j].y(), vVertices[j].z());
			if (hasNormals())
				pMesh->mNormals[itr - vVertices.begin()] = aiVector3D(_normals[j].x(), _normals[j].y(), _normals[j].z());
			if (hasTexCoords())
				pMesh->mTextureCoords[0][itr - vVertices.begin()] = aiVector3D(_texcoords[j][0], _texcoords[j][1], 0);
			j++;
		}

		pMesh->mFaces = new aiFace[_triangles.size()];
		pMesh->mNumFaces = (unsigned int)(_triangles.size());

		for (uint i = 0; i < _triangles.size(); ++i)
		{
			const Vector3u& tri = _triangles[i];
			aiFace& face = pMesh->mFaces[i];
			face.mIndices = new unsigned int[3];
			face.mNumIndices = 3;

			face.mIndices[0] = tri[0];
			face.mIndices[1] = tri[1];
			face.mIndices[2] = tri[2];
		}
		Assimp::Exporter mAiExporter;
		const aiScene* s = (const aiScene*)&(scene);

		SIBR_LOG << "Saving (via ASSIMP) " << filename << "'..." << std::endl;
		mAiExporter.Export(s, "obj", filename);

		// mesh and scene destructors free memory


		return true;
	}


	bool		Mesh::saveToBinaryPLY(const std::string& filename, bool universal, const std::string& textureName)  const
	{
		assert(_vertices.size());

		SIBR_LOG << "Saving '" << filename << "'..." << std::endl;

		// Note that Assimp supports also export for some formats. However,
		// when I tried with the current Assimp version (3.0) it failed to
		// do a good export for .ply (using binary version).
		// In addition, at this time there is no control/ExportProperties.
		// Thus I just do it myself.

		std::ofstream	file(filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

		if (file)
		{
			file << "ply" << std::endl;
			file << "format binary_big_endian 1.0" << std::endl;
			file << "comment Created by SIBR project" << std::endl;
			if (hasTexCoords())
			{
				file << "comment TextureFile " << textureName << std::endl;
			}
			file << "element vertex " << _vertices.size() << std::endl;
			file << "property float x" << std::endl;
			file << "property float y" << std::endl;
			file << "property float z" << std::endl;
			if (hasColors())
			{
				if (universal)
				{
					file << "property uchar red" << std::endl;
					file << "property uchar green" << std::endl;
					file << "property uchar blue" << std::endl;
				}
				else
				{
					file << "property ushort red" << std::endl;
					file << "property ushort green" << std::endl;
					file << "property ushort blue" << std::endl;
				}

			}
			if (hasNormals())
			{
				file << "property float nx" << std::endl;
				file << "property float ny" << std::endl;
				file << "property float nz" << std::endl;
			}
			if (hasTexCoords())
			{
				file << "property float texture_u" << std::endl;
				file << "property float texture_v" << std::endl;
			}

			file << "element face " << _triangles.size() << std::endl;
			file << "property list uchar uint vertex_indices" << std::endl;
			file << "end_header" << std::endl;

			/// BINARY version /////
			ByteStream	bytes;

			for (uint i = 0; i < _vertices.size(); ++i)
			{
				const Vector3f& v = _vertices[i];

				bytes << float(v[0]) << float(v[1]) << float(v[2]);

				if (hasColors())
				{
					const Vector3f& c = _colors[i];
					if (universal)
						bytes
						<< uint8(c[0] * (UINT8_MAX - 1))
						<< uint8(c[1] * (UINT8_MAX - 1))
						<< uint8(c[2] * (UINT8_MAX - 1)); // ! converting colors explicitly
					else
						bytes
						<< uint16(c[0] * (UINT16_MAX - 1))
						<< uint16(c[1] * (UINT16_MAX - 1))
						<< uint16(c[2] * (UINT16_MAX - 1)); // ! converting colors explicitly
				}

				if (hasNormals())
				{
					const Vector3f& n = _normals[i];

					bytes << float(n[0]) << float(n[1]) << float(n[2]);
				}

				if (hasTexCoords())
				{
					const Vector2f& uv = _texcoords[i];

					bytes << float(uv[0]) << float(uv[1]);
				}
			}

			for (uint i = 0; i < _triangles.size(); ++i)
			{
				const Vector3u& tri = _triangles[i];

				bytes << uint8(3);
				for (uint j = 0; j < 3; ++j)
					bytes << uint32(tri[j]);
			}

			file.write(reinterpret_cast<const char*>(bytes.buffer()), bytes.bufferSize());
			file.close();
			SIBR_LOG << "Saving '" << filename << "'... done" << std::endl;
			return true;
		}
		SIBR_LOG << "error: cannot write to file '" << filename << "'." << std::endl;
		return false;

	}

	bool		Mesh::saveToASCIIPLY(const std::string& filename, bool universal, const std::string& textureName) const
	{
		assert(_vertices.size());

		// Note that Assimp supports also export for some formats. However,
		// when I tried with the current Assimp version (3.0) it failed to
		// do a good export for .ply (using binary version).
		// In addition, at this time there is no control/ExportProperties.
		// Thus I just do it myself.

		std::ofstream	file(filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

		if (file)
		{
			file << "ply" << std::endl;
			file << "format ascii 1.0" << std::endl;
			file << "comment Created by SIBR project" << std::endl;
			if (hasTexCoords())
			{
				file << "comment TextureFile " << textureName << std::endl;
			}
			file << "element vertex " << _vertices.size() << std::endl;
			file << "property float x" << std::endl;
			file << "property float y" << std::endl;
			file << "property float z" << std::endl;

			if (hasColors())
			{
				if (universal)
				{
					file << "property uchar red" << std::endl;
					file << "property uchar green" << std::endl;
					file << "property uchar blue" << std::endl;
				}
				else
				{
					file << "property ushort red" << std::endl;
					file << "property ushort green" << std::endl;
					file << "property ushort blue" << std::endl;
				}
			}

			if (hasNormals())
			{
				file << "property float nx" << std::endl;
				file << "property float ny" << std::endl;
				file << "property float nz" << std::endl;
			}

			if (hasTexCoords())
			{
				file << "property float texture_u" << std::endl;
				file << "property float texture_v" << std::endl;
			}

			file << "element face " << _triangles.size() << std::endl;
			file << "property list uchar uint vertex_indices" << std::endl;
			file << "end_header" << std::endl;

			/////// ASCII version /////

			for (uint i = 0; i < _vertices.size(); ++i)
			{
				const Vector3f& v = _vertices[i];

				file << v[0] << " " << v[1] << " " << v[2] << " ";

				if (hasColors())
				{
					const Vector3f& c = _colors[i];

					if (universal)
					{
						file << int(c[0] * (UINT8_MAX - 1)) << " "
							<< int(c[1] * (UINT8_MAX - 1)) << " "
							<< int(c[2] * (UINT8_MAX - 1)) << " ";
					}
					else
					{
						file << int(c[0] * (UINT16_MAX - 1)) << " "
							<< int(c[1] * (UINT16_MAX - 1)) << " "
							<< int(c[2] * (UINT16_MAX - 1)) << " ";
					}
				}

				if (hasNormals())
				{
					const Vector3f& n = _normals[i];

					file << n[0] << " " << n[1] << " " << n[2] << " ";
				}

				if (hasTexCoords())
				{
					const Vector2f& uv = _texcoords[i];

					file << uv[0] << " " << uv[1] << " ";
				}

				file << std::endl;
			}

			for (uint i = 0; i < _triangles.size(); ++i)
			{
				const Vector3u& tri = _triangles[i];

				file << 3;
				for (uint j = 0; j < 3; ++j)
					file << " " << tri[j];
				file << std::endl;
			}

			file.close();
			SIBR_LOG << "'" << filename << "' saved." << std::endl;
			return true;
		}
		SIBR_LOG << "error: cannot write to file '" << filename << "'." << std::endl;
		return false;

	}
	bool	Mesh::load(const std::string& filename, const std::string& dataset_path )
	{
		// Does the file exists?
		if (!sibr::fileExists(filename)) {
			SIBR_LOG << "Error: can't load mesh '" << filename << "." << std::endl;
			return false;
		}
		Assimp::Importer	importer;
		//importer.SetPropertyBool(AI_CONFIG_PP_FD_REMOVE, true); // cause Assimp to remove all degenerated faces as soon as they are detected
		const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_FindDegenerates);

		if (!scene)
		{
			SIBR_LOG << "error: can't load mesh '" << filename
				<< "' (" << importer.GetErrorString() << ")." << std::endl;
			return false;
		}

		// check for texture
		aiMaterial *material;
		if( scene->mNumMaterials > 0 ) {
			material = scene->mMaterials[0];
			aiString Path;
			if(material->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS ) {
				_textureImageFileName = Path.data;
				std::cerr << "Texture name " << _textureImageFileName << std::endl;
			}

		}

		if (scene->mNumMeshes == 0)
		{
			SIBR_LOG << "error: the loaded model file ('" << filename
				<< "') contains zero or more than one mesh. Number of meshes : " << scene->mNumMeshes << std::endl;
			return false;
		}

		auto convertVec = [](const aiVector3D& v) { return Vector3f(v.x, v.y, v.z); };
		_triangles.clear();

		uint offsetVertices = 0;
		uint offsetFaces = 0;
		uint matId = 0;
		std::map<std::string, int> matName2Id;
		Matrix3f converter;
		converter <<
			1, 0, 0,
			0, 1, 0,
			0, 0, 1;

		for (uint meshId = 0; meshId < scene->mNumMeshes; ++meshId) {
			const aiMesh* mesh = scene->mMeshes[meshId];

			_vertices.resize(offsetVertices + mesh->mNumVertices);
			for (uint i = 0; i < mesh->mNumVertices; ++i)
				_vertices[offsetVertices + i] = converter * convertVec(mesh->mVertices[i]);


			if (mesh->HasVertexColors(0) && mesh->mColors[0])
			{
				_colors.resize(offsetVertices + mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i)
				{
					_colors[offsetVertices + i] = Vector3f(
						mesh->mColors[0][i].r,
						mesh->mColors[0][i].g,
						mesh->mColors[0][i].b);
				}
			}

			if (mesh->HasNormals())
			{
				_normals.resize(offsetVertices + mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i) {
					_normals[offsetVertices + i] = converter * convertVec(mesh->mNormals[i]);
				}

			}

			if (mesh->HasTextureCoords(0))
			{
				_texcoords.resize(offsetVertices + mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i)
					_texcoords[offsetVertices + i] = convertVec(mesh->mTextureCoords[0][i]).xy();
				// TODO: make a clean function
				std::string texFileName = dataset_path + "/capreal/" + _textureImageFileName;
				if( !fileExists(texFileName))
					texFileName = parentDirectory(parentDirectory(dataset_path)) + "/capreal/" + _textureImageFileName;
				if( !fileExists(texFileName))
					texFileName = parentDirectory(dataset_path) + "/capreal/" + _textureImageFileName;

				if (!mesh->HasVertexColors(0) && fileExists(texFileName)) {
					// Sample the texture
					sibr::ImageRGB texImg;
					texImg.load(texFileName);
					std::cout << "Computing vertex colors ..";
					_colors.resize(offsetVertices + mesh->mNumVertices);
					for (uint ci = 0; ci < mesh->mNumVertices; ++ci)
					{
						Vector2f uv = _texcoords[offsetVertices + ci];
						Vector3ub col = texImg((uv[0]*texImg.w()), uint((1-uv[1])*texImg.h()));
						_colors[offsetVertices + ci] = Vector3f(float(col[0]) / 255.0, float(col[1]) / 255.0, float(col[2]) / 255.0);
					}
					SIBR_WRG << "Done." << std::endl;
				}
			}
			if (meshId == 0) {
				SIBR_LOG << "Mesh contains: colors: " << mesh->HasVertexColors(0)
					<< ", normals: " << mesh->HasNormals()
					<< ", texcoords: " << mesh->HasTextureCoords(0) << std::endl;
			}

			_triangles.reserve(offsetFaces + mesh->mNumFaces);
			for (uint i = 0; i < mesh->mNumFaces; ++i)
			{
				const aiFace* f = &mesh->mFaces[i];
				if (f->mNumIndices != 3)
					SIBR_LOG << "warning: discarding a face (not a triangle, num indices: "
					<< f->mNumIndices << ")" << std::endl;
				else
				{
					Vector3u tri = Vector3u(offsetVertices + f->mIndices[0], offsetVertices + f->mIndices[1], offsetVertices + f->mIndices[2]);
					if (tri[0] < 0 || tri[0] >= _vertices.size()
						|| tri[1] < 0 || tri[1] >= _vertices.size()
						|| tri[2] < 0 || tri[2] >= _vertices.size())
						SIBR_WRG << "face num [" << i << "] contains invalid vertex id(s)" << std::endl;
					else {
						_triangles.push_back(tri);
					}

				}
			}

			offsetFaces = (uint)_triangles.size();
			offsetVertices = (uint)_vertices.size();

		}

		_meshPath = filename;

		SIBR_LOG << "Mesh '" << filename << " successfully loaded. " << scene->mNumMeshes << " meshes were loaded with a total of "
			<< " (" << _triangles.size() << ") faces and "
			<< " (" << _vertices.size() << ") vertices detected. Init GL ..." << std::endl;
		SIBR_LOG << "Init GL mesh complete " << std::endl;

		_gl.dirtyBufferGL = true;
		return true;
	}
	
	bool	Mesh::loadSfM(const std::string& filename, const std::string& dataset_path )
	{
		// Does the file exist?
	
		std::string fname = dataset_path + "points3D.bin";

		std::cerr << "LOADSFM: Try to open " << fname << std::endl;

		if (!sibr::fileExists(fname)) {
			SIBR_LOG << "Error: can't load mesh '" << fname << "." << std::endl;
			return false;
		}
		Vertices verts;
		Colors cols;
		int numverts;

		ReadPoints3DBinary(fname, verts, cols, numverts);
		_triangles.clear();

		uint matId = 0;

		_vertices.resize(numverts);

		for (uint i = 0; i < numverts;  ++i)
				_vertices[i] = verts[i];

		_colors.resize(numverts);

		for (uint i = 0; i < numverts; ++i)
			_colors[i] = Vector3f(
				cols[i].x(), cols[i].y(), cols[i].z());

		_meshPath = dataset_path + "/points3D.bin";
		_renderingOptions.mode = PointRenderMode;

		SIBR_LOG << "SfM Mesh '" << filename << " successfully loaded. " << " (" << _vertices.size() << ") vertices detected. Init GL ..." << std::endl;
		SIBR_LOG << "Init GL mesh complete " << std::endl;

		_gl.dirtyBufferGL = true;
		return true;
	}


	bool sibr::Mesh::loadMtsXML(const std::string& xmlFile)
	{
		bool allLoaded = true;
		std::string pathFolder = boost::filesystem::path(xmlFile).parent_path().string();
		sibr::XMLTree doc(xmlFile);

		std::map<std::string, sibr::Mesh> meshes;
		std::map<std::string, std::string> idToFilename;

		rapidxml::xml_node<>* nodeScene = doc.first_node("scene");

		for (rapidxml::xml_node<>* node = nodeScene->first_node("shape");
			node; node = node->next_sibling("shape"))
		{
			if (strcmp(node->first_attribute()->name(), "type") == 0 &&
				strcmp(node->first_attribute()->value(), "shapegroup") == 0) {

				std::cout << "Found : " << node->first_attribute("id")->value() << std::endl;

				std::string id = node->first_attribute("id")->value();
				std::string filename = node->first_node("shape")->first_node("string")->first_attribute("value")->value();
				idToFilename[id] = filename;
			}
		}

		for (rapidxml::xml_node<>* node = nodeScene->first_node("shape");
			node; node = node->next_sibling("shape"))
		{

			if (strcmp(node->first_attribute()->name(), "type") == 0 &&
				strcmp(node->first_attribute()->value(), "instance") == 0
				) {
				rapidxml::xml_node<>* nodeRef = node->first_node("ref");
				const std::string id = nodeRef->first_attribute("id")->value();
				const std::string filename = idToFilename[id];
				const std::string meshPath = pathFolder + "/" + filename;

				if (meshes.find(filename) == meshes.end()) {
					meshes[filename] = sibr::Mesh();
					if (!meshes[filename].load(meshPath)) {
						return false;
					}

				}

				std::cout << "Adding one instance of : " << filename << std::endl;

				rapidxml::xml_node<>* nodeTrans = node->first_node("transform");
				if (nodeTrans) {
					sibr::Mesh toWorldMesh = meshes[filename];
					rapidxml::xml_node<>* nodeM1 = nodeTrans->first_node("matrix");
					std::string matrix1 = nodeM1->first_attribute("value")->value();
					rapidxml::xml_node<>* nodeM2 = nodeM1->next_sibling("matrix");
					std::string matrix2 = nodeM2->first_attribute("value")->value();


					std::istringstream issM1(matrix1);
					std::vector<std::string> splitM1(std::istream_iterator<std::string>{issM1},
						std::istream_iterator<std::string>());
					sibr::Matrix4f m1;
					m1 <<
						std::stof(splitM1[0]), std::stof(splitM1[1]), std::stof(splitM1[2]), std::stof(splitM1[3]),
						std::stof(splitM1[4]), std::stof(splitM1[5]), std::stof(splitM1[6]), std::stof(splitM1[7]),
						std::stof(splitM1[8]), std::stof(splitM1[9]), std::stof(splitM1[10]), std::stof(splitM1[11]),
						std::stof(splitM1[12]), std::stof(splitM1[13]), std::stof(splitM1[14]), std::stof(splitM1[15]);

					std::istringstream issM2(matrix2);
					std::vector<std::string> splitM2(std::istream_iterator<std::string>{issM2},
						std::istream_iterator<std::string>());
					sibr::Matrix4f m2;
					m2 <<
						std::stof(splitM2[0]), std::stof(splitM2[1]), std::stof(splitM2[2]), std::stof(splitM2[3]),
						std::stof(splitM2[4]), std::stof(splitM2[5]), std::stof(splitM2[6]), std::stof(splitM2[7]),
						std::stof(splitM2[8]), std::stof(splitM2[9]), std::stof(splitM2[10]), std::stof(splitM2[11]),
						std::stof(splitM2[12]), std::stof(splitM2[13]), std::stof(splitM2[14]), std::stof(splitM2[15]);

					sibr::Mesh::Vertices vertices;
					for (int v = 0; v < toWorldMesh.vertices().size(); v++) {
						sibr::Vector4f v4(toWorldMesh.vertices()[v].x(), toWorldMesh.vertices()[v].y(), toWorldMesh.vertices()[v].z(), 1.0);
						vertices.push_back((m2 * (m1 * v4)).xyz());

					}

					toWorldMesh.vertices(vertices);
					merge(toWorldMesh);

				}
				else {
					merge(meshes[filename]);
				}


			}
			else if (strcmp(node->first_attribute()->name(), "type") == 0 &&
				strcmp(node->first_attribute()->value(), "obj") == 0
				) {
				rapidxml::xml_node<>* nodeRef = node->first_node("string");
				const std::string filename = nodeRef->first_attribute("value")->value();
				const std::string meshPath = pathFolder + "/" + filename;

				if (meshes.find(filename) == meshes.end()) {
					meshes[filename] = sibr::Mesh();
					if (!meshes[filename].load(meshPath)) {
						return false;
					}

				}

				std::cout << "Adding one instance of : " << filename << std::endl;

				rapidxml::xml_node<>* nodeTrans = node->first_node("transform");
				if (nodeTrans) {
					sibr::Mesh toWorldMesh = meshes[filename];
					rapidxml::xml_node<>* nodeM1 = nodeTrans->first_node("matrix");
					std::string matrix1 = nodeM1->first_attribute("value")->value();


					std::istringstream issM1(matrix1);
					std::vector<std::string> splitM1(std::istream_iterator<std::string>{issM1},
						std::istream_iterator<std::string>());
					sibr::Matrix4f m1;
					m1 <<
						std::stof(splitM1[0]), std::stof(splitM1[1]), std::stof(splitM1[2]), std::stof(splitM1[3]),
						std::stof(splitM1[4]), std::stof(splitM1[5]), std::stof(splitM1[6]), std::stof(splitM1[7]),
						std::stof(splitM1[8]), std::stof(splitM1[9]), std::stof(splitM1[10]), std::stof(splitM1[11]),
						std::stof(splitM1[12]), std::stof(splitM1[13]), std::stof(splitM1[14]), std::stof(splitM1[15]);


					sibr::Mesh::Vertices vertices;
					for (int v = 0; v < toWorldMesh.vertices().size(); v++) {
						sibr::Vector4f v4(toWorldMesh.vertices()[v].x(), toWorldMesh.vertices()[v].y(), toWorldMesh.vertices()[v].z(), 1.0);
						vertices.push_back((m1 * v4).xyz());

					}

					toWorldMesh.vertices(vertices);
					merge(toWorldMesh);

				}
				else {
					merge(meshes[filename]);
				}


			}
		}

		return true;

	}

	void	Mesh::save(const std::string& filename, bool universal, const std::string &textureName) const
	{
		if (vertices().empty())
			SIBR_ERR << "cannot save this mesh (no vertices found)" << std::endl;
		// This function is a just a switch (so we can change format details
		// internally).

		const std::string ext = sibr::getExtension(filename);
		if (ext == "obj") {
			saveToObj(filename);
		}
		else {
			// If you encounter problem with the resulting mesh, you can switch
			// to the ASCII version for easy reading
			// Meshlab does not support uint16 colors, if you want to use the mesh in such
			// program, set 'universal' = true
			if (universal) {
				saveToASCIIPLY(filename, true, textureName);
			}
			else {
				saveToBinaryPLY(filename, false);
			}
		}

	}

	void	Mesh::vertices(const std::vector<float>& vertices)
	{
		_gl.dirtyBufferGL = true;
		_vertices.clear();

		// iterator for values
		std::vector<float>::const_iterator it = vertices.begin();

		//
		while (it != vertices.end())
		{
			Vector3f vertex;

			for (int i = 0; i < 3; ++i, ++it)
				vertex[i] = (*it);

			_vertices.push_back(vertex);
		}
	}

	void	Mesh::triangles(const std::vector<uint>& triangles)
	{
		_gl.dirtyBufferGL = true;
		_triangles.clear();

		// iterator for values
		std::vector<uint>::const_iterator it = triangles.begin();

		//
		while (it != triangles.end())
		{
			Vector3u triangle;

			for (int i = 0; i < 3; ++i, ++it)
				triangle[i] = (*it);

			_triangles.push_back(triangle);
		}
	}

	void		Mesh::texCoords(const std::vector<float>& texcoords)
	{
		_gl.dirtyBufferGL = true;
		_texcoords.clear();

		// iterator for values
		std::vector<float>::const_iterator it = texcoords.begin();

		//
		while (it != texcoords.end())
		{
			Vector2f texcoord;

			for (int i = 0; i < 2; ++i, ++it)
				texcoord[i] = (*it);

			_texcoords.push_back(texcoord);
		}
	}

	void	Mesh::normals(const std::vector<float>& normals)
	{
		_gl.dirtyBufferGL = true;
		_normals.clear();

		// iterator for values
		std::vector<float>::const_iterator it = normals.begin();

		//
		while (it != normals.end())
		{
			Vector3f normal;

			for (int i = 0; i < 3; ++i, ++it)
				normal[i] = (*it);

			_normals.push_back(normal);
		}
	}

	void	Mesh::generateNormals(void)
	{

		// will store a list of normals (of all triangles around each vertex)
		std::vector<std::vector<Vector3f>>	vertexNormals(_vertices.size());

		auto normalizeNormal = [](const Vector3f& normal) -> Vector3f {
			float len = normal.norm();
			if (len > std::numeric_limits<float>::epsilon())
				return normal / len;
			//else // may happen on tiny sharp edge, in this case points up
			return Vector3f(0.f, 1.f, 0.f);
		};

		int i = 0;
		for (const Vector3u& tri : _triangles)
		{
			i++;
			if (tri[0] > _vertices.size() || tri[1] > _vertices.size() ||
				tri[2] > _vertices.size()) {
				SIBR_ERR << "Incorrect indices (" << i << ") " << tri[0] << ":" << tri[1] << ":" << tri[2] << std::endl;
			}
			else {
				Vector3f u = _vertices[tri[0]] - _vertices[tri[2]];
				Vector3f v = _vertices[tri[0]] - _vertices[tri[1]];
				Vector3f normal = normalizeNormal(u.cross(v));

				vertexNormals[tri[0]].push_back(normal);
				vertexNormals[tri[1]].push_back(normal);
				vertexNormals[tri[2]].push_back(normal);
			}
		}

		_normals.resize(vertexNormals.size());
		for (uint i = 0; i < _normals.size(); ++i)
		{
			Vector3f n = std::accumulate(vertexNormals[i].begin(), vertexNormals[i].end(), Vector3f(0.f, 0.f, 0.f));
			n = (n / (float)vertexNormals.size());
			n = normalizeNormal(n);
			_normals[i] = -n;
		}

		_gl.dirtyBufferGL = true;
	}

	void	Mesh::generateSmoothNormals(int numIter)
	{
		SIBR_LOG << "Generate vertex normals..." << std::endl;
		// will store a list of normals (of all triangles around each vertex)
		std::vector<std::vector<Vector3f>>	vertexNormals(_vertices.size());

		auto normalizeNormal = [](const Vector3f& normal) -> Vector3f {
			float len = normal.norm();
			if (len > std::numeric_limits<float>::epsilon())
				return normal / len;
			//else // may happen on tiny sharp edge, in this case points up
			return Vector3f(0.f, 1.f, 0.f);
		};


		for (int i = 0; i < _triangles.size(); i++)
		{
			const Vector3u& tri = _triangles[i];
			if (tri[0] > _vertices.size() || tri[1] > _vertices.size() ||
				tri[2] > _vertices.size()) {
				SIBR_ERR << "Incorrect indices (" << i << ") " << tri[0] << ":" << tri[1] << ":" << tri[2] << std::endl;
			}
			else {
				Vector3f u = _vertices[tri[1]] - _vertices[tri[0]];
				Vector3f v = _vertices[tri[2]] - _vertices[tri[0]];
				Vector3f normal = u.cross(v);

				vertexNormals[tri[0]].push_back(normal);
				vertexNormals[tri[1]].push_back(normal);
				vertexNormals[tri[2]].push_back(normal);
			}
		}

		_normals.resize(vertexNormals.size());
		//#pragma omp parallel for
		for (int i = 0; i < _normals.size(); ++i)
		{
			Vector3f n = std::accumulate(vertexNormals[i].begin(), vertexNormals[i].end(), Vector3f(0.f, 0.f, 0.f));
			if (numIter == 0)//no iteration
				n = normalizeNormal(n);
			_normals[i] = n;
		}

		//Here we computed normals based on surrounding triangles

		for (int it = 0; it < numIter; it++) {

			std::vector<std::vector<Vector3f>>	vertexNormalsIter(_vertices.size());

			for (int i = 0; i < _triangles.size(); i++)
			{
				const Vector3u& tri = _triangles[i];
				if (tri[0] > _vertices.size() || tri[1] > _vertices.size() ||
					tri[2] > _vertices.size()) {
					SIBR_ERR << "Incorrect indices (" << i << ") " << tri[0] << ":" << tri[1] << ":" << tri[2] << std::endl;
				}
				else {
					for (int tId = 0; tId < 3; tId++) {
						Vector3f normal = _normals[tri[tId]];
						vertexNormalsIter[tri[(tId + 1) % 3]].push_back(normal);
						vertexNormalsIter[tri[(tId + 2) % 3]].push_back(normal);
					}
				}
			}

			float maxLength = 0.0f;
			for (int i = 0; i < _normals.size(); ++i)
			{
				Vector3f n = std::accumulate(vertexNormalsIter[i].begin(), vertexNormalsIter[i].end(), Vector3f(0.f, 0.f, 0.f));
				if (it + 1 == numIter)//last iteration
					n = normalizeNormal(n);
				_normals[i] = n;
				maxLength = std::max(maxLength, _normals[i].norm());
			}

			// To avoid float overflow after multiple iterations, we need to normalize.
			// But we can't just normalize each normal separately because we want to
			// preserve the relative triangle area weighting.
			// So instead we just send everything in [0,1] each time apart from the last iteration.
			if (maxLength > 0.0f && (it + 1 < numIter)) {
				for (int i = 0; i < _normals.size(); ++i)
				{
					_normals[i] /= maxLength;
				}
			}

		}

		_gl.dirtyBufferGL = true;
	}

	void	Mesh::generateSmoothNormalsDisconnected(int numIter)
	{
		SIBR_LOG << "Generate vertex normals..." << std::endl;
		// will store a list of normals (of all triangles around each vertex)
		std::vector<std::vector<Vector3f>>	vertexNormals(_vertices.size());

		auto normalizeNormal = [](const Vector3f& normal) -> Vector3f {
			float len = normal.norm();
			if (len > std::numeric_limits<float>::epsilon())
				return normal / len;
			//else // may happen on tiny sharp edge, in this case points up
			return Vector3f(0.f, 1.f, 0.f);
		};

		std::vector<std::pair<sibr::Vector3f, int>> vertCopy;
		for (int i = 0; i < _vertices.size(); ++i)
		{
			vertCopy.push_back(std::make_pair(_vertices[i], i));
		}

		std::sort(vertCopy.begin(), vertCopy.end());

		for (int i = 0; i < 100; ++i)
		{
			std::cout << "\t " << vertCopy[i].first << std::endl;
		}

		std::vector<int> v2firstCopy(_vertices.size(), -1);
		int dupCount = 0;
		for (int i = 0; i < _vertices.size(); ++i)
		{
			if (i % 1000 == 0)
				std::cout << "\t " << i << " of " << _vertices.size() << std::endl;

			int ii = i - 1;
			if ((vertCopy[ii].first - vertCopy[i].first).norm() > 0.000001f) {

				v2firstCopy[vertCopy[i].second] = vertCopy[i].second;
				continue;
			}
			else {
				dupCount++;
				v2firstCopy[vertCopy[i].second] = v2firstCopy[vertCopy[ii].second];
			}

		}

		std::cout << "Duplicates found :" << dupCount << std::endl;

		for (int i = 0; i < _triangles.size(); i++)
		{
			const Vector3u& tri = _triangles[i];
			if (tri[0] > _vertices.size() || tri[1] > _vertices.size() ||
				tri[2] > _vertices.size()) {
				SIBR_ERR << "Incorrect indices (" << i << ") " << tri[0] << ":" << tri[1] << ":" << tri[2] << std::endl;
			}
			else {
				Vector3f u = _vertices[tri[1]] - _vertices[tri[0]];
				Vector3f v = _vertices[tri[2]] - _vertices[tri[0]];
				Vector3f normal = u.cross(v);

				vertexNormals[tri[0]].push_back(normal);
				vertexNormals[tri[1]].push_back(normal);
				vertexNormals[tri[2]].push_back(normal);
			}
		}

		sibr::Mesh::Normals normalsCopy;
		normalsCopy.resize(vertexNormals.size());
		//#pragma omp parallel for
		for (int i = 0; i < normalsCopy.size(); ++i)
		{
			normalsCopy[i] = sibr::Vector3f(0, 0, 0);
		}
		for (int i = 0; i < normalsCopy.size(); ++i)
		{
			Vector3f n = std::accumulate(vertexNormals[i].begin(), vertexNormals[i].end(), Vector3f(0.f, 0.f, 0.f));
			normalsCopy[v2firstCopy[i]] += n;
		}

		//Here we computed normals based on surrounding triangles

		for (int it = 0; it < numIter; it++) {

			std::vector<std::vector<Vector3f>>	vertexNormalsIter(_vertices.size());

			for (int i = 0; i < _triangles.size(); i++)
			{
				const Vector3u& tri = _triangles[i];
				if (tri[0] > _vertices.size() || tri[1] > _vertices.size() ||
					tri[2] > _vertices.size()) {
					SIBR_ERR << "Incorrect indices (" << i << ") " << tri[0] << ":" << tri[1] << ":" << tri[2] << std::endl;
				}
				else {
					for (int tId = 0; tId < 3; tId++) {
						Vector3f normal = normalsCopy[v2firstCopy[tri[tId]]];
						vertexNormalsIter[tri[(tId + 1) % 3]].push_back(normal);
						vertexNormalsIter[tri[(tId + 2) % 3]].push_back(normal);
					}
				}
			}

			//#pragma omp parallel for
			for (int i = 0; i < normalsCopy.size(); ++i)
			{
				normalsCopy[i] = sibr::Vector3f(0, 0, 0);
			}
			for (int i = 0; i < normalsCopy.size(); ++i)
			{
				Vector3f n = std::accumulate(vertexNormalsIter[i].begin(), vertexNormalsIter[i].end(), Vector3f(0.f, 0.f, 0.f));
				normalsCopy[v2firstCopy[i]] += n;
			}

		}

		_normals.resize(normalsCopy.size());
		for (int i = 0; i < _normals.size(); ++i)
		{
			_normals[i] += normalizeNormal(normalsCopy[v2firstCopy[i]]);
		}

		_gl.dirtyBufferGL = true;
	}

	void Mesh::laplacianSmoothing(int numIter, bool updateNormals) {

		if (numIter < 1) {
			return;
		}

		/// Build neighbors information.
		/// \todo TODO: we could also detect vertices on the edges of the mesh to preserve their positions.
		std::vector<std::set<unsigned>> neighbors(_vertices.size());
		for (const sibr::Vector3u& tri : _triangles) {
			neighbors[tri[0]].emplace(tri[1]);
			neighbors[tri[0]].emplace(tri[2]);
			neighbors[tri[1]].emplace(tri[0]);
			neighbors[tri[1]].emplace(tri[2]);
			neighbors[tri[2]].emplace(tri[1]);
			neighbors[tri[2]].emplace(tri[0]);
		}

		/// Smooth by averaging.
		const size_t verticesSize = _vertices.size();

		for (int it = 0; it < numIter; ++it) {
			std::vector<sibr::Vector3f> newVertices(verticesSize);

			for (size_t vid = 0; vid < verticesSize; ++vid) {
				newVertices[vid] = sibr::Vector3f(0.0f, 0.0f, 0.f);
				for (const auto& ovid : neighbors[vid]) {
					newVertices[vid] += _vertices[ovid];
				}
				newVertices[vid] /= float(neighbors[vid].size());
			}

			vertices(newVertices);
		}

		if (updateNormals) {
			generateNormals();
		}
	}

	void Mesh::adaptativeTaubinSmoothing(int numIter, bool updateNormals) {

		if (numIter < 1) {
			return;
		}

		/// Build neighbors information.
		/// \todo TODO: we could also detect vertices on the edges of the mesh to preserve their positions.
		std::vector<std::set<unsigned>> neighbors(_vertices.size());
		std::map<int, std::map<int, std::set<float>>> cotanW;
		for (const sibr::Vector3u& tri : _triangles) {
			neighbors[tri[0]].emplace(tri[1]);
			neighbors[tri[0]].emplace(tri[2]);
			neighbors[tri[1]].emplace(tri[0]);
			neighbors[tri[1]].emplace(tri[2]);
			neighbors[tri[2]].emplace(tri[1]);
			neighbors[tri[2]].emplace(tri[0]);

			std::vector<sibr::Vector3f> vs;
			for (int i = 0; i < 3; i++)
				vs.push_back(_vertices[tri[i]]);

			for (int i = 0; i < 3; i++) {
				float angle = acos((vs[i] - vs[(i + 2) % 3]).normalized().dot((vs[(i + 1) % 3] - vs[(i + 2) % 3]).normalized()));
				cotanW[tri[i]][tri[(i + 1) % 3]].emplace(1.0f / (tan(angle) + 0.00001f));
				cotanW[tri[(i + 1) % 3]][tri[i]].emplace(1.0f / (tan(angle) + 0.00001f));
			}

		}

		/// Smooth by averaging.
		const size_t verticesSize = _vertices.size();

		std::vector<sibr::Vector3f> newColors(verticesSize);
		for (int it = 0; it < numIter; ++it) {
			std::vector<sibr::Vector3f> newVertices(verticesSize);
#pragma omp parallel for
			for (int vid = 0; vid < verticesSize; ++vid) {
				sibr::Vector3f v = _vertices[vid];
				sibr::Vector3f dtV = sibr::Vector3f(0.0f, 0.0f, 0.f);
				float totalW = 0;

				std::vector<sibr::Vector3f> colorsLocal;
				colorsLocal.push_back(_colors[vid]);
				for (const auto& ovid : neighbors[vid]) {
					float w = 0;
					for (const auto& cot : cotanW[vid][ovid]) {
						w += 0.5 * cot;
					}
					totalW += w;
					dtV += w * _vertices[ovid];
					colorsLocal.push_back(_colors[ovid]);
				}

				sibr::Vector3f meanColor;
				for (const auto& c : colorsLocal) {
					meanColor += c;
				}
				meanColor /= colorsLocal.size();
				sibr::Vector3f varColor;
				for (const auto& c : colorsLocal) {
					pow(c.x() - meanColor.x(), 2);
					varColor += sibr::Vector3f(pow(c.x() - meanColor.x(), 2), pow(c.y() - meanColor.y(), 2), pow(c.z() - meanColor.z(), 2));
				}
				varColor /= colorsLocal.size();

				newColors[vid] = varColor;

				if (totalW > 0) {
					dtV /= totalW;
					dtV = dtV - v;
					if (it % 2 == 0) {
						newVertices[vid] = v + 0.25 * dtV;
					}
					else {
						newVertices[vid] = v + 0.25 * dtV;
					}
				}
			}

			vertices(newVertices);
		}

		colors(newColors);
		if (updateNormals) {
			generateNormals();
		}
	}



	Mesh Mesh::generateSubMesh(std::function<bool(int)> func) const
	{

		sibr::Mesh::Vertices newVertices;
		sibr::Mesh::Triangles newTriangles;

		sibr::Mesh::Colors newColors;
		sibr::Mesh::Normals newNormals;
		sibr::Mesh::UVs newUVs;

		std::map<int, int> mapIdVert;

		int cmptValidVert = 0;
		int cmptVert = 0;

		sibr::Mesh::Colors oldColors;
		if (hasColors())
			oldColors = colors();

		sibr::Mesh::Normals oldNormals;
		if (hasNormals())
			oldNormals = normals();

		sibr::Mesh::UVs oldUVs;
		if (hasTexCoords())
			oldUVs = texCoords();

		for (int v = 0; v < vertices().size(); v++) {

			if (func(v)) {

				newVertices.push_back(vertices()[v]);

				if (hasColors())
					newColors.push_back(oldColors[cmptVert]);
				if (hasNormals())
					newNormals.push_back(oldNormals[cmptVert]);
				if (hasTexCoords())
					newUVs.push_back(oldUVs[cmptVert]);

				mapIdVert[cmptVert] = cmptValidVert;
				cmptValidVert++;

			}
			else {
				mapIdVert[cmptVert] = -1;
			}

			cmptVert++;

		}

		for (sibr::Vector3u t : triangles()) {

			if (mapIdVert[t.x()] != -1 &&
				mapIdVert[t.y()] != -1 &&
				mapIdVert[t.z()] != -1) {
				sibr::Vector3u newt(mapIdVert[t.x()], mapIdVert[t.y()], mapIdVert[t.z()]);
				newTriangles.push_back(newt);
			}
		}

		Mesh newMesh;
		newMesh.vertices(newVertices);
		newMesh.triangles(newTriangles);
		if (hasColors())
			newMesh.colors(newColors);
		if (hasNormals())
			newMesh.normals(newNormals);
		if (hasTexCoords())
			newMesh.texCoords(newUVs);

		return newMesh;
	}

	void	Mesh::forceBufferGLUpdate(bool adjacency) const
	{
		if (!_gl.bufferGL) { SIBR_ERR << "Tried to forceBufferGL on a non OpenGL Mesh" << std::endl; return; }
		_gl.dirtyBufferGL = false;
		_gl.bufferGL->build(*this, adjacency);
	}

	void	Mesh::freeBufferGLUpdate(void) const
	{
		_gl.dirtyBufferGL = false;
		_gl.bufferGL->free();
	}

	void	Mesh::render(bool depthTest, bool backFaceCulling, RenderMode mode, bool frontFaceCulling, bool invertDepthTest, bool tessellation, bool adjacency) const
	{
		if (!_gl.bufferGL) { SIBR_ERR << "Tried to render a non OpenGL Mesh" << std::endl; return; }

		if(adjacency && _renderingOptions.adjacency != adjacency)
		{
			_renderingOptions.adjacency = adjacency;
			_gl.dirtyBufferGL = true;
		}
		
		_renderingOptions.depthTest = depthTest;
		_renderingOptions.backFaceCulling = backFaceCulling;
		_renderingOptions.mode = mode;
		_renderingOptions.frontFaceCulling = frontFaceCulling;
		_renderingOptions.invertDepthTest = invertDepthTest;
		_renderingOptions.tessellation = tessellation;

		if (_gl.dirtyBufferGL)
			forceBufferGLUpdate(adjacency);

		if (depthTest)
			glEnable(GL_DEPTH_TEST);
		else
			glDisable(GL_DEPTH_TEST);

		if (backFaceCulling)
		{
			glEnable(GL_CULL_FACE);
			if (!frontFaceCulling)
				glCullFace(GL_BACK);
			else
				glCullFace(GL_FRONT);
		}
		else
			glDisable(GL_CULL_FACE);

		if (invertDepthTest) {
			glDepthFunc(GL_GEQUAL);
		}

		switch (mode)
		{
		case sibr::Mesh::FillRenderMode:
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			break;
		case sibr::Mesh::PointRenderMode:
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
			break;
		case sibr::Mesh::LineRenderMode:
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			break;
		default:
			break;
		}

		if (_triangles.size() != 0) {
			if (tessellation) {
				_gl.bufferGL->drawTessellated();
			}
			else {
				_gl.bufferGL->draw(adjacency);
			}
		}
		else if (_vertices.size() != 0) {
			_gl.bufferGL->draw_points();
		}

		// Reset default state (Policy is 'restore default values')
		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDepthFunc(GL_LESS);
	}

	void	Mesh::renderSubMesh(unsigned int begin, unsigned int end,
		bool depthTest,
		bool backFaceCulling,
		RenderMode mode,
		bool frontFaceCulling,
		bool invertDepthTest
	) const {
		if (!_gl.bufferGL) { SIBR_ERR << "Tried to render a non OpenGL Mesh" << std::endl; return; }
		if (_gl.dirtyBufferGL)
			forceBufferGLUpdate();

		if (depthTest)
			glEnable(GL_DEPTH_TEST);
		else
			glDisable(GL_DEPTH_TEST);

		if (backFaceCulling)
		{
			glEnable(GL_CULL_FACE);
			if (!frontFaceCulling)
				glCullFace(GL_BACK);
			else
				glCullFace(GL_FRONT);
		}
		else
			glDisable(GL_CULL_FACE);

		if (invertDepthTest) {
			glDepthFunc(GL_GEQUAL);
		}

		switch (mode)
		{
		case sibr::Mesh::FillRenderMode:
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			break;
		case sibr::Mesh::PointRenderMode:
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
			break;
		case sibr::Mesh::LineRenderMode:
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			break;
		default:
			break;
		}

		if (_triangles.size() != 0) {
			_gl.bufferGL->draw(begin, end);
		}
		else if (_vertices.size() != 0) {
			_gl.bufferGL->draw_points(begin, end);
		}

		// Reset default state (Policy is 'restore default values')
		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDepthFunc(GL_LESS);
	}


	void	Mesh::render_points(void) const
	{
		forceBufferGLUpdate();
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
		_gl.bufferGL->draw_points();
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	void	Mesh::render_points(bool depthTest) const
	{
		if (depthTest) {
			glEnable(GL_DEPTH_TEST);
		}
		else {
			glDisable(GL_DEPTH_TEST);
		}

		render_points();

		glDisable(GL_DEPTH_TEST);
	}

	void	Mesh::render_lines(void) const
	{
		if (!_gl.bufferGL) { SIBR_ERR << "Tried to render a non OpenGL Mesh" << std::endl; return; }
		if (_gl.dirtyBufferGL)
			forceBufferGLUpdate();
		_gl.bufferGL->draw_lines();
	}

	Mesh::SubMesh Mesh::extractSubMesh(const std::vector<int>& newVerticesIds, VERTEX_LIST_CHOICE v_choice) const
	{

		int numOldVertices = (int)vertices().size();
		bool keep_from_list = (v_choice == VERTEX_LIST_CHOICE::KEEP);
		std::vector<bool> willBeKept(numOldVertices, !keep_from_list);
		for (int id : newVerticesIds) {
			if (id >= 0 && id < numOldVertices) {
				willBeKept[id] = keep_from_list;
			}
		}

		int numValidNewVertices = 0;
		for (bool b : willBeKept) {
			if (b) {
				++numValidNewVertices;
			}
		}

		std::vector<int> oldToNewVertexId(numOldVertices, -1);

		sibr::Mesh::Vertices newVertices(numValidNewVertices);
		sibr::Mesh::Triangles newTriangles;

		sibr::Mesh::Colors newColors;
		sibr::Mesh::Normals newNormals;
		sibr::Mesh::UVs newUVs;

		if (hasColors()) {
			newColors.resize(numValidNewVertices);
		}
		if (hasNormals()) {
			newNormals.resize(numValidNewVertices);
		}
		if (hasTexCoords()) {
			newUVs.resize(numValidNewVertices);
		}

		int new_vertex_id = 0;
		for (int id = 0; id < numOldVertices; ++id) {
			if (willBeKept[id]) {
				newVertices[new_vertex_id] = vertices()[id];
				oldToNewVertexId[id] = new_vertex_id;

				if (hasColors()) {
					newColors[new_vertex_id] = colors()[id];
				}
				if (hasNormals()) {
					newNormals[new_vertex_id] = normals()[id];
				}
				if (hasTexCoords()) {
					newUVs[new_vertex_id] = texCoords()[id];
				}
				++new_vertex_id;
			}
		}

		std::vector<bool> isInRemovedTriangle(numOldVertices, false);
		for (const auto& t : triangles()) {
			sibr::Vector3i newVerticesId = t.cast<int>().unaryExpr([&oldToNewVertexId](int v_id) { return oldToNewVertexId[v_id];  });
			if (newVerticesId.unaryViewExpr([](int v_id) { return v_id >= 0 ? 1 : 0; }).all()) {
				newTriangles.push_back(newVerticesId.cast<unsigned>());
			}
			else {
				for (int c = 0; c < 3; c++) {
					isInRemovedTriangle[t[c]] = true;
				}
			}
		}

		bool oldMeshHasGraphics = (_gl.bufferGL.get() != nullptr);

		Mesh::SubMesh subMesh;
		subMesh.meshPtr = std::make_shared<sibr::Mesh>(oldMeshHasGraphics);
		sibr::Mesh& mesh = *subMesh.meshPtr;
		mesh.vertices(newVertices);
		mesh.triangles(newTriangles);
		if (hasColors()) {
			mesh.colors(newColors);
		}
		if (hasNormals()) {
			mesh.normals(newNormals);
		}
		if (hasTexCoords()) {
			mesh.texCoords(newUVs);
		}

		for (int id = 0; id < numOldVertices; ++id) {
			if (isInRemovedTriangle[id]) {
				subMesh.complementaryVertices.push_back(id);
			}
		}

		return subMesh;
	}

	sibr::Mesh Mesh::invertedFacesMesh() const
	{
		sibr::Mesh invertedFacesMesh(_gl.bufferGL != nullptr);
		invertedFacesMesh.vertices(vertices());
		if (hasColors()) {
			invertedFacesMesh.colors(colors());
		}
		if (hasNormals()) {
			invertedFacesMesh.normals(normals());
		}
		if (hasTexCoords()) {
			invertedFacesMesh.texCoords(texCoords());
		}

		sibr::Mesh::Triangles invertedTriangles(triangles().size());
		for (int t_id = 0; t_id < (int)triangles().size(); ++t_id) {
			invertedTriangles[t_id] = triangles()[t_id].yxz();
		}
		invertedFacesMesh.triangles(invertedTriangles);

		return invertedFacesMesh;
	}

	sibr::Mesh::Ptr Mesh::invertedFacesMesh2() const
	{
		auto invertedFacesMesh = std::make_shared<Mesh>(_gl.bufferGL != nullptr);

		int nVertices = (int)vertices().size();
		int nTriangles = (int)triangles().size();

		Mesh::Vertices Nvertices(2 * nVertices);
		Mesh::Triangles Ntriangles(2 * nTriangles);
		Mesh::Colors Ncolors(hasColors() ? 2 * nVertices : 0);
		Mesh::Normals Nnormals(hasNormals() ? 2 * nVertices : 0);
		Mesh::UVs Nuvs(hasTexCoords() ? 2 * nVertices : 0);

		int v_id = 0;
		for (const auto& v : vertices()) {
			Nvertices[v_id] = v;
			Nvertices[v_id + nVertices] = v;

			if (hasNormals()) {
				Nnormals[v_id] = normals()[v_id];
				Nnormals[v_id + nVertices] = -normals()[v_id];
			}

			if (hasColors()) {
				Ncolors[v_id] = colors()[v_id];
				Ncolors[v_id + nVertices] = colors()[v_id];
			}

			if (hasTexCoords()) {
				Nuvs[v_id] = texCoords()[v_id];
				Nuvs[v_id + nVertices] = texCoords()[v_id];
			}

			++v_id;
		}
		invertedFacesMesh->vertices(Nvertices);

		if (hasNormals()) {
			invertedFacesMesh->normals(Nnormals);
		}
		if (hasColors()) {
			invertedFacesMesh->colors(Ncolors);
		}
		if (hasTexCoords()) {
			invertedFacesMesh->texCoords(Nuvs);
		}

		sibr::Vector3u shift(nVertices, nVertices, nVertices);
		int t_id = 0;
		for (const auto& t : triangles()) {
			Ntriangles[t_id] = t;
			Ntriangles[t_id + nTriangles] = t.yxz() + shift;
			++t_id;
		}
		invertedFacesMesh->triangles(Ntriangles);

		return invertedFacesMesh;
	}

	const std::string Mesh::getMeshFilePath(void) const
	{
		return _meshPath;
	}

	void					Mesh::getBoundingSphere(Vector3f& outCenter, float& outRadius, bool referencedOnly, bool usePCcenter) const
	{
		// Get the center of mass

		double totalArea = 0, currentArea;
		double xCenter = 0, yCenter = 0, zCenter = 0;

		const Triangles& tri = _triangles;
		const Vertices& vert = _vertices;
		if (usePCcenter) {
			sibr::Vector3d outCenterDbl;
			for (const Vector3f& v : vert)
			{
				outCenterDbl += v.cast<double>();
			}

			outCenter = (outCenterDbl / vert.size()).cast<float>();
		}
		else {
			if (tri.size() == 0) {
				SIBR_WRG << "No triangles found for evaluation of sphere center, result will be NaN";
			}
			for (const Vector3u& t : tri)
			{
				float trix1 = vert[t[0]].x();
				float triy1 = vert[t[0]].y();
				float triz1 = vert[t[0]].z();

				float trix2 = vert[t[1]].x();
				float triy2 = vert[t[1]].y();
				float triz2 = vert[t[1]].z();

				float trix3 = vert[t[2]].x();
				float triy3 = vert[t[2]].y();
				float triz3 = vert[t[2]].z();

				currentArea = ((vert[t[1]] - vert[t[0]]).cross(vert[t[2]] - vert[t[0]])).norm() / 2.0;
				totalArea += currentArea;

				xCenter += ((trix1 + trix2 + trix3) / 3) * currentArea;
				yCenter += ((triy1 + triy2 + triy3) / 3) * currentArea;
				zCenter += ((triz1 + triz2 + triz3) / 3) * currentArea;
			}

			outCenter = Vector3f(float(xCenter / totalArea), float(yCenter / totalArea), float(zCenter / totalArea));
		}


		outRadius = 0.f;
		if (referencedOnly) {
			for (const Vector3u& t : tri)
			{
				outRadius = std::max(outRadius, distance(vert[t[0]], outCenter));
				outRadius = std::max(outRadius, distance(vert[t[1]], outCenter));
				outRadius = std::max(outRadius, distance(vert[t[2]], outCenter));
			}
		}
		else {
			for (const Vector3f& v : vert)
				outRadius = std::max(outRadius, distance(v, outCenter));
		}
	}


	Eigen::AlignedBox<float, 3> Mesh::getBoundingBox(void) const
	{
		Eigen::AlignedBox<float, 3> box;
		for (const auto& vertex : _vertices) {
			box.extend(vertex);
		}
		return box;
	}

	sibr::Mesh::Ptr sibr::Mesh::getEnvSphere(sibr::Vector3f center, float radius, sibr::Vector3f zenith, sibr::Vector3f north,
		PartOfSphere part) {

		sibr::Vector3f east = north.cross(zenith);
		sibr::Mesh::Ptr envMesh(new sibr::Mesh());
		sibr::Mesh::Vertices vert;
		sibr::Mesh::UVs uvs;
		sibr::Mesh::Triangles tri;

		int highLimit = 0, lowLimit = 0;
		switch (part)
		{
		case PartOfSphere::WHOLE:
			highLimit = 90;
			lowLimit = -90;
			break;
		case PartOfSphere::UP:
			highLimit = 90;
			lowLimit = 0;
			break;
		case PartOfSphere::BOTTOM:
			highLimit = 0;
			lowLimit = -90;
			break;
		}

		for (int lat = lowLimit; lat <= highLimit; lat++) {
			for (int lgt = 0; lgt <= 360; lgt++) {

				sibr::Vector3f point = cos(0.5 * M_PI * lat / 90.0f) * (cos(2 * M_PI * lgt / 360.0f) * north + sin(2 * M_PI * lgt / 360.0f) * east)
					+ sin(0.5 * M_PI * lat / 90.0f) * zenith;

				vert.push_back(10.f * radius * point + center);

				uvs.push_back(sibr::Vector2f(lgt / 360.0f, 0.5 + lat / 180.0f));

			}
		}

		for (int lat = lowLimit; lat < highLimit; lat++) {
			for (int lgt = 0; lgt < 360; lgt++) {

				int delta = 1;
				int lgtShift = lgt + 361 * (lat - lowLimit);
				tri.push_back(sibr::Vector3u(lgtShift, lgtShift + delta, lgtShift + 361 + delta));
				tri.push_back(sibr::Vector3u(lgtShift, lgtShift + 361 + delta, lgtShift + 361));
			}
		}
		envMesh->vertices(vert);
		envMesh->texCoords(uvs);
		envMesh->triangles(tri);

		return envMesh;
	}

	sibr::Vector3f Mesh::centroid() const
	{
		sibr::Vector3d centroid(0, 0, 0);
		for (auto& vertex : _vertices) {
			centroid += vertex.cast<double>();
		}
		if (_vertices.size() > 0) {
			centroid /= static_cast<double>(_vertices.size());
		}
		return centroid.cast<float>();
	}

	std::stringstream Mesh::getOffStream(bool verbose) const
	{
		if (verbose) {
			std::cout << "[sibr::Mesh::getOffStream()] ... " << std::flush;
		}

		std::stringstream s;
		s << "OFF \n \n" << vertices().size() << " " << triangles().size() << " 0 " << std::endl;

		for (const auto& v : vertices()) {
			s << v.x() << " " << v.y() << " " << v.z() << std::endl;
		}

		for (const auto& t : triangles()) {
			s << "3 " << t.x() << " " << t.y() << " " << t.z() << std::endl;
		}

		if (verbose) {
			std::cout << " done " << std::endl;
		}

		return s;
	}

	void Mesh::fromOffStream(std::stringstream& stream, bool computeNormals)
	{
		int n_vert;
		int n_faces;
		int n_edges;

		std::string line;
		safeGetline(stream, line);
		safeGetline(stream, line);
		std::istringstream iss(line);

		iss >> n_vert >> n_faces >> n_edges;

		_vertices.resize(n_vert);
		for (int v = 0; v < n_vert; ++v) {
			safeGetline(stream, line);
			std::istringstream lineStream(line);

			lineStream >> _vertices[v][0] >> _vertices[v][1] >> _vertices[v][2];

		}

		_triangles.resize(0);
		_triangles.reserve(3 * n_faces);
		int face_size;
		for (int t = 0; t < n_faces; ++t) {
			safeGetline(stream, line);
			std::istringstream lineStream(line);

			lineStream >> face_size;

			if (face_size == 3) {
				sibr::Vector3u t;
				lineStream >> t[0] >> t[1] >> t[2];
				_triangles.push_back(t);

			}
			else if (face_size == 4) {
				sibr::Vector3u t1, t2;
				lineStream >> t1[0] >> t1[1] >> t1[2] >> t2[2];
				t2[0] = t1[0];
				t2[1] = t1[2];
				_triangles.push_back(t1);
				_triangles.push_back(t2);
			}
		}

		if (computeNormals) {
			generateNormals();
		}

		if (_gl.bufferGL.get()) {
			_gl.dirtyBufferGL = true;
		}
	}

	sibr::Mesh::Ptr Mesh::getTestCube(bool withGraphics)
	{
		std::vector<sibr::Vector3f> vertices = {
			{ +1, +1, +1 }, { -1, +1, +1 }, { -1, -1, +1 },
			{ +1, -1, +1 }, { +1, +1, -1 }, { -1, +1, -1 },
			{ -1, -1, -1 }, { +1, -1, -1 }
		};

		std::vector<sibr::Vector3u> indices = {
			{ 0, 1, 2 }, { 0, 2, 3 }, { 7, 4, 0 }, { 7, 0, 3 }, { 4, 5, 1 },
			{ 4, 1, 0 }, { 5, 6, 2 }, { 5, 2, 1 }, { 3, 2, 6 }, { 3, 6, 7 },
			{ 6, 5, 4 }, { 6, 4, 7 }
		};

		sibr::Mesh::Ptr mesh(new sibr::Mesh(withGraphics));
		mesh->vertices(vertices);
		mesh->triangles(indices);
		mesh->generateNormals();

		return mesh;
	}

	Mesh::Ptr Mesh::getSphereMesh(const Vector3f& center, float radius, bool withGraphics, int precision) {
		const int nTheta = precision;
		const int nPhi = precision;
		const int nPoints = nTheta * nPhi;
		std::vector<Vector3f> vertices(nPoints), normals(nPoints);

		for (int t = 0; t < nTheta; ++t) {
			double theta = (t / (double)(nTheta - 1)) * M_PI;
			double cosT = std::cos(theta);
			double sinT = std::sin(theta);
			for (int p = 0; p < nPhi; ++p) {
				double phi = 2.0 * (p / (double)(nPhi - 1) - 0.5) * M_PI;
				double cosP = std::cos(phi), sinP = std::sin(phi);
				normals[p + nPhi * t] = Vector3d(sinT * cosP, sinT * sinP, cosT).cast<float>();
				vertices[p + nPhi * t] = center + radius * normals[p + nPhi * t];
			}
		}

		std::vector<uint> indices(6 * (nTheta - 1) * nPhi);
		int triangle_id = 0;
		for (int t = 0; t < nTheta - 1; ++t) {
			for (int p = 0; p < nPhi; ++p) {
				int current_id = p + nPhi * t;
				int offset_row = 1 - (p == nPhi - 1 ? nPhi : 0);
				int next_in_row = current_id + offset_row;
				int next_in_col = current_id + nPhi;
				int next_next = next_in_col + offset_row;
				indices[3 * triangle_id + 0] = current_id;
				indices[3 * triangle_id + 1] = next_in_col;
				indices[3 * triangle_id + 2] = next_in_row;
				indices[3 * triangle_id + 3] = next_in_row;
				indices[3 * triangle_id + 4] = next_in_col;
				indices[3 * triangle_id + 5] = next_next;
				triangle_id += 2;
			}
		}

		Mesh::Ptr sphereMesh = Mesh::Ptr(new Mesh(withGraphics));
		sphereMesh->vertices(vertices);
		sphereMesh->normals(normals);
		sphereMesh->triangles(indices);

		return sphereMesh;
	}

	sibr::Mesh::Ptr Mesh::subDivide(float limitSize, size_t maxRecursion) const
	{
		struct Less {
			bool operator()(const sibr::Vector3f& a, const sibr::Vector3f& b) const {
				return a < b;
			}
		};

		struct Edge {
			sibr::Vector3f midPoint;
			sibr::Vector3f midNormal;
			std::vector<int> triangles_ids;
			float length;
			int v_ids[2];
		};

		struct Triangle {
			Triangle() : edges_ids(std::vector<int>(3, -1)), edges_flipped(std::vector<bool>(3, false)) {}
			std::vector<int> edges_ids;
			std::vector<bool> edges_flipped;
		};

		auto subMeshPtr = std::make_shared<sibr::Mesh>();

		std::map<sibr::Vector3f, int, Less> mapEdges;
		std::vector<Edge> edges;
		std::vector<Triangle> tris(triangles().size());

		int t_id = 0;
		int e_id = 0;
		for (const auto& t : triangles()) {
			bool degenerate = false;
			for (int k = 0; k < 3; ++k) {
				int v0 = t[k];
				int v1 = t[(k + 1) % 3];
				// Skip degenerate faces.
				if (v0 == v1) {
					degenerate = true;
					break;
				}
				const sibr::Vector3f midPoint = 0.5f * (vertices()[v0] + vertices()[v1]);
				sibr::Vector3f midNormal(0.0f, 0.0f, 0.0f);
				if (hasNormals()) {
					midNormal = (0.5f * (normals()[v0] + normals()[v1])).normalized();
				}

				const float length = (vertices()[v0] - vertices()[v1]).norm();
				if (mapEdges.count(midPoint) == 0) {
					mapEdges[midPoint] = e_id;
					const Edge edge = { midPoint, midNormal, {t_id}, length, {v0,v1} };
					tris[t_id].edges_ids[k] = e_id;
					edges.push_back(edge);
					++e_id;
				}
				else {
					const int edge_id = mapEdges[midPoint];
					edges[edge_id].triangles_ids.push_back(t_id);
					tris[t_id].edges_ids[k] = edge_id;
					if (v0 != edges[edge_id].v_ids[0]) {
						tris[t_id].edges_flipped[k] = true;
					}
				}
			}
			if (degenerate) {
				continue;
			}
			++t_id;
		}

		const int nOldVertices = (int)vertices().size();
		sibr::Mesh::Vertices newVertices = vertices();
		sibr::Mesh::Normals newNormals = normals();

		std::vector<int> edge_to_divided_edges(edges.size(), -1);

		sibr::Mesh::Triangles newTriangles;

		bool dbg = false;
		int num_divided_edges = 0;
		for (const Triangle& t : tris) {
			// Ignore undef triangles.
			if (t.edges_ids[0] == -1 && t.edges_ids[1] == -1 && t.edges_ids[2] == -1) {
				continue;
			}
			std::vector<int> ks(3, -1);
			std::vector<int> non_ks(3, -1);
			for (int k = 0; k < 3; ++k) {
				const int e_id = t.edges_ids[k];
				if (edges[e_id].length > limitSize) {
					if (ks[0] < 0) {
						ks[0] = k;
					}
					else if (ks[1] < 0) {
						ks[1] = k;
					}
					else {
						ks[2] = k;
					}

					if (edge_to_divided_edges[e_id] < 0) {
						edge_to_divided_edges[e_id] = num_divided_edges;
						newVertices.push_back(edges[e_id].midPoint);
						newNormals.push_back(edges[e_id].midNormal);
						++num_divided_edges;

					}

				}
				else {
					if (non_ks[0] < 0) {
						non_ks[0] = k;
					}
				}
			}

			const sibr::Vector3i corners_ids = sibr::Vector3i(0, 1, 2).unaryViewExpr([&](int i) {
				return edges[t.edges_ids[i]].v_ids[t.edges_flipped[i] ? 1 : 0];
				});
			const sibr::Vector3i midpoints_ids = sibr::Vector3i(0, 1, 2).unaryViewExpr([&](int i) {
				return  edge_to_divided_edges[t.edges_ids[i]] >= 0 ? nOldVertices + edge_to_divided_edges[t.edges_ids[i]] : -1;
				});

			if (ks[2] >= 0) {
				newTriangles.push_back(sibr::Vector3u(corners_ids[0], midpoints_ids[0], midpoints_ids[2]));
				newTriangles.push_back(sibr::Vector3u(corners_ids[1], midpoints_ids[1], midpoints_ids[0]));
				newTriangles.push_back(sibr::Vector3u(corners_ids[2], midpoints_ids[2], midpoints_ids[1]));
				newTriangles.push_back(midpoints_ids.cast<unsigned>());
			}
			else if (ks[1] >= 0) {
				const int candidate_edge_1_v_id_1 = corners_ids[non_ks[0]];
				const int candidate_edge_1_v_id_2 = midpoints_ids[(non_ks[0] + 1) % 3];
				const int candidate_edge_2_v_id_1 = corners_ids[(non_ks[0] + 1) % 3];
				const int candidate_edge_2_v_id_2 = midpoints_ids[(non_ks[0] + 2) % 3];
				const float candidate_edge_1_norm = (newVertices[candidate_edge_1_v_id_1] - newVertices[candidate_edge_1_v_id_2]).norm();
				const float candidate_edge_2_norm = (newVertices[candidate_edge_2_v_id_1] - newVertices[candidate_edge_2_v_id_2]).norm();
				if (candidate_edge_1_norm < candidate_edge_2_norm) {
					newTriangles.push_back(sibr::Vector3u(candidate_edge_1_v_id_1, corners_ids[(non_ks[0] + 1) % 3], candidate_edge_1_v_id_2));
					newTriangles.push_back(sibr::Vector3u(candidate_edge_1_v_id_2, midpoints_ids[(non_ks[0] + 2) % 3], candidate_edge_1_v_id_1));
				}
				else {
					newTriangles.push_back(sibr::Vector3u(candidate_edge_2_v_id_1, candidate_edge_2_v_id_2, corners_ids[non_ks[0]]));
					newTriangles.push_back(sibr::Vector3u(candidate_edge_2_v_id_1, midpoints_ids[(non_ks[0] + 1) % 3], candidate_edge_2_v_id_2));
				}
				newTriangles.push_back(sibr::Vector3u(midpoints_ids[(non_ks[0] + 1) % 3], corners_ids[(non_ks[0] + 2) % 3], midpoints_ids[(non_ks[0] + 2) % 3]));
			}
			else if (ks[0] >= 0) {
				newTriangles.push_back(sibr::Vector3u(midpoints_ids[ks[0]], corners_ids[(ks[0] + 1) % 3], corners_ids[(ks[0] + 2) % 3]));
				newTriangles.push_back(sibr::Vector3u(midpoints_ids[ks[0]], corners_ids[(ks[0] + 2) % 3], corners_ids[ks[0]]));
			}
			else {
				newTriangles.push_back(corners_ids.cast<unsigned>());
			}
		}
		std::cout << "." << std::flush;

		subMeshPtr->vertices(newVertices);
		if (hasNormals()) {
			subMeshPtr->normals(newNormals);
		}
		subMeshPtr->triangles(newTriangles);
		if (num_divided_edges > 0 && maxRecursion > 0) {
			return subMeshPtr->subDivide(limitSize, maxRecursion - 1);
		}

		return subMeshPtr;
	}

	float Mesh::meanEdgeSize() const
	{
		double sumSizes = 0;
		for (const auto& t : triangles()) {
			const auto& v1 = vertices()[t[0]];
			const auto& v2 = vertices()[t[1]];
			const auto& v3 = vertices()[t[2]];
			sumSizes += (double)((v1 - v2).norm() + (v2 - v3).norm() + (v3 - v1).norm());
		}

		return (float)(sumSizes / (3 * triangles().size()));
	}

	Mesh::Ptr Mesh::clone() const
	{
		auto outMesh = std::make_shared<sibr::Mesh>(_gl.bufferGL != nullptr);
		outMesh->vertices(vertices());
		outMesh->triangles(triangles());

		if (hasNormals()) {
			outMesh->normals(normals());
		}

		if (hasColors()) {
			outMesh->colors(colors());
		}

		if (hasTexCoords()) {
			outMesh->texCoords(texCoords());
		}

		return outMesh;
	}

	void		Mesh::merge(const Mesh& other)
	{
		bool withGraphics = (_gl.bufferGL != nullptr);

		if (_vertices.empty())
		{
			this->operator = (other);
		}
		else
		{
			Vertices	vertices;
			Normals		normals;
			Colors		colors;
			UVs			texcoords;

			const uint offset = static_cast<uint>(_vertices.size());
			Vector3u	triOffset(offset, offset, offset);
			Triangles	triangles = other.triangles();

			for (Vector3u& t : triangles)
				t += triOffset;


			if (hasNormals())
				_normals.insert(_normals.end(), other.normals().begin(), other.normals().end());

			if (hasColors())
				_colors.insert(_colors.end(), other.colors().begin(), other.colors().end());

			if (hasTexCoords())
				_texcoords.insert(_texcoords.end(), other.texCoords().begin(), other.texCoords().end());


			_vertices.insert(_vertices.end(), other.vertices().begin(), other.vertices().end());
			_triangles.insert(_triangles.end(), triangles.begin(), triangles.end());

		}

		if (withGraphics)
			_gl.bufferGL.reset(new MeshBufferGL);
	}

	void sibr::Mesh::makeWhole(void)
	{
		if (!hasNormals())
			_normals = Normals(vertices().size());

		if (!hasColors())
			_colors = Colors(vertices().size());

		if (!hasTexCoords())
			_texcoords = UVs(vertices().size());

	}

	void		Mesh::eraseTriangles(const std::vector<uint>& faceIDList)
	{
		uint indexMax = 0;
		for (uint i = 0; i < triangles().size(); ++i)
			indexMax = std::max(indexMax, triangles()[i].maxCoeff());

		std::vector<bool>	faceToErase(triangles().size(), false);
		for (uint faceID : faceIDList)
			faceToErase[faceID] = true;


		Mesh::Triangles		newTris;
		Mesh::Vertices		newVerts;
		std::vector<int>	indexRemap(indexMax + 1, -1);

		newTris.reserve(triangles().size());
		newVerts.reserve(vertices().size());

		for (uint i = 0; i < triangles().size(); ++i)
		{
			if (faceToErase[i])
				continue;

			Vector3u t = triangles()[i];
			Vector3u newT;
			for (uint j = 0; j < 3; ++j)
			{
				if (indexRemap[t[j]] == -1)
				{
					indexRemap[t[j]] = (int)newVerts.size();
					newVerts.push_back(vertices().at(t[j]));
				}
				newT[j] = (uint)indexRemap[t[j]];

			}

			newTris.push_back(newT);
		}

		if (hasColors())
		{
			Mesh::Colors	newColors(newVerts.size());
			for (uint i = 0; i < indexRemap.size(); ++i)
				if (indexRemap[i] != -1)
					newColors[indexRemap[i]] = colors().at(i);
			colors(newColors);
		}

		if (hasNormals())
		{
			Mesh::Normals	newNormals(newVerts.size());
			for (uint i = 0; i < indexRemap.size(); ++i)
				if (indexRemap[i] != -1)
					newNormals[indexRemap[i]] = normals().at(i);
			normals(newNormals);
		}


		if (hasTexCoords())
		{
			Mesh::UVs	newUVs(newVerts.size());
			for (uint i = 0; i < indexRemap.size(); ++i)
				if (indexRemap[i] != -1)
					newUVs[indexRemap[i]] = texCoords().at(i);
			texCoords(newUVs);
		}

		triangles(newTris);
		vertices(newVerts);
	}

	std::vector<std::vector<int> > Mesh::removeDisconnectedComponents()
	{
		std::vector<std::vector<int> > allComponents;

		std::vector<std::vector<int>> v_triangles(vertices().size());
		int t_id = 0;
		for (const auto& t : triangles()) {
			for (int k = 0; k < 3; ++k) {
				v_triangles[t[k]].push_back(t_id);
			}
			++t_id;
		}


		std::vector<bool> wasVisited(vertices().size(), false);
		int v_id = 0;
		for (const auto& v : vertices()) {

			if (!wasVisited[v_id]) {
				std::priority_queue<int> next_ids;
				next_ids.push(v_id);
				wasVisited[v_id] = true;

				std::vector<int> component;

				while (next_ids.size() > 0) {
					int next_id = next_ids.top();
					next_ids.pop();
					component.push_back(next_id);

					for (int t : v_triangles[next_id]) {
						for (int k = 0; k < 3; ++k) {
							int other_v_id = triangles()[t][k];
							if (!wasVisited[other_v_id]) {
								next_ids.push(other_v_id);
								wasVisited[other_v_id] = true;
							}
						}
					}
				}

				allComponents.emplace_back(component);
			}
			++v_id;
		}

		return allComponents;
	}

} // namespace sibr
