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
#include "core/graphics/MaterialMesh.hpp"
#include "core/system/Transform3.hpp"
#include "boost/filesystem.hpp"
#include "core/system/XMLTree.h"
#include "core/system/Matrix.hpp"
#include <set>
#include <boost/variant/detail/substitute.hpp>

namespace sibr
{
	bool	MaterialMesh::load(const std::string& filename)
	{

		srand(static_cast <unsigned> (time(0)));
		Assimp::Importer	importer;
		importer.SetPropertyBool(AI_CONFIG_PP_FD_REMOVE, true);
		// cause Assimp to remove all degenerated faces as soon as they are detected
		const aiScene* scene = importer.ReadFile(filename,
			aiProcess_Triangulate | aiProcess_JoinIdenticalVertices |
			aiProcess_FindDegenerates);

		if (!scene)
		{
			SIBR_WRG << "error: can't load mesh '" << filename
				<< "' (" << importer.GetErrorString() << ")." << std::endl;
			return false;
		}

		if (scene->mNumMeshes == 0)
		{
			SIBR_WRG << "error: the loaded model file ('" << filename
				<< "') contains zero or more than one mesh. Number of meshes : " <<
				scene->mNumMeshes << std::endl;
			return false;
		}

		auto convertVec = [](const aiVector3D& v) {
			return Vector3f(v.x, v.y, v.z); };
		_triangles.clear();

		uint offsetVertices = 0;
		uint offsetFaces = 0;
		uint matId = 0; // Material
		std::map<std::string, int> matName2Id; // Material

		_maxMeshId = size_t(int(scene->mNumMeshes) - 1);

		SIBR_LOG << "Mesh with " << scene->mNumMeshes << " elements." << std::endl;
		for (uint meshId = 0; meshId < scene->mNumMeshes; ++meshId) {
			const aiMesh* mesh = scene->mMeshes[meshId];

			_vertices.resize(offsetVertices + mesh->mNumVertices);
			_meshIds.resize(offsetVertices + mesh->mNumVertices);
			for (uint i = 0; i < mesh->mNumVertices; ++i) {
				_vertices[offsetVertices + i] = convertVec(mesh->mVertices[i]);
				_meshIds[offsetVertices + i] = meshId;
			}

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
					_normals[offsetVertices + i] = convertVec(mesh->mNormals[i]);
				}

			}

			bool randomUV = true;

			if (mesh->HasTextureCoords(0))
			{
				_texcoords.resize(offsetVertices + mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i) {
					_texcoords[offsetVertices + i] =
						convertVec(mesh->mTextureCoords[0][i]).xy();
					if (convertVec(mesh->mTextureCoords[0][i]).xy().x() != 0.f ||
						convertVec(mesh->mTextureCoords[0][i]).xy().y() != 0.f)
					{
						randomUV = false;
					}
				}

			}

			if (randomUV) {
				SIBR_LOG << "using random UVs." << std::endl;
				_texcoords.resize(offsetVertices + mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i) {
					float u = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
					float v = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
					_texcoords[offsetVertices + i] =
						Vector2f(u * 5.f, v * 5.f);
				}
			}


			if (meshId == 0) {
				SIBR_LOG << "Mesh contains: colors: " << mesh->HasVertexColors(0)
					<< ", normals: " << mesh->HasNormals()
					<< ", texcoords: " << mesh->HasTextureCoords(0) << std::endl;
			}

			//----------------------------- FULL MATERIAL ------------------------
			uint currentMatId = matId;
			aiString aiMatName;

			if (AI_SUCCESS != scene->mMaterials[mesh->mMaterialIndex]->
				Get(AI_MATKEY_NAME, aiMatName)) {
				SIBR_LOG << "material not found " << mesh->mMaterialIndex
					<< std::endl;
			}

			std::string matName = aiMatName.C_Str();

			if (matName2Id.find(matName) == matName2Id.end()) {

				matName2Id[matName] = matId;
				_matId2Name.push_back(matName);

				matId++;
			}
			else {
				currentMatId = matName2Id[matName];
			}

			//-------------------------- END FULL MATERIAL ---------------------

			_triangles.reserve(offsetFaces + mesh->mNumFaces);
			_matIds.reserve(offsetFaces + mesh->mNumFaces); //material
			_matIdsVertices.resize(_vertices.size());
			for (uint i = 0; i < mesh->mNumFaces; ++i)
			{
				const aiFace* f = &mesh->mFaces[i];
				if (f->mNumIndices != 3)
					SIBR_WRG << "Discarding a face (not a triangle, num indices: "
					<< f->mNumIndices << ")" << std::endl;
				else
				{
					Vector3u tri = Vector3u(offsetVertices + f->mIndices[0], offsetVertices + f->mIndices[1], offsetVertices + f->mIndices[2]);
					if (tri[0] < 0 || tri[0] >= _vertices.size()
						|| tri[1] < 0 || tri[1] >= _vertices.size()
						|| tri[2] < 0 || tri[2] >= _vertices.size())
						SIBR_WRG << "Face num [" << i << "] contains invalid vertex id(s)" << std::endl;
					else {
						_triangles.push_back(tri);
						_matIds.push_back(currentMatId);  //material

					}

				}
			}

			offsetFaces = (uint)_triangles.size();
			offsetVertices = (uint)_vertices.size();

		}

		SIBR_LOG << "Mesh '" << filename << " successfully loaded. " << scene->mNumMeshes << " meshes were loaded with a total of "
			<< " (" << _triangles.size() << ") faces and "
			<< " (" << _vertices.size() << ") vertices detected." << std::endl;
		SIBR_LOG << "Init material part complete." << std::endl;

		_gl.dirtyBufferGL = true;
		return true;
	}

	sibr::Matrix4f parseTransform(const rapidxml::xml_node<>* nodeTrans)
	{
		sibr::Matrix4f objectToWorld = sibr::Matrix4f::Identity();
		if (nodeTrans)
		{
			// Helper functions
			const auto getfValue = [&](rapidxml::xml_node<>* n, std::string const& name, float& value) {
				if (n && n->first_attribute(name.c_str())) {
					value = std::stof(n->first_attribute(name.c_str())->value());
				}
			};
			const auto getVector = [&](rapidxml::xml_node<>* n, std::string const& name, sibr::Vector3f& vec) {
				const auto isNonFloat = [](const char c) {
					return c == ',' || c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '"' || c == 0;
				};
				const auto eatNonFloat = [&](const char* str, int& i) {
					while (isNonFloat(str[i])) ++i;
				};
				const auto eatFloat = [&](const char* str, int& i) {
					while (!isNonFloat(str[i])) ++i;
				};
				if (n && n->first_attribute(name.c_str())) {
					int index = 0;
					const char* value = n->first_attribute(name.c_str())->value();
					for (int i = 0; i < vec.size(); ++i) {
						eatNonFloat(value, index);
						vec[i] = std::atof(value + index);
						eatFloat(value, index);
					}
				}
			};
			const auto getAxesValues = [&](rapidxml::xml_node<>* n, float& x, float& y, float& z) {
				getfValue(n, "x", x);
				getfValue(n, "y", y);
				getfValue(n, "z", z);
			};

			// Loop through all the transform commands,
							// and "accumulate" the transform matrices into "objectToWorld"
			for (rapidxml::xml_node<>* node = nodeTrans->first_node(); node; node = node->next_sibling()) {
				sibr::Matrix4f nodeMatrix = sibr::Matrix4f::Identity();
				const char* transformType = node->name();
				if (strcmp(transformType, "matrix") == 0) {
					std::string matrixValue = node->first_attribute("value")->value();
					std::istringstream issMatrix(matrixValue);
					std::vector<std::string> split(
						std::istream_iterator<std::string>{issMatrix},
						std::istream_iterator<std::string>());
					nodeMatrix <<
						std::stof(split[0]), std::stof(split[1]), std::stof(split[2]), std::stof(split[3]),
						std::stof(split[4]), std::stof(split[5]), std::stof(split[6]), std::stof(split[7]),
						std::stof(split[8]), std::stof(split[9]), std::stof(split[10]), std::stof(split[11]),
						std::stof(split[12]), std::stof(split[13]), std::stof(split[14]), std::stof(split[15]);
				}
				else if (strcmp(transformType, "translate") == 0) {
					getAxesValues(node, nodeMatrix(0, 3), nodeMatrix(1, 3), nodeMatrix(2, 3));
				}
				else if (strcmp(transformType, "scale") == 0) {
					float scale = 1.f;
					getfValue(node, "value", scale);
					getAxesValues(node, nodeMatrix(0, 0), nodeMatrix(1, 1), nodeMatrix(2, 2));
					nodeMatrix(0, 0) *= scale;
					nodeMatrix(1, 1) *= scale;
					nodeMatrix(2, 2) *= scale;
				}
				else if (strcmp(transformType, "rotate") == 0) {
					float rotateX = 0.f, rotateY = 0.f, rotateZ = 0.f;
					float angleDegrees = 0.f;
					getAxesValues(node, rotateX, rotateY, rotateZ);
					getfValue(node, "angle", angleDegrees);
					float angleRadians = angleDegrees * (float)M_PI / 180.f;
					sibr::Transform3<float> transform;
					transform.rotate(Eigen::Quaternionf(
						Eigen::AngleAxisf(angleRadians,
							Eigen::Vector3f(rotateX, rotateY, rotateZ))));
					nodeMatrix = transform.matrix();
				}
				else if (strcmp(transformType, "lookat") == 0) {
					sibr::Vector3f eye{ 0, 0, 0 }, target{ 0, 0, 1 }, up{ 0, 1, 0 };
					getVector(node, "origin", eye);
					getVector(node, "target", target);
					getVector(node, "up", up);
					// WTF why inverse the lookat???
					nodeMatrix = sibr::lookAt(eye, target, up).inverse();
				}
				else {
					throw std::runtime_error(std::string("Mitsuba xml parser: Unknown tranform type: ") + transformType);
				}
				objectToWorld = nodeMatrix * objectToWorld;
			}
		}
		return objectToWorld;
	}

	// Extracts Name from (can accept node == nullptr)
	// <ref id="Name"/>
	std::string parseMatID(const rapidxml::xml_node<>* node)
	{
		std::string res;
		if (node)
		{
			const rapidxml::xml_attribute<>* id = node->first_attribute("id");
			res = id->value();
		}
		return res;
	}

	bool shouldFlipNormals(const rapidxml::xml_node<>* shape)
	{
		assert(shape != nullptr);
		for (const rapidxml::xml_node<>* node = shape->first_node("boolean"); node; node = node->next_sibling("boolean"))
		{
			if (node->first_attribute("name")->value() == std::string("flipNormals"))
			{
				return node->first_attribute("value")->value() == std::string("true");
			}
		}
		return false;
	}

	std::string parseFilename(const rapidxml::xml_node<>* shape)
	{
		assert(shape);
		for (const rapidxml::xml_node<>* node = shape->first_node("string"); node; node = node->next_sibling("string"))
		{
			if (node->first_attribute("name")->value() == std::string("filename"))
			{
				return node->first_attribute("value")->value();
			}
		}
		return "";
	}

	bool	MaterialMesh::loadMtsXML(const std::string& xmlFile, bool loadTextures)
	{
		srand(static_cast <unsigned> (time(0)));
		bool allLoaded = true;
		std::string pathFolder = boost::filesystem::path(xmlFile).parent_path()
			.string();
		sibr::XMLTree doc(xmlFile);
		std::map<std::string, sibr::MaterialMesh> meshes;

		struct ShapeGroup {
			struct Shape
			{
				std::string filename;
				std::string matname;
				sibr::Matrix4f toWorld = sibr::Matrix4f::Identity();
				Shape& operator=(Shape const&) = default;
				bool flipNormals = false;
			};
			std::vector<Shape> shapes;
			sibr::Matrix4f objectToWorld = sibr::Matrix4f::Identity();
		};

		std::map<std::string, ShapeGroup> idToShapegroups;

		rapidxml::xml_node<> *nodeScene = doc.first_node("scene");

		// First parse all the shapegroups
		for (rapidxml::xml_node<> *node = nodeScene->first_node("shape");
			node; node = node->next_sibling("shape"))
		{
			if (strcmp(node->first_attribute()->name(), "type") == 0 &&
				strcmp(node->first_attribute()->value(), "shapegroup") == 0) {

				//std::cout << "Found : " << node->first_attribute("id")->value() << std::endl;

				std::string id = node->first_attribute("id")->value();
				ShapeGroup shapeGroup;
				for (rapidxml::xml_node<>* shapeNode = node->first_node("shape");shapeNode;shapeNode=shapeNode->next_sibling())
				{
					ShapeGroup::Shape shape;
					shape.filename = parseFilename(shapeNode);
					shape.toWorld = parseTransform(shapeNode->first_node("transform"));
					shape.matname = parseMatID(shapeNode->first_node("ref"));
					shape.flipNormals = shouldFlipNormals(shapeNode);
					shapeGroup.shapes.push_back(shape);
				}
				rapidxml::xml_node<>* tNode = node->first_node("transform");
				shapeGroup.objectToWorld = parseTransform(tNode);
				idToShapegroups[id] = shapeGroup;
			}
		}
		// Second: Create all the actual shapes
		for (rapidxml::xml_node<> *node = nodeScene->first_node("shape");
			node; node = node->next_sibling("shape"))
		{
			for (rapidxml::xml_attribute<> *browserAttributes = node->
				first_attribute();
				browserAttributes;
				browserAttributes = browserAttributes->next_attribute()) {
				// Create the instances of the shapegroups
				if (strcmp(browserAttributes->name(), "type") == 0 &&
					strcmp(browserAttributes->value(), "instance") == 0
					) {
					rapidxml::xml_node<>* nodeRef = node->first_node("ref");
					const std::string _id = nodeRef->first_attribute("id")->value();
					SIBR_LOG << "Instancing " << _id << std::endl;
					if (idToShapegroups.find(_id) == idToShapegroups.end())
					{
						SIBR_WRG << "Could not find shapegroup " << _id << "!!!" << std::endl;
						continue;
					}
					const ShapeGroup& shapeGroup = idToShapegroups[_id];
					rapidxml::xml_node<>* nodeTrans = node->first_node("transform");
					// I am not 100% sure of the order of the matrices
					sibr::Matrix4f objectToWorld = shapeGroup.objectToWorld * parseTransform(nodeTrans);
					sibr::MaterialMesh instance;
					for (const auto& shape : shapeGroup.shapes)
					{
						std::cout << shape.filename;
						const std::string meshPath = pathFolder + "/" + shape.filename;

						if (meshes.find(meshPath) == meshes.end())
						{
							meshes[meshPath] = MaterialMesh();
							meshes[meshPath].load(meshPath);
						}

						sibr::MaterialMesh toWorldMesh = meshes[meshPath];
						
						if (shape.flipNormals && toWorldMesh.hasNormals())
						{
							const auto& refNormals = toWorldMesh.normals();
							sibr::Mesh::Normals normals(refNormals.size());
							for (int nid = 0; nid < refNormals.size(); ++nid)
							{
								normals[nid] = -refNormals[nid];
							}
							toWorldMesh.normals(normals);
						}
						// Convert to world coordinates
						sibr::Matrix4f matrix = shape.toWorld * objectToWorld;
						{
							sibr::MaterialMesh::Vertices vertices(toWorldMesh.vertices().size());
							for (int v = 0; v < toWorldMesh.vertices().size(); v++) {
								sibr::Vector4f v4(toWorldMesh.vertices()[v].x(),
									toWorldMesh.vertices()[v].y(),
									toWorldMesh.vertices()[v].z(), 1.0);
								vertices[v] = (matrix* v4).xyz();
							}

							toWorldMesh.vertices(vertices);
						}

						// If the mesh has normals, we should transform them also.
						if (toWorldMesh.hasNormals()) {
							sibr::Mesh::Normals normals(toWorldMesh.normals().size());
							const sibr::Matrix3f normalTMatrix = matrix.block(0, 0, 3, 3).inverse().transpose();
							for (int v = 0; v < toWorldMesh.normals().size(); v++) {
								const sibr::Vector3f& ln = toWorldMesh.normals()[v];
								normals[v] = (normalTMatrix * ln).xyz();
							}
							toWorldMesh.normals(normals);
						}
						
						if (!shape.matname.empty())
						{
							toWorldMesh.matId2Name({ shape.matname });
						}

						instance.merge(toWorldMesh);
					}
					merge(instance);
				}
				// Create the "unique" shapes
				else if (strcmp(browserAttributes->name(), "type") == 0 &&
					(strcmp(browserAttributes->value(), "obj") == 0 ||
						strcmp(browserAttributes->value(), "ply") == 0)) {

					rapidxml::xml_node<> *nodeRef = node->first_node("string");
					const std::string filename = nodeRef->first_attribute("value")
						->value();
					const std::string meshPath = pathFolder + "/" + filename;
					// Search for any normal options:
					rapidxml::xml_node<> *nodeOpt = node->first_node("boolean");
					bool flipNormals = false;
					while (nodeOpt) {
						const std::string name = nodeOpt->first_attribute("name")->value();
						const std::string value = nodeOpt->first_attribute("value")->value();
						if (name == "flipNormals" && value == "true") {
							flipNormals = true;
						}
						nodeOpt = nodeOpt->next_sibling("boolean");
					}


					if (meshes.find(filename) == meshes.end()) {
						meshes[filename] = sibr::MaterialMesh();
						if (!meshes[filename].load(meshPath)) {
							return false;
						}
						if (meshes[filename].matIds().empty())
						{
							SIBR_WRG << "Material (" << filename << ") not present ..." << std::endl;
						}

					}

					SIBR_LOG << "Adding one instance of: " << filename
						<< std::endl;

					rapidxml::xml_node<> *nodeRefMat = node->first_node("ref");
					if (nodeRefMat) {
						const std::string matName = nodeRefMat
							->first_attribute("id")->value();

						MatId2Name newmatIdtoName;
						newmatIdtoName.push_back(matName);
						meshes[filename].matId2Name(newmatIdtoName);
					}

					rapidxml::xml_node<> *nodeTrans = node
						->first_node("transform");

					sibr::Matrix4f objectToWorld = parseTransform(nodeTrans);

					sibr::MaterialMesh toWorldMesh = meshes[filename];

					// Apply normals transformation if needed.
					if (flipNormals && toWorldMesh.hasNormals()) {
						const auto & refNormals = toWorldMesh.normals();
						sibr::Mesh::Normals normals(refNormals.size());
						for (int nid = 0; nid < refNormals.size(); nid++) {
							normals[nid] = -refNormals[nid];
						}
						toWorldMesh.normals(normals);
					}

					if (nodeTrans) {
						// Transform the vertices position
						{
							sibr::Mesh::Vertices vertices(toWorldMesh.vertices().size());
							for (int v = 0; v < toWorldMesh.vertices().size(); v++) {
								sibr::Vector4f v4(toWorldMesh.vertices()[v].x(),
									toWorldMesh.vertices()[v].y(),
									toWorldMesh.vertices()[v].z(), 1.0);
								vertices[v] = (objectToWorld*v4).xyz();

							}
							toWorldMesh.vertices(vertices);
						}
						
						// Transform the normals too
						if (toWorldMesh.hasNormals()) {
							sibr::Mesh::Normals normals(toWorldMesh.normals().size());
							const sibr::Matrix3f normalTMatrix = objectToWorld.block(0, 0, 3, 3).inverse().transpose();
							for (int v = 0; v < toWorldMesh.normals().size(); v++) {
								const sibr::Vector3f& ln = toWorldMesh.normals()[v];
								normals[v] = (normalTMatrix * ln);
							}
							toWorldMesh.normals(normals);
						}
					}

					merge(toWorldMesh);
				}
			}
		}

		SIBR_LOG << "Loaded mesh: " << vertices().size() << " verts, " << meshIds().size() << " ids." << std::endl;
		// Load all the materials
		for (rapidxml::xml_node<> *node = nodeScene->first_node("bsdf");
			node; node = node->next_sibling("bsdf"))
		{
			//getting id 
			rapidxml::xml_attribute<> *attribute = node->first_attribute("id");
			if (attribute != nullptr) {

				std::string nameMat = attribute->value();


				// Check if a texture exists in our node with diffuse reflectance
				// If none is found, explore each BRDF until found.
				std::vector <rapidxml::xml_node<>*> queue;
				queue.push_back(node);

				bool breakBool = false;

				while (!queue.empty()) {


					//Texture Case
					for (rapidxml::xml_node<> *nodeTexture =
						queue.front()->first_node("texture");
						nodeTexture;
						nodeTexture = nodeTexture->next_sibling("texture")) {

						if (strcmp(nodeTexture->first_attribute("name")->value(),
							"diffuseReflectance") == 0 ||
							strcmp(nodeTexture->first_attribute("name")->value(),
								"reflectance") == 0 ||
							strcmp(nodeTexture->first_attribute("name")->value(),
								"specularReflectance") == 0
							)
						{
							//std::cout << "DiffuseReflectance Texture found" << std::endl;
							rapidxml::xml_node<> *firstTexture = nodeTexture->
								first_node("texture");
							if (firstTexture == nullptr) {
								firstTexture = nodeTexture;
							}
							for (rapidxml::xml_node<> *nodeString = firstTexture->
								first_node("string");
								nodeString;
								nodeString = nodeString->next_sibling("string"))
							{
								std::string textureName =
									nodeString->first_attribute("value")->value();
								sibr::ImageRGBA::Ptr texture(new sibr::ImageRGBA());
								// If we skip loading the textures, still set them as empty images.
								if (!loadTextures || texture->load(pathFolder + "/" + textureName)) {
									/*std::cout << "Diffuse " << pathFolder + "/"
										+ textureName << std::endl;*/
									_diffuseMaps[nameMat] = texture;
									breakBool = true;
									break;
								}
								else {
									SIBR_ERR << "Diffuse layer for: " <<
										nameMat << " not found" << std::endl;
								}

							}
							if (breakBool)
								break;
						}
					}


					//Color case

					if (!breakBool) {
						std::list<std::string> colorsFormatList;
						colorsFormatList.push_back("rgb");
						colorsFormatList.push_back("srgb");
						for (std::string colorsFormat : colorsFormatList)
							for (rapidxml::xml_node<> *nodeTexture =
								queue.front()->first_node(colorsFormat.c_str());
								nodeTexture;
								nodeTexture = nodeTexture->next_sibling(
									colorsFormat.c_str())) {

								if (strcmp(nodeTexture->first_attribute("name")->value(),
									"diffuseReflectance") == 0 ||
									strcmp(nodeTexture->first_attribute("name")->value(),
										"reflectance") == 0 ||
									strcmp(nodeTexture->first_attribute("name")->value(),
										"specularReflectance") == 0
									)
								{
									/*std::cout << "DiffuseReflectance Color found"
										<< std::endl;*/
									rapidxml::xml_node<> *firstTexture = nodeTexture->
										first_node(colorsFormat.c_str());
									if (firstTexture == nullptr) {
										firstTexture = nodeTexture;
									}
									std::string colorString =
										nodeTexture->first_attribute("value")->value();
									sibr::Vector3f colorMaterial;
									float redComponent, greenComponent, blueComponent;
#ifdef SIBR_OS_WINDOWS
									sscanf_s(colorString.c_str(), "%f, %f, %f",
										&redComponent, &greenComponent, &blueComponent);
#else
									sscanf(colorString.c_str(), "%f, %f, %f",
										&redComponent, &greenComponent, &blueComponent);
#endif
									const sibr::ImageRGBA::Pixel color(
										static_cast<const unsigned char>(redComponent * 255),
										static_cast<const unsigned char>(greenComponent * 255),
										static_cast<const unsigned char>(blueComponent * 255),
										255);
									sibr::ImageRGBA::Ptr texture(new sibr::ImageRGBA(
										1, 1, color));
									if (texture) {
										/*std::cout << "Diffuse color : " <<
											redComponent << ", " << blueComponent <<
											", " << greenComponent << ", " << std::endl;*/
										_diffuseMaps[nameMat] = texture;
										_tagsCoveringMaps[nameMat] = nullptr;

										breakBool = true;
										break;
									}
									else {
										SIBR_ERR << "Diffuse layer for: " << nameMat
											<< " not found" << std::endl;
									}

									if (breakBool)
										break;
								}
							}
					}

					queue.erase(queue.begin());

					for (rapidxml::xml_node<> *node = queue.front()->
						first_node("bsdf");
						node; node = node->next_sibling("bsdf"))
					{
						queue.push_back(node);
					}

				}
				if (_diffuseMaps[nameMat].get() == nullptr) {

					float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
					float g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
					float b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

					const sibr::ImageRGBA::Pixel color(
						static_cast<const unsigned char>(r * 255),
						static_cast<const unsigned char>(g * 255),
						static_cast<const unsigned char>(b * 255),
						static_cast<const unsigned char> (255)
					);
					sibr::ImageRGBA::Ptr texture(new sibr::ImageRGBA(
						1, 1, color));
					SIBR_WRG << "Warning: No color and no texture found for " << nameMat << ", " <<
						"material will be chosen randomly." << std::endl;
					_diffuseMaps[nameMat] = texture;

					_tagsCoveringMaps[nameMat] = nullptr;
					/*if (_hasTagsCoveringFile) {
						_tagsCoveringMaps[nameMat] = _listCoveringImagesTags.at(
							_tagsCoveringMaps.size() % _listCoveringImagesTags.size());
					}*/
				}



			}
		}

		bool breakBool = false;
		for (rapidxml::xml_node<> *node = nodeScene->first_node("bsdf");
			node; node = node->next_sibling("bsdf"))
		{
			if (node != nullptr && node->first_attribute("id") != nullptr)
			{
				std::string nameMat = node->first_attribute("id")->value();

				bool breakBool = false;
				for (rapidxml::xml_node<> *nodeTexture = node->
					first_node("texture");
					nodeTexture; nodeTexture = nodeTexture->
					next_sibling("texture")) {

					if (strcmp(nodeTexture->first_attribute("name")->value(),
						"opacity") == 0 &&
						nodeTexture->first_attribute("type") &&
						strcmp(nodeTexture->first_attribute("type")->value(),
							"scale") == 0) {
						//std::cout << "Found opacity mask:" << nameMat << std::endl;

						for (rapidxml::xml_node<> *nodeString = nodeTexture->
							first_node("texture")->first_node("string");
							nodeString; nodeString = nodeString->
							next_sibling("string"))
						{
							std::string textureName = nodeString->
								first_attribute("value")->value();
							sibr::ImageRGB::Ptr texture(new sibr::ImageRGB());
							if (!loadTextures || texture->load(pathFolder + "/" + textureName)) {
								_opacityMaps[nameMat] = texture;
								breakBool = true;
								break;
							}
							else {
								SIBR_ERR << "Opacity layer for: " <<
									nameMat << " not found" << std::endl;
							}


						}
						if (breakBool)
							break;
					}

				}
				if (!breakBool) {
					const sibr::ImageRGB::Pixel color(255, 255, 255);
					sibr::ImageRGB::Ptr texture(new sibr::ImageRGB(1, 1, color));
					_opacityMaps[nameMat] = texture;

				}
			}
		}

		createSubMeshes();
		return true;

	}

	void	MaterialMesh::loadCoveringTagsTexture(
		const std::vector<std::string>& listFilesTags) {

		for (const std::string filename : listFilesTags) {

			sibr::ImageRGB::Ptr textureTag(new sibr::ImageRGB());
			if (textureTag->load(filename)) {
				_listCoveringImagesTags.push_back(textureTag);
			}
			else {
				SIBR_ERR << "Diffuse layer for: " <<
					filename << " not found" << std::endl;
			}
		}
		if (_listCoveringImagesTags.size() > 0)
			_hasTagsCoveringFile = true;

		if (_hasTagsCoveringFile) {
			unsigned int counter = 0;
			for (auto it = matId2Name().begin(); it != matId2Name().end(); ++it) {
				if (_tagsCoveringMaps.find(*it) != _tagsCoveringMaps.end()) {
					_tagsCoveringMaps[*it] = _listCoveringImagesTags.at(
						counter % _listCoveringImagesTags.size());
					counter++;
				}
			}
		}

	}

	void MaterialMesh::fillColorsWithIndexMaterials(void)
	{
		sibr::Mesh::Colors colorsIdsMaterials(vertices().size());

		sibr::Mesh::Colors randomsColors;

		srand(static_cast <unsigned> (time(0)));
		for (std::string material : _matId2Name) {
			float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
			float g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
			float b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

			randomsColors.push_back(sibr::Vector3f(r, g, b));
		}

		for (unsigned int i = 0; i < _matIds.size(); i++)
		{
			colorsIdsMaterials.at(_triangles.at(i)[0]) = randomsColors.at(
				_matIds.at(i));
			colorsIdsMaterials.at(_triangles.at(i)[1]) = randomsColors.at
			(_matIds.at(i));
			colorsIdsMaterials.at(_triangles.at(i)[2]) = randomsColors.at
			(_matIds.at(i));
		}

		colors(colorsIdsMaterials);
	}


	void MaterialMesh::fillColorsWithMatIds()
	{
		sibr::Mesh::Colors colorsIdsMaterials(vertices().size());

		for (unsigned int i = 0; i < _matIds.size(); i++)
		{
			const uint matId = uint(_matIds.at(i) + 1);
			const sibr::Vector3u col = { uchar(matId & 0xff), uchar((matId >> 8) & 0xff) , uchar((matId >> 16) & 0xff) };
			const sibr::Vector3f finalCol = col.cast<float>() / 255.0f;
			colorsIdsMaterials.at(_triangles.at(i)[0]) = finalCol;
			colorsIdsMaterials.at(_triangles.at(i)[1]) = finalCol;
			colorsIdsMaterials.at(_triangles.at(i)[2]) = finalCol;
		}

		colors(colorsIdsMaterials);
	}

	Mesh MaterialMesh::generateSubMaterialMesh(int material) const
	{

		sibr::Mesh::Vertices newVertices;
		sibr::Mesh::Triangles newTriangles;

		sibr::Mesh::Colors newColors;
		sibr::Mesh::Normals newNormals;
		sibr::Mesh::UVs newTexCoords;

		std::map<int, int> mapIdVert;

		int cmptValidVert = 0;
		int cmptVert = 0;

		sibr::Mesh::Colors oldColors;
		if (hasColors())
			oldColors = colors();

		sibr::Mesh::Normals oldNormals;
		if (hasNormals())
			oldNormals = normals();

		sibr::Mesh::UVs oldTexCoords;
		if (hasTexCoords())
			oldTexCoords = texCoords();

		for (int i = 0; i < matIds().size(); i++)
		{
			if (matIds().at(i) == material) {

				uint v1, v2, v3;
				v1 = triangles().at(i)[0];
				v2 = triangles().at(i)[1];
				v3 = triangles().at(i)[2];

				auto search = mapIdVert.find(v1);
				if (search == mapIdVert.end()) {
					newVertices.push_back(vertices()[v1]);
					if (hasColors()) {
						newColors.push_back(oldColors[v1]);
					}
					if (hasNormals()) {
						newNormals.push_back(oldNormals[v1]);
					}
					if (hasTexCoords()) {
						newTexCoords.push_back(oldTexCoords[v1]);
					}
					mapIdVert[v1] = cmptValidVert;
					cmptValidVert++;
				}

				search = mapIdVert.find(v2);
				if (search == mapIdVert.end()) {
					newVertices.push_back(vertices()[v2]);
					if (hasColors()) {
						newColors.push_back(oldColors[v2]);
					}
					if (hasNormals()) {
						newNormals.push_back(oldNormals[v2]);
					}
					if (hasTexCoords()) {
						newTexCoords.push_back(oldTexCoords[v2]);
					}
					mapIdVert[v2] = cmptValidVert;
					cmptValidVert++;
				}

				search = mapIdVert.find(v3);
				if (search == mapIdVert.end()) {
					newVertices.push_back(vertices()[v3]);
					if (hasColors()) {
						newColors.push_back(oldColors[v3]);
					}
					if (hasNormals()) {
						newNormals.push_back(oldNormals[v3]);
					}
					if (hasTexCoords()) {
						newTexCoords.push_back(oldTexCoords[v3]);
					}
					mapIdVert[v3] = cmptValidVert;
					cmptValidVert++;
				}
				newTriangles.push_back(sibr::Vector3u(mapIdVert[v1], mapIdVert[v2]
					, mapIdVert[v3]));

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
			newMesh.texCoords(newTexCoords);

		return newMesh;
	}

	void	MaterialMesh::forceBufferGLUpdate(void) const
	{
		if (!_gl.bufferGL) { SIBR_ERR << "Tried to forceBufferGL on a non OpenGL Mesh" << std::endl; return; }
		_gl.dirtyBufferGL = false;
		_gl.bufferGL->build(*this);
	}

	void	MaterialMesh::freeBufferGLUpdate(void) const
	{
		_gl.dirtyBufferGL = false;
		_gl.bufferGL->free();
	}

	void	MaterialMesh::subdivideMesh2(float threshold) {

		auto areaHeronsFormula = [](sibr::Vector3f A, sibr::Vector3f B,
			sibr::Vector3f C) -> float {
			float a = distance(A, B);
			float b = distance(B, C);
			float c = distance(C, A);
			return sqrtf((a + (b + c))*(c - (a - b))*(c + (a - b))*
				(a + (b - c))) / 4.f;
		};

		bool mustChange = true;
		while (mustChange) {
			mustChange = false;
			sibr::Mesh::Colors newColors(colors());
			sibr::Mesh::Normals newNormals(normals());
			sibr::Mesh::UVs newTexCoords(texCoords());
			sibr::Mesh::Vertices newVertices(vertices());
			sibr::MaterialMesh::MeshIds newMeshIds(meshIds());

			sibr::Mesh::Triangles newTriangles;
			sibr::MaterialMesh::MatIds newMatIds;
			std::cout << triangles().size() << " triangles" << std::endl;
			for (unsigned int i = 0; i < triangles().size(); i++) {
				sibr::Vector3u t = triangles().at(i);

				int tMatId;
				if (i < matIds().size())
					tMatId = matIds().at(i);

				sibr::Vector3f a = vertices().at(t.x());
				sibr::Vector3f b = vertices().at(t.y());
				sibr::Vector3f c = vertices().at(t.z());

				if (areaHeronsFormula(a, b, c) >= (_averageArea*threshold)) {
					mustChange = true;

					sibr::Vector3f aColor, bColor, cColor, aNormal, bNormal, cNormal;
					sibr::Vector2f aTexCoords, bTexCoords, cTexCoords;

					sibr::Vector3f newColor, newNormal;
					sibr::Vector2f newTexCoord;
					sibr::Vector3f newVertex;

					if (hasColors()) {
						newColor = (colors().at(t.x()) + colors().at(t.y())
							+ colors().at(t.z())) / 3.f;
					}

					if (hasNormals()) {
						newNormal = (normals().at(t.x()) + normals().at(t.y())
							+ normals().at(t.z())) / 3.f;
					}

					if (hasTexCoords()) {
						newTexCoord = (texCoords().at(t.x()) + texCoords().at(t.y())
							+ texCoords().at(t.z())) / 3.f;
					}


					newVertex = (vertices().at(t.x()) + vertices().at(t.y())
						+ vertices().at(t.z())) / 3.f;

					newVertices.push_back(newVertex);

					if (hasColors()) {
						newColors.push_back(newColor);
					}

					if (hasNormals()) {
						newNormals.push_back(newNormal);
					}

					if (hasTexCoords()) {
						newTexCoords.push_back(newTexCoord);
					}

					if (hasMeshIds()) {
						// Pick the first referenced vertex as the provoking vertex.
						newMeshIds.push_back(meshIds().at(t.x()));
					}

					int newIndexVertex = static_cast<int> (newVertices.size()) - 1;
					newTriangles.push_back(sibr::Vector3u(t.x(),
						t.y(),
						newIndexVertex));
					newTriangles.push_back(sibr::Vector3u(t.y(),
						t.z(),
						newIndexVertex));
					newTriangles.push_back(sibr::Vector3u(t.z(),
						t.x(),
						newIndexVertex));
					if (i < matIds().size()) {
						for (unsigned int n = 0; n < 3; ++n)
							newMatIds.push_back(tMatId);
					}
				}
				else {
					newTriangles.push_back(t);
					if (i < matIds().size())
						newMatIds.push_back(tMatId);
				}
			}
			vertices(newVertices);
			colors(newColors);
			normals(newNormals);
			texCoords(newTexCoords);
			triangles(newTriangles);
			matIds(newMatIds);
			meshIds(newMeshIds);
		}
	}

	void	MaterialMesh::subdivideMesh(float threshold) {

		bool mustChange = true;
		while (mustChange) {
			mustChange = false;
			sibr::Mesh::Colors newColors(colors());
			sibr::Mesh::Normals newNormals(normals());
			sibr::Mesh::UVs newTexCoords(texCoords());
			sibr::Mesh::Vertices newVertices(vertices());
			sibr::MaterialMesh::MeshIds newMeshIds(meshIds());

			sibr::Mesh::Triangles newTriangles;
			sibr::MaterialMesh::MatIds newMatIds;


			std::cout << triangles().size() << " triangles" << std::endl;

			for (unsigned int i = 0; i < triangles().size(); i++) {
				sibr::Vector3u t = triangles().at(i);

				int tMatId;
				if (i < matIds().size())
					tMatId = matIds().at(i);

				sibr::Vector3f a = vertices().at(t.x());
				sibr::Vector3f b = vertices().at(t.y());
				sibr::Vector3f c = vertices().at(t.z());

				float localMaximum = 0.f;
				int longestSide;

				float d = distance(a, b);
				if (d > localMaximum) {
					localMaximum = d; longestSide = 0;
				}
				d = distance(b, c);
				if (d > localMaximum) {
					localMaximum = d; longestSide = 1;
				}
				d = distance(c, a);
				if (d > localMaximum) {
					localMaximum = d; longestSide = 2;
				}


				if (localMaximum >= (_averageSize*threshold)) {
					mustChange = true;

					sibr::Vector3f aColor, bColor, cColor, aNormal, bNormal,
						cNormal;
					sibr::Vector2f aTexCoords, bTexCoords, cTexCoords;

					sibr::Vector3f *v1Pos, *v2Pos, *v1Color, *v2Color,
						*v1Normal, *v2Normal;
					sibr::Vector2f *v1TexCoords, *v2TexCoords;

					sibr::Vector3f newColor, newNormal;
					sibr::Vector2f newTexCoord;
					sibr::Vector3f newVertex;

					if (hasColors()) {
						aColor = colors().at(t.x());
						bColor = colors().at(t.y());
						cColor = colors().at(t.z());
					}

					if (hasNormals()) {
						aNormal = normals().at(t.x());
						bNormal = normals().at(t.y());
						cNormal = normals().at(t.z());
					}

					if (hasTexCoords()) {
						aTexCoords = texCoords().at(t.x());
						bTexCoords = texCoords().at(t.y());
						cTexCoords = texCoords().at(t.z());
					}
					if (longestSide == 0) {
						v1Pos = &a; v2Pos = &b;
						v1Color = &aColor; v2Color = &bColor;
						v1Normal = &aNormal; v2Normal = &bNormal;
						v1TexCoords = &aTexCoords; v2TexCoords = &bTexCoords;
					}
					else if (longestSide == 1) {
						v1Pos = &b; v2Pos = &c;
						v1Color = &bColor; v2Color = &cColor;
						v1Normal = &bNormal; v2Normal = &cNormal;
						v1TexCoords = &bTexCoords; v2TexCoords = &cTexCoords;
					}
					else if (longestSide == 2) {
						v1Pos = &c; v2Pos = &a;
						v1Color = &cColor; v2Color = &aColor;
						v1Normal = &cNormal; v2Normal = &aNormal;
						v1TexCoords = &cTexCoords; v2TexCoords = &aTexCoords;
					}
					newVertex = sibr::Vector3f((v1Pos->x() + v2Pos->x()) / 2.f,
						(v1Pos->y() + v2Pos->y()) / 2.f,
						(v1Pos->z() + v2Pos->z()) / 2.f);
					newColor = sibr::Vector3f((v1Color->x() + v2Color->x()) / 2.f,
						(v1Color->y() + v2Color->y()) / 2.f,
						(v1Color->z() + v2Color->z()) / 2.f);
					newNormal = sibr::Vector3f((v1Normal->x() + v2Normal->x()) / 2.f,
						(v1Normal->y() + v2Normal->y()) / 2.f,
						(v1Normal->z() + v2Normal->z()) / 2.f);
					newTexCoord = sibr::Vector2f((v1TexCoords->x() +
						v2TexCoords->x()) / 2.f,
						(v1TexCoords->y() + v2TexCoords->y()) / 2.f);

					newVertices.push_back(newVertex);

					if (hasColors()) {
						newColors.push_back(newColor);
					}

					if (hasNormals()) {
						newNormals.push_back(newNormal);
					}

					if (hasTexCoords()) {
						newTexCoords.push_back(newTexCoord);
					}

					if (hasMeshIds()) {
						// Use the first referenced vertex as the provoking vertex.
						newMeshIds.push_back(meshIds().at(t.x()));
					}

					int newIndexVertex = static_cast<int> (newVertices.size()) - 1;
					if (i < matIds().size()) {
						newMatIds.push_back(tMatId);
						newMatIds.push_back(tMatId);
					}
					if (longestSide == 0) {
						newTriangles.push_back(sibr::Vector3u(t.x(),
							newIndexVertex,
							t.z()));
						newTriangles.push_back(sibr::Vector3u(newIndexVertex,
							t.y(),
							t.z()));
					}
					else if (longestSide == 1) {
						newTriangles.push_back(sibr::Vector3u(t.x(),
							t.y(),
							newIndexVertex));
						newTriangles.push_back(sibr::Vector3u(t.x(),
							newIndexVertex,
							t.z()));
					}
					else if (longestSide == 2) {
						newTriangles.push_back(sibr::Vector3u(t.x(),
							t.y(),
							newIndexVertex));
						newTriangles.push_back(sibr::Vector3u(newIndexVertex,
							t.y(),
							t.z()));
					}
				}
				else {
					newTriangles.push_back(t);
					if (i < matIds().size())
						newMatIds.push_back(tMatId);
				}
			}
			vertices(newVertices);
			colors(newColors);
			normals(newNormals);
			texCoords(newTexCoords);
			triangles(newTriangles);
			matIds(newMatIds);
			meshIds(newMeshIds);
		}
		// We can now subdivide the large triangles with sub-Triangles
	}

	void MaterialMesh::ambientOcclusion(const MaterialMesh::AmbientOcclusion & ao)
	{
		if (!_aoInitialized) {

			_ambientOcclusion = ao;
			colors(_aoFunction(*this, 64));
			createSubMeshes();
			float averageDistance = 0.f;
			for (sibr::Vector3u t : triangles()) {

				float maximumDistance = 0.f;

				sibr::Vector3f a = vertices().at(t.x());
				sibr::Vector3f b = vertices().at(t.y());
				sibr::Vector3f c = vertices().at(t.z());

				float d = distance(a, b);
				if (d > maximumDistance)  maximumDistance = d;
				d = distance(b, c);
				if (d > maximumDistance)  maximumDistance = d;
				d = distance(a, c);
				if (d > maximumDistance)  maximumDistance = d;

				averageDistance += maximumDistance;
			}


			const auto areaHeronsFormula = [](sibr::Vector3f A, sibr::Vector3f B,
				sibr::Vector3f C) -> float {
				float a = distance(A, B);
				float b = distance(B, C);
				float c = distance(C, A);
				return sqrtf((a + (b + c))*(c - (a - b))*(c + (a - b))*
					(a + (b - c))) / 4.f;
			};

			float averageArea = 0.f;
			for (sibr::Vector3u t : triangles()) {

				averageArea += areaHeronsFormula(vertices().at(t.x()),
					vertices().at(t.y()),
					vertices().at(t.z()));
			}



			averageDistance /= triangles().size();
			_averageSize = averageDistance;
			averageArea /= triangles().size();
			_averageArea = averageArea;
			std::cout << "Average distance SIZE = " << _averageSize << std::endl;
			std::cout << "Average distance SIZE = " << _averageArea << std::endl;
			_aoInitialized = true;
		}
		if (ao.AttenuationDistance != _ambientOcclusion.AttenuationDistance) {
			_ambientOcclusion = ao;
			colors(_aoFunction(*this, 64));
			createSubMeshes();
		}
		if (ao.SubdivideThreshold < _ambientOcclusion.SubdivideThreshold) {
			_ambientOcclusion = ao;
			subdivideMesh(_ambientOcclusion.SubdivideThreshold);
			colors(_aoFunction(*this, 64));
			createSubMeshes();
		}
		_ambientOcclusion = ao;
	}


	void	MaterialMesh::initAlbedoTextures(void) {

		//Creates textures for albedo
		if (_albedoTexturesInitialized) {
			return;
		}

		_albedoTextures.resize(matId2Name().size());
		_idTextures.resize(matId2Name().size());
		_opacityTextures.resize(matId2Name().size());
		_idTexturesOpacity.resize(matId2Name().size());
		unsigned int i = 0;
		for (auto it = matId2Name().begin();
			it != matId2Name().end();
			++it)
		{
			sibr::ImageRGBA::Ptr texturePtr = diffuseMap(*it);
			if (texturePtr) {
				_albedoTextures[i] = std::shared_ptr<sibr::Texture2DRGBA>(
					new sibr::Texture2DRGBA(*texturePtr,SIBR_GPU_LINEAR_SAMPLING));
				_idTextures[i] = _albedoTextures[i]->handle();
			}
			else {
				_albedoTextures[i] = std::shared_ptr<sibr::Texture2DRGBA>(
					new sibr::Texture2DRGBA());
				_idTextures[i] = _albedoTextures[i]->handle();
			}

			sibr::ImageRGB::Ptr texturePtrOpacity = opacityMap(*it);
			if (texturePtrOpacity && texturePtr) {
				_opacityTextures[i] = std::shared_ptr<sibr::Texture2DRGB>(
					new sibr::Texture2DRGB(*texturePtrOpacity,SIBR_GPU_LINEAR_SAMPLING));
				_idTexturesOpacity[i] = _opacityTextures[i]->handle();
			}
			else {
				_opacityTextures[i] = std::shared_ptr<sibr::Texture2DRGB>(
					new sibr::Texture2DRGB());
				_idTexturesOpacity[i] = _opacityTextures[i]->handle();
			}

			if (_hasTagsCoveringFile && _tagsCoveringMaps[*it]) {
				sibr::ImageRGB::Ptr texturePtrTag = tagsCoveringMap(*it);
				_tagsCoveringTexture[*it] = std::shared_ptr<sibr::Texture2DRGB>(
					new sibr::Texture2DRGB(*texturePtrTag,SIBR_GPU_LINEAR_SAMPLING));
				_idTagsCoveringTexture[*it] = _tagsCoveringTexture[*it]->handle();
			}

			_switchTags[*it] = false;

			i++;
		}
		if (_hasTagsFile) {
			sibr::ImageRGB::Ptr texturePtr = _tagsMap;
			_tagTexture = std::shared_ptr<sibr::Texture2DRGB>(
				new sibr::Texture2DRGB(*texturePtr,SIBR_GPU_LINEAR_SAMPLING));
			_idTagTexture = _tagTexture->handle();
		}


		_albedoTexturesInitialized = true;
	}

	void	MaterialMesh::renderAlbedo(bool depthTest, bool backFaceCulling,
		RenderMode mode, bool frontFaceCulling, bool invertDepthTest,
		bool specificMaterial, std::string nameOfSpecificMaterial
		) const
	{
		if (_subMeshes.empty()) {
			return;
		}

		unsigned int i = 0;
		bool textureFound = false;
		std::string  texName;
		for (auto it = matId2Name().begin(); it != matId2Name().end() && !textureFound; ++it) {

			if (_albedoTextures[i] != nullptr) {

				sibr::ImageRGB::Ptr coveringTagImage = tagsCoveringMap(*it);
				if (_hasTagsCoveringFile && coveringTagImage
					&& tagsCoveringMaps().find(*it) != tagsCoveringMaps().end()) {
					texName = *it;
					textureFound = true;
				}
			}
			i++;
		}

		i = 0;
		for (auto it = matId2Name().begin(); it != matId2Name().end(); ++it)
		{
			if (!specificMaterial || *it == nameOfSpecificMaterial)
				if (_albedoTextures[i] != nullptr) {
					glActiveTexture(GL_TEXTURE0);
					glBindTexture(GL_TEXTURE_2D, _idTextures[i]);

					sibr::ImageRGB::Ptr coveringTagImage = tagsCoveringMap(*it);
					if (_hasTagsCoveringFile && coveringTagImage
						&& tagsCoveringMaps().find(*it) != tagsCoveringMaps().end()) {
						glActiveTexture(GL_TEXTURE1);
						if (_switchTags.find(*it) != _switchTags.end() && _switchTags.at(*it)
							&& _hasTagsFile && _tagTexture)
							glBindTexture(GL_TEXTURE_2D, _idTagTexture);
						else
							glBindTexture(GL_TEXTURE_2D, _idTagsCoveringTexture.at(*it));
					}
					else if (_hasTagsFile && _tagTexture != nullptr) {

						glActiveTexture(GL_TEXTURE1);
						if (_switchTags.find(*it) != _switchTags.end() && _switchTags.at(*it)
							&& _idTagsCoveringTexture.size() > 0)
							glBindTexture(GL_TEXTURE_2D, _idTagsCoveringTexture.at(texName));
						else
							glBindTexture(GL_TEXTURE_2D, _idTagTexture);
					}

					glActiveTexture(GL_TEXTURE2);
					glBindTexture(GL_TEXTURE_2D, _idTexturesOpacity[i]);
					_subMeshes[i].render(depthTest, backFaceCulling, mode,
						frontFaceCulling, invertDepthTest);
				}
			i++;
		}

		
	}


	void	MaterialMesh::renderThreeSixty(bool depthTest, bool backFaceCulling,
		RenderMode mode, bool frontFaceCulling, bool invertDepthTest) const
	{

		Mesh::render(depthTest, backFaceCulling, mode, frontFaceCulling,
			invertDepthTest, true);
	}

	void	MaterialMesh::render(bool depthTest, bool backFaceCulling,
		RenderMode mode, bool frontFaceCulling, bool invertDepthTest,
		bool tessellation, bool adjacency) const
	{
		if (_typeOfRender == RenderCategory::classic)
		{
			Mesh::render(depthTest, backFaceCulling, mode, frontFaceCulling,
				invertDepthTest, adjacency);
		}
		else if (_typeOfRender == RenderCategory::diffuseMaterials)
		{
			renderAlbedo(depthTest, backFaceCulling, mode, frontFaceCulling,
				invertDepthTest);
		}
		else if (_typeOfRender == RenderCategory::threesixtyMaterials ||
			_typeOfRender == RenderCategory::threesixtyDepth)
		{
			renderThreeSixty(depthTest, backFaceCulling, mode, frontFaceCulling,
				invertDepthTest);
		}
	}

	void	MaterialMesh::merge(const MaterialMesh& other)
	{

		if (_vertices.empty())
		{
			this->operator = (other);
			return;
		}

		const size_t oldVerticesCount = vertices().size();
		const bool thisHasIds = hasMeshIds();

		sibr::Mesh::merge(other);

		uint		matIdsOffset = static_cast<unsigned int> (_matId2Name.size());
		MatIds		matIds = other.matIds();
		MatId2Name	matId2Name;

		unsigned int nbOfSimilarity = 0;
		for (unsigned int i = 0; i < other.matId2Name().size(); ++i) {
			bool foundSimilarity = false;
			unsigned int indexSimilarMaterial = 0;
			for (unsigned int j = 0; j < _matId2Name.size()
				&& !foundSimilarity; ++j) {

				if (other.matId2Name().at(i).compare(_matId2Name.at(j)) == 0) {
					//We find a similar material present on the two meshes
					//Now we modify all triangles ids corresponding to this 
					// material	
					foundSimilarity = true;
					nbOfSimilarity++;
					indexSimilarMaterial = j;
				}
			}
			if (!foundSimilarity) {
				//It's a new material.
				//We have found a new material, We will merge it in our list
				//of materials later
				matId2Name.push_back(other.matId2Name().at(i));
				//We substract the number of similarity to avoid
				//the "gap" about the materials index
				for (unsigned int j = 0; j < other.matIds().size(); ++j) {
					unsigned int id = other.matIds().at(j);
					if (id == i) {
						matIds[j] = id + matIdsOffset - nbOfSimilarity;
					}
				}
			}
			else {
				for (unsigned int j = 0; j < other.matIds().size(); ++j) {
					unsigned int id = other.matIds().at(j);
					if (id == i) {
						matIds[j] = indexSimilarMaterial;
					}
				}
			}
		}

		_matIds.insert(_matIds.end(), matIds.begin(), matIds.end());
		_matId2Name.insert(_matId2Name.end(), matId2Name.begin(),
			matId2Name.end());
		_opacityMaps.insert(other.opacityMaps().begin(),
			other.opacityMaps().end());
		_diffuseMaps.insert(other.diffuseMaps().begin(),
			other.diffuseMaps().end());

		// We have to shift all meshes ids.
		const bool otherHasIds = other.hasMeshIds();
		if (thisHasIds && otherHasIds) {
			// Shift all other IDs by _maxMeshId+1.
			_maxMeshId += 1;
			MaterialMesh::MeshIds oIds(other.meshIds());
			const int shift = int(_maxMeshId);
			for (size_t vid = 0; vid < oIds.size(); ++vid) {
				oIds[vid] = shift + oIds[vid];
			}
			_meshIds.insert(_meshIds.end(), oIds.begin(), oIds.end());
			_maxMeshId += other._maxMeshId;


		}
		else if (thisHasIds) {
			// In that case other has no IDs.
			_maxMeshId += 1;
			MaterialMesh::MeshIds newMeshIds(other.vertices().size(), int(_maxMeshId));
			_meshIds.insert(_meshIds.end(), newMeshIds.begin(), newMeshIds.end());

		}
		else if (otherHasIds) {
			// in that case give a new ID to the current mesh and insert the other IDs.
			_maxMeshId = other._maxMeshId + 1;
			_meshIds = MaterialMesh::MeshIds(oldVerticesCount, int(_maxMeshId));
			_meshIds.insert(_meshIds.end(), other.meshIds().begin(), other.meshIds().end());

		}
	}

	void	MaterialMesh::makeWhole(void)
	{
		sibr::Mesh::makeWhole();
		if (!hasMatIds()) {
			_matIds = MatIds(triangles().size(), 0);
			_matIdsVertices = MatIds(vertices().size(), 0);
			_matId2Name.push_back("emptyMat");
		}
		if (!hasMeshIds()) {
			_meshIds = MatIds(vertices().size(), 0);
			_maxMeshId = 0;
		}
	}

	void	MaterialMesh::createSubMeshes(void) {

		_subMeshes.clear();

		for (unsigned int i = 0; i < _matId2Name.size(); i++)
		{
			_subMeshes.push_back(generateSubMaterialMesh(i));
		}
	}

	sibr::MaterialMesh::Ptr MaterialMesh::invertedFacesMesh2() const
	{
		const auto invertedFacesMesh = sibr::Mesh::invertedFacesMesh2();
		auto invertedFacesMaterialMesh = std::make_shared<MaterialMesh>
			(*invertedFacesMesh);
		// If we have some mesh IDs, just clone them as-is, no need for doubling.
		if (hasMeshIds()) {
			invertedFacesMaterialMesh->meshIds(meshIds());
		}

		const int nVertices = (int)vertices().size();
		const int nTriangles = (int)triangles().size();

		Mesh::Triangles Ntriangles(2 * nTriangles);
		MaterialMesh::MatIds NmatIds(hasMatIds() ? (2 * nTriangles) : 0);

		int v_id = 0;
		sibr::Vector3u shift(nVertices, nVertices, nVertices);
		int t_id = 0;
		for (const auto & t : triangles()) {
			Ntriangles[t_id] = t;
			Ntriangles[t_id + nTriangles] = t.yxz() + shift;
			++t_id;
		}
		invertedFacesMaterialMesh->triangles(Ntriangles);

		if (hasMatIds()) {
			int m_id = 0;
			for (const auto & m : matIds()) {
				NmatIds[m_id] = m;
				NmatIds[m_id + nTriangles] = m;
				++m_id;
			}
		}

		invertedFacesMaterialMesh->matIds(NmatIds);
		invertedFacesMaterialMesh->matId2Name(_matId2Name);
		invertedFacesMaterialMesh->opacityMaps(_opacityMaps);
		invertedFacesMaterialMesh->diffuseMaps(_diffuseMaps);

		return invertedFacesMaterialMesh;
	}

	void MaterialMesh::addEnvironmentMap(float* forcedCenterX,
		float* forcedCenterY,
		float* forcedCenterZ,
		float* forcedRadius)
	{
		sibr::Vector3f center;
		float radius;
		getBoundingSphere(center, radius);

		if (forcedCenterX) center.x() = *forcedCenterX;
		if (forcedCenterY) center.y() = *forcedCenterY;
		if (forcedCenterZ) center.z() = *forcedCenterZ;
		if (forcedRadius) radius = *forcedRadius;

		//std::vector<std::string> partsOfSphere;
		std::vector<PartOfSphere> partsOfSphere = { PartOfSphere::BOTTOM, PartOfSphere::UP };
		//partsOfSphere.push_back("bottom");
		//partsOfSphere.push_back("up");

		for (PartOfSphere part : partsOfSphere) {
			std::shared_ptr<Mesh> pSphere = getEnvSphere(center, radius,
				Vector3f(0.f, 1.f, 0.f),
				Vector3f(1.f, 0.f, 0.f),
				part
			);

			sibr::MaterialMesh sphere(*pSphere);

			MatId2Name materialNames;
			MatIds materialIds;

			std::string matName;
			if (part == PartOfSphere::BOTTOM)
				matName = std::string("SibrSkyEmissivebottom");
			else
				matName = std::string("SibrSkyEmissiveup");
			materialNames.push_back(matName);

			std::vector<int> matIdsSphere;
			for (unsigned int i = 0; i < sphere.triangles().size(); ++i) {
				matIdsSphere.push_back(0);
			}
			sphere.matId2Name(materialNames);
			sphere.matIds(matIdsSphere);

			const sibr::ImageRGBA::Pixel color(0,
				255,
				255,
				255);
			sibr::ImageRGBA::Ptr textureDiffuse(new sibr::ImageRGBA(1, 1, color));
			_diffuseMaps[matName] = textureDiffuse;

			const sibr::ImageRGB::Pixel opacityAlpha(255, 255, 255);
			sibr::ImageRGB::Ptr textureOpacity(new sibr::ImageRGB(
				1, 1, opacityAlpha));
			_opacityMaps[matName] = textureOpacity;
			sphere.generateNormals();
			merge(sphere);
		}

	}



} // namespace sibr
