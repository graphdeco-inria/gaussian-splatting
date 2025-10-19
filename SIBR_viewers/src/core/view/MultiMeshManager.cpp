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


#include "MultiMeshManager.hpp"

#include <imgui/imgui.h>

namespace sibr {

	MeshData MeshData::dummy = MeshData("dummy", Mesh::Ptr(), DUMMY, Mesh::FillRenderMode);

	MeshData::MeshData(const std::string & _name, Mesh::Ptr mesh_ptr, MeshType mType, Mesh::RenderMode render_mode) :
		meshPtr(mesh_ptr), renderMode(render_mode), name(_name), meshType(mType)
	{
		if (mType == POINTS) {
			renderMode = Mesh::PointRenderMode;
		}
		if (mType == LINES && renderMode == Mesh::FillRenderMode) {
			renderMode = Mesh::LineRenderMode;
		}
		if (renderMode != Mesh::FillRenderMode) {
			backFaceCulling = false;
		}
	}

	MeshData MeshData::getNormalsMeshData() const
	{
		MeshData data(name + "_normals", meshPtr, normalMode == PER_TRIANGLE ? TRIANGLES : POINTS);
		data.setColor(normalsColor).setDepthTest(depthTest);
		data.normalsLength = (normalsInverted ? -normalsLength : normalsLength);
		return data;
	}

	MeshData::operator bool() const
	{
		return meshType != DUMMY;
	}

	void MeshData::renderGeometry() const
	{
		CHECK_GL_ERROR;
		if (!meshPtr) {
			return;
		}
		if (renderMode == Mesh::PointRenderMode) {
			meshPtr->render_points(depthTest);
		} else {
			meshPtr->render(depthTest, backFaceCulling, renderMode, frontFaceCulling, invertDepthTest);
		}
		CHECK_GL_ERROR;
	}

	void MeshData::onGUI(const std::string & name)
	{
		// rendering mode
		static const std::string renderModeStrs[3] = { "Points",  "Lines",  "Fill" };

		if (ImGui::BeginCombo(("##render_mode_" + name).c_str(), renderModeStrs[(int)renderMode].data())) {
			for (int t = (int)meshType; t >= 0; --t) {
				if (ImGui::Selectable(renderModeStrs[t].data(), t == (int)renderMode)) {
					renderMode = (Mesh::RenderMode)t;
				}			
			}	
			ImGui::EndCombo();
		}
		ImGui::NextColumn();

		//alpha
		ImGui::SliderFloat(("##alpha_" + name).c_str(), &alpha, 0, 1);
		ImGui::NextColumn();

		//user color
		static const std::string colorModeStrs[2] = { "User-defined",  "Vertex" };
		if (ImGui::BeginCombo(("##color_mode_" + name).c_str(), colorModeStrs[(int)colorMode].data())) {	
			if (meshPtr && meshPtr->hasColors()) {
				if (ImGui::Selectable(colorModeStrs[VERTEX].data(), colorMode == VERTEX)) {
					colorMode = VERTEX;
				}
			}
			if (ImGui::Selectable(colorModeStrs[USER_DEFINED].data(), colorMode == USER_DEFINED)) {
				colorMode = USER_DEFINED;
			}
			ImGui::EndCombo();
		}
		if (colorMode == USER_DEFINED) {
			ImGui::SameLine();
			ImGui::ColorEdit3(("##color_picker_" + name).c_str(), &userColor[0], ImGuiColorEditFlags_NoInputs);
		} 
		ImGui::NextColumn();

		//rendering options
		if (ImGui::ArrowButton(("##OptionsArrow" + name).c_str(), ImGuiDir_Down)) {
			ImGui::OpenPopup(("##Options_popup_" + name).c_str());
		}	
		if (ImGui::BeginPopup(("##Options_popup_" + name).c_str())) {
			ImGui::Checkbox(("Depth Test##" + name).c_str(), &depthTest);
			if (meshType == TRIANGLES) {
				ImGui::Checkbox(("Cull faces##" + name).c_str(), &backFaceCulling);
				ImGui::Checkbox(("Swap back/front##" + name).c_str(), &frontFaceCulling);
			}
			if (renderMode == Mesh::PointRenderMode) {
				ImGui::PushItemWidth(75);
				ImGui::SliderInt(("PointSize##" + name).c_str(), &radius, 1, 50);
				ImGui::PopItemWidth();
			}
			if (meshType == TRIANGLES) {
				ImGui::Separator();
				ImGui::Checkbox(("ShowNormals##" + name).c_str(), &showNormals);
				if (showNormals) {		
					static const std::string normalModeStrs[2] = { "Per-triangle", "Per-vertex"};
					if (ImGui::BeginCombo(("##normal_mode_" + name).c_str(), normalModeStrs[(int)normalMode].data())) {
						if (ImGui::Selectable(normalModeStrs[PER_TRIANGLE].data(), normalMode == PER_TRIANGLE)) {
							normalMode = PER_TRIANGLE;
						}
						if (meshPtr && meshPtr->hasNormals()) {
							if (ImGui::Selectable(normalModeStrs[PER_VERTEX].data(), normalMode == PER_VERTEX)) {
								normalMode = PER_VERTEX;
							}
						}
						ImGui::EndCombo();
					}
					ImGui::Checkbox(("NormalInverted##" + name).c_str(), &normalsInverted);
					ImGui::PushItemWidth(90);
					ImGui::SliderFloat(("NormalSize##" + name).c_str(), &normalsLength, 0.001f, 10.0f, "%.3f", 3.0f);
					ImGui::PopItemWidth();
					ImGui::ColorEdit3(("NormalsColor##color_picker_" + name).c_str(), &normalsColor[0], ImGuiColorEditFlags_NoInputs);
				}
				if (!meshPtr->hasNormals()) {
					if (ImGui::Button(("Compute Normals##" + name).c_str()) ){
						meshPtr->generateNormals();
					}
				} else {
					ImGui::Checkbox(("Phong shading##" + name).c_str(), &phongShading);
				}
				ImGui::Separator();
			}
			ImGui::EndPopup();
		}
		ImGui::NextColumn();
	}

	std::string MeshData::getInfos() const
	{
		if (!meshPtr) {
			return "no mesh";
		}

		std::stringstream s;
		s << meshPtr->vertices().size() << " vertices \n" <<
			meshPtr->triangles().size() << " triangles \n" <<
			"hasNormals() : " << meshPtr->hasNormals() << "\n" <<
			"hasColors() : " << meshPtr->hasColors() << "\n" <<
			"hasTexCoords() : " << meshPtr->hasTexCoords() << "\n"
			;

		return s.str();
	}

	MeshData & MeshData::setColor(const Vector3f & col)
	{
		userColor = col;
		return *this;
	}

	MeshData & MeshData::setBackFace(bool backface)
	{
		backFaceCulling = backface;
		return *this;
	}

	MeshData & MeshData::setDepthTest(bool depth_test)
	{
		depthTest = depth_test;
		return *this;
	}

	MeshData & MeshData::setColorRandom()
	{
		static const auto baseHash = [](uint p) {
			p = 1103515245U * ((p >> 1U) ^ (p));
			uint h32 = 1103515245U * ((p) ^ (p >> 3U));
			return h32 ^ (h32 >> 16);
		};

		static const uint mask = 0x7fffffffU;

		static int seed_x = 0;
		
		++seed_x;
		
		uint n = baseHash(uint(seed_x));
		Vector3u tmp = Vector3u(n, n * 16807U, n * 48271U);
		for (int c = 0; c < 3; ++c) {
			userColor[c] = (tmp[c] & mask) / float(0x7fffffff);
 		}
		 
		 return *this;
	}

	MeshData & MeshData::setRadiusPoint(int rad)
	{
		radius = rad;
		return *this;
	}

	MeshData& MeshData::setScale(float s)
	{
		// TODO: insérer une instruction return ici
		scale = s;
		return *this;
	}

	MeshData& MeshData::setTransformation(sibr::Matrix4f& tr)
	{
		// TODO: insérer une instruction return ici
		transformation = tr;
		return *this;
	}

	MeshData & MeshData::setAlpha(float _alpha) {
		alpha = _alpha;
		return *this;
	}

	MeshData & MeshData::setColorMode(ColorMode mode)
	{
		colorMode = mode;
		return *this;
	}

	void ShaderAlphaMVP::initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom)
	{
		shader.init(name, vert, frag, geom);
		mvp.init(shader, "mvp");	
		alpha.init(shader, "alpha");
	}

	void ShaderAlphaMVP::setUniforms(const Camera & eye, const MeshData & data)
	{
		mvp.set(eye.viewproj()*data.transformation);
		alpha.set(data.alpha);
	}

	void ShaderAlphaMVP::render(const Camera & eye, const MeshData & data)
	{
		shader.begin();

		setUniforms(eye, data);

		data.renderGeometry();

		shader.end();
	}

	void ColorMeshShader::initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom)
	{
		ShaderAlphaMVP::initShader(name, vert, frag, geom);
		user_color.init(shader, "user_color");
	}

	void ColorMeshShader::setUniforms(const Camera & eye, const MeshData & data)
	{
		ShaderAlphaMVP::setUniforms(eye, data);
		user_color.set(data.userColor);
	}

	void PointShader::initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom)
	{
		ColorMeshShader::initShader(name, vert, frag, geom);
		radius.init(shader, "radius");
	}

	void PointShader::setUniforms(const Camera & eye, const MeshData & data)
	{
		ColorMeshShader::setUniforms(eye, data);
		radius.set(data.radius);
	}

	void PointShader::render(const Camera & eye, const MeshData & data)
	{
		glEnable(GL_PROGRAM_POINT_SIZE);
		ColorMeshShader::render(eye, data);
		glDisable(GL_PROGRAM_POINT_SIZE);
	}

	void NormalRenderingShader::initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom)
	{
		ColorMeshShader::initShader(name, vert, frag, geom);
		normals_size.init(shader, "normals_size");
	}

	void NormalRenderingShader::setUniforms(const Camera & eye, const MeshData & data)
	{
		ColorMeshShader::setUniforms(eye, data);
		normals_size.set(data.normalsLength);
	}

	void MeshShadingShader::initShader(const std::string & name, const std::string & vert, const std::string & frag, const std::string & geom)
	{
		ColorMeshShader::initShader(name, vert, frag, geom);
		light_position.init(shader, "light_position");
		phong_shading.init(shader, "phong_shading");
		use_mesh_color.init(shader, "use_mesh_color");
	}

	void MeshShadingShader::setUniforms(const Camera & eye, const MeshData & data)
	{
		ColorMeshShader::setUniforms(eye, data);
		light_position.set(eye.position());
		phong_shading.set(data.phongShading);
		use_mesh_color.set(data.colorMode == MeshData::ColorMode::VERTEX);
	}

	MultiMeshManager::MultiMeshManager(const std::string & _name) : name(_name)
	{
		initShaders();

		auto cube = Mesh::getTestCube();
		TrackBall tb;
		tb.fromMesh(*cube, Viewport(0,0,1600,1200));
		camera_handler.fromCamera(tb.getCamera());

		camera_handler.switchMode(InteractiveCameraHandler::InteractionMode::TRACKBALL);
	}

	void MultiMeshManager::onUpdate(Input & input, const Viewport & vp)
	{
		if (!camera_handler.isSetup() && list_meshes.size() > 0) {
			for (const auto & mesh_data : list_meshes) {
				if (mesh_data.raycaster && mesh_data.meshPtr) {
					auto bbox = mesh_data.meshPtr->getBoundingBox();
					if (bbox.volume() > 0) {
						camera_handler.getRaycaster() = mesh_data.raycaster;
						TrackBall tb;
						tb.fromBoundingBox(bbox, vp);
						camera_handler.fromCamera(tb.getCamera());
						break;
					}
				}
			}
		}

		if (camera_handler.isSetup() && !camera_handler.getRaycaster()) {
			for (const auto & mesh : list_meshes) {
				if (mesh.raycaster) {
					camera_handler.getRaycaster() = mesh.raycaster;
					break;
				}
			}
		}

		if (selected_mesh_it_is_valid) {
			const auto & mesh = *selected_mesh_it;		
			if( mesh.raycaster) {
				camera_handler.getRaycaster() = mesh.raycaster;
			}
		}

		camera_handler.update(input, 1 / 60.0f, vp);
	}

	void MultiMeshManager::onRender(const Viewport & viewport)
	{
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Multimesh manager");

		viewport.clear(backgroundColor);
		viewport.bind();
		renderMeshes();
		camera_handler.onRender(viewport);

		glPopDebugGroup();
	}

	void MultiMeshManager::onRender(IRenderTarget & dst)
	{
		dst.bind();

		const Viewport vp(0.0f, 0.0f, (float)dst.w(), (float)dst.h());
		onRender(vp);

		dst.unbind();
	}

	void MultiMeshManager::onGUI()
	{
		if (ImGui::Begin(name.c_str())) {
			ImGui::Separator();

			list_mesh_onGUI();
	
		}
		ImGui::End();
	}

	void MultiMeshManager::removeMesh(const std::string & name)
	{
		for (auto it = list_meshes.begin(); it != list_meshes.end(); ++it) {
			if (it->name == name) {
				list_meshes.erase(it);
				return;
			}
		}
	}

	void MultiMeshManager::setIntialView(const std::string& dataset_path)
	{
		const std::string topViewPath = dataset_path + "/cameras/topview.txt";
		std::ifstream topViewFile(topViewPath);
		if (topViewFile.good())
		{
			SIBR_LOG << "Loaded saved topview (" << topViewPath << ")." << std::endl;
			// Intialize a temp camera (used to load the saved top view pose) with
			// the current top view camera to get the resolution/fov right.
			InputCamera cam(camera_handler.getCamera());
			cam.readFromFile(topViewFile);
			// Apply it to the top view FPS camera.
			//camera_handler.fromCamera(cam, false);
			camera_handler.fromTransform(cam.transform(), false, true);
		}
	}

	void MultiMeshManager::initShaders()
	{
		const std::string folder = sibr::getShadersDirectory("core") + "/";

		colored_mesh_shader.initShader("colored_mesh_shader",
			loadFile(folder + "alpha_colored_mesh.vert"), 
			loadFile(folder + "alpha_colored_mesh.frag")
		);
		points_shader.initShader("points_shader",
			loadFile(folder + "alpha_points.vert"),
			loadFile(folder + "alpha_points.frag")
		);
		per_vertex_normals_shader.initShader("per_vertex_normal_shader",
			loadFile(folder + "alpha_colored_per_vertex_normals.vert"),
			loadFile(folder + "alpha_colored_mesh.frag"),
			loadFile(folder + "alpha_colored_per_vertex_normals.geom")
		);
		per_triangle_normals_shader.initShader("per_triangle_normal_shader",
			loadFile(folder + "alpha_colored_per_triangle_normals.vert"),
			loadFile(folder + "alpha_colored_mesh.frag"),
			loadFile(folder + "alpha_colored_per_triangle_normals.geom")
		);
	}

	void MultiMeshManager::renderMeshes()
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);

		for (const auto & mesh_data : list_meshes) {
			if (!mesh_data.active) {
				continue;
			}

			if (mesh_data.renderMode == Mesh::PointRenderMode) {
				points_shader.render(camera_handler.getCamera(), mesh_data);
			} else {
				colored_mesh_shader.render(camera_handler.getCamera(), mesh_data);
			}

			if (mesh_data.showNormals) {
				if (mesh_data.normalMode == MeshData::PER_VERTEX ) {
					per_vertex_normals_shader.render(camera_handler.getCamera(), mesh_data.getNormalsMeshData());
				} else {
					per_triangle_normals_shader.render(camera_handler.getCamera(), mesh_data.getNormalsMeshData());
				}
				
			}
		}

		glDisable(GL_BLEND);
	}

	void MultiMeshManager::list_mesh_onGUI()
	{
		Iterator swap_it_src, swap_it_dst;
		bool do_swap = false;
		static int num_swap = 1;
		
		if (ImGui::CollapsingHeader(("Meshes list##" + name).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {

			static char loaded_mesh_str[128] = "";
			static std::string loaded_mesh_path;
			static int loaded_mesh_counter = 0;

			if(ImGui::Button("load Mesh##MeshesList") && showFilePicker(loaded_mesh_path, FilePickerMode::Default, "", "obj,ply")) {
				Mesh::Ptr mesh = std::make_shared<Mesh>();			

				if (mesh->load(loaded_mesh_path)) {			
					Path mesh_path = loaded_mesh_path;			
					std::string mesh_name = loaded_mesh_str;
					mesh_name = (mesh_name == "") ? mesh_path.stem().string() : loaded_mesh_str;

					for (const auto & mesh_it : list_meshes) {
						if (mesh_name == mesh_it.name) {
							mesh_name += std::to_string(loaded_mesh_counter);
							break;
						}
					}

					addMesh(mesh_name, mesh);
					++loaded_mesh_counter;
				}
			}

			ImGui::SameLine();
			ImGui::InputText("mesh name##MeshesList", loaded_mesh_str, IM_ARRAYSIZE(loaded_mesh_str));

			// 0 name | 1 snapto delete | 2 active | 3 rendering mode | 4 alpha | 5 color | 6 Options
			ImGui::Columns(7, "mesh options");

			//ImGui::SetColumnWidth(4, 50);

			ImGui::Separator();
			if (ImGui::Button("Mesh##MeshesList")) {
				list_meshes.reverse();
			}
			ImGui::NextColumn();

			ImGui::NextColumn();

			if (ImGui::Button("Active##MeshesList")) {
				for (auto & mesh : list_meshes) {
					mesh.active = !mesh.active;
				}
			}
			ImGui::SameLine();
			if (ImGui::Button("All##MeshesList")) {
				for (auto & mesh : list_meshes) {
					mesh.active = true;
				}
			}
			ImGui::NextColumn();

			ImGui::Text("Mode");
			ImGui::NextColumn();

			static bool full_alpha = false;
			if (ImGui::Button("Alpha##MeshesList")) {
				for (auto & mesh : list_meshes) {
					if (mesh.active) {
						mesh.alpha = full_alpha ? 1.0f : 0.0f;
					}
				}
				full_alpha = !full_alpha;
			}
			ImGui::NextColumn();

			ImGui::Text("Color"); 
			ImGui::SameLine();
			ImGui::ColorEdit3(("Background##" + name).c_str(), &backgroundColor[0], ImGuiColorEditFlags_NoInputs);
			ImGui::NextColumn();

			ImGui::Text("Options");
			ImGui::NextColumn();

			ImGui::Separator();

			selected_mesh_it_is_valid = false;
			for (auto mesh_it = list_meshes.begin(); mesh_it != list_meshes.end(); ++mesh_it) {
				auto & mesh = *mesh_it;
				if (ImGui::Selectable(mesh.name.c_str(), (selected_mesh_it_is_valid && mesh_it == selected_mesh_it))) {
					selected_mesh_it = mesh_it;
					selected_mesh_it_is_valid = true;
				}
				if (ImGui::IsItemActive()) {
					float threshold = ImGui::GetItemRectSize().y + 5.0f;
					ImVec2 value_raw = ImGui::GetMouseDragDelta(0, 0.0f);

					if (value_raw.y > threshold * num_swap) {
						swap_it_dst = swap_it_src = mesh_it;
						++swap_it_dst;
						if (swap_it_dst != list_meshes.end()) {
							do_swap = true;
						}
					} else if (value_raw.y < -threshold * num_swap) {
						swap_it_dst = swap_it_src = mesh_it;
						--swap_it_dst;
						if (swap_it_src != list_meshes.begin()) {
							do_swap = true;
						}
					}
				}
				if (ImGui::IsItemHovered()) {
					ImGui::BeginTooltip();
					ImGui::Text(mesh.getInfos().c_str());
					ImGui::EndTooltip();
				}
				ImGui::NextColumn();

				if (ImGui::Button(("SnapTo##" + mesh_it->name).c_str()) && mesh_it->meshPtr) {
					auto box = mesh_it->meshPtr->getBoundingBox();
					if ((box.diagonal().array() > 1e-6f ).all()) {
						InputCamera cam = camera_handler.getCamera();
						cam.setLookAt(box.center() + 2.0f*box.diagonal(), box.center(), { 0,1,0 });
						camera_handler.fromCamera(cam);
					}			
				}
				ImGui::SameLine();
				if (ImGui::Button(("X##" + mesh_it->name).c_str())) {
					removeMesh(mesh_it->name);
				}
				ImGui::NextColumn();
				
				ImGui::Checkbox(("##active_" + mesh.name).c_str(), &mesh.active);
				ImGui::SameLine();
				if (ImGui::Button(("OnlyMe##" + mesh.name).c_str())) {
					for (auto other_it = list_meshes.begin(); other_it != list_meshes.end(); ++other_it) {
						other_it->active = (other_it == mesh_it);
					}
				}

				
				ImGui::NextColumn();

				mesh.onGUI(mesh.name);
				ImGui::Separator();
			}

			ImGui::Columns(1);
		}
		if (do_swap) {
			std::swap(*swap_it_src, *swap_it_dst);
			++num_swap;
		}
		if (ImGui::IsMouseReleased(0)) {
			num_swap = 1;
		}
	}

	MeshData & MultiMeshManager::addMesh(const std::string & name, Mesh::Ptr mesh, bool use_raycaster)
	{
		if (!mesh) {
			SIBR_WRG << "no mesh ptr in " << name;
			return MeshData::dummy;
		}

		return addMesh(name, mesh, 0, use_raycaster);
	}

	MeshData & MultiMeshManager::addMesh(const std::string & name, Mesh::Ptr mesh, Raycaster::Ptr raycaster, bool create_raycaster)
	{
		if (!mesh) {
			SIBR_WRG << "no mesh ptr in " << name;
			return MeshData::dummy;
		}

		MeshData data(name, mesh, MeshData::TRIANGLES, Mesh::FillRenderMode);
		data.colorMode = mesh->hasColors() ? MeshData::ColorMode::VERTEX : MeshData::ColorMode::USER_DEFINED;
		data.normalMode = (mesh->hasNormals() ? MeshData::PER_VERTEX : MeshData::PER_TRIANGLE);
		data.phongShading = mesh->hasNormals();
		data.raycaster = raycaster;

		return addMeshData(data, create_raycaster).setColorRandom();
	}

	MeshData & MultiMeshManager::addMeshAsLines(const std::string & name, Mesh::Ptr mesh)
	{
		if (!mesh) {
			SIBR_WRG << "no mesh ptr in " << name;
			return MeshData::dummy;
		}

		MeshData data(name, mesh, MeshData::LINES, Mesh::LineRenderMode);
		return addMeshData(data).setColorRandom().setDepthTest(false);
	}

	MeshData & MultiMeshManager::addLines(const std::string & name, const std::vector<Vector3f>& endPoints, const Vector3f & color)
	{
		Mesh::Triangles tris(endPoints.size() / 2);
		for (uint t = 0; t < tris.size(); ++t) {
			tris[t] = Vector3u(2 * t, 2 * t, 2 * t + 1);
		}

		Mesh::Ptr mesh = std::make_shared<Mesh>();
		mesh->vertices(endPoints);
		mesh->triangles(tris);

		MeshData data(name, mesh, MeshData::LINES, Mesh::LineRenderMode);
		data.userColor = color;
		data.depthTest = false;

		return addMeshData(data).setColorMode(MeshData::USER_DEFINED);
	}

	MeshData & MultiMeshManager::addPoints(const std::string & name, const std::vector<Vector3f>& points, const Vector3f & color)
	{
		Mesh::Vertices vertices(points);

		Mesh::Ptr mesh = std::make_shared<Mesh>();
		mesh->vertices(vertices);

		MeshData data(name, mesh, MeshData::POINTS, Mesh::PointRenderMode);
		data.userColor = color;
		data.depthTest = false;

		return addMeshData(data).setColorMode(MeshData::USER_DEFINED);
	}

	MeshData & MultiMeshManager::getMeshData(const std::string & name)
	{
		for (auto & m : list_meshes) {
			if (m.name == name ) {
				return m;
			}
		}
		return MeshData::dummy;
	}

	MeshData & MultiMeshManager::addMeshData(MeshData & data, bool create_raycaster)
	{
		bool collision = false;
		Iterator collision_it;
		for (collision_it = list_meshes.begin(); collision_it != list_meshes.end(); ++collision_it) {
			if (collision_it->name == data.name) {
				collision = true;
				break;
			}
		}

		if (collision) {
			collision_it->meshPtr = data.meshPtr;
			return MeshData::dummy;
		} else {
			Raycaster::Ptr raycaster;
			if (create_raycaster) {
				raycaster = std::make_shared<Raycaster>();
				raycaster->init();
				raycaster->addMesh(*data.meshPtr);
			}
			data.raycaster = raycaster;

			list_meshes.push_back(data);


			auto box = data.meshPtr->getBoundingBox();
			if (!box.isEmpty()) {
				InputCamera cam = camera_handler.getCamera();
				cam.zfar(std::max(cam.zfar(), 5.0f*box.diagonal().norm()));
				camera_handler.fromCamera(cam);
			}

			return list_meshes.back();
		}
	}

}
