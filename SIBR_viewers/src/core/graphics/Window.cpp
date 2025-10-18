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



#include "core/graphics/Input.hpp"
#include "core/graphics/Window.hpp"
#include "core/graphics/RenderUtility.hpp"

#include "imgui/imgui.cpp" // needed for loading ini settings
#include "imgui/imgui.h"
#include "imgui_impl_glfw_gl3.h"

#include <regex>

namespace sibr
{
	int Window::contextId = -1;

	static void glfwErrorCallback(int error, const char* description)
	{
		SIBR_ERR << description << std::endl;
	}

	static void glErrorCallback(GLenum src, GLenum type, GLuint id, GLenum severity, GLsizei size, const GLchar* str, const void* user) {
		// For now we only log errors, and we ignore severity.
		if(type != GL_DEBUG_TYPE_ERROR) {
			//SIBR_LOG << "[API]" << "(" << src << "," << type << "," << id << "," << severity << "): " << std::string(str, size) << std::endl;
			return;
		}
		std::string errStr;
		switch(src) {
		case GL_DEBUG_SOURCE_API:
			errStr = "[API] ";
			break;
		case GL_DEBUG_SOURCE_SHADER_COMPILER:
			errStr = "[Shader] ";
			break;
		case GL_DEBUG_SOURCE_THIRD_PARTY:
			errStr = "[3rd party] ";
			break;
		case GL_DEBUG_SOURCE_APPLICATION:
			errStr = "[Application] ";
			break;
		case GL_DEBUG_SOURCE_OTHER:
			errStr = "[Other] ";
			break;
		default:
			break;
		}

		const std::string errStr2(str, size);

		SIBR_ERR << "OpenGL: " << errStr << errStr2 << std::endl;
	
	}

	static void glfwKeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		key = std::max(0, key);

		// We only pass the key input to our code if the interface isn't currently using it.
		if (!ImGui::GetIO().WantCaptureKeyboard) {
			if (action == GLFW_PRESS) {
				sibr::Input::global().key().press((sibr::Key::Code)key);
			} else if (action == GLFW_RELEASE) {
				sibr::Input::global().key().release((sibr::Key::Code)key);
			}
		} else {
			sibr::Input::global() = sibr::Input();
		}
		ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
	}

	static void glfwResizeCallback(GLFWwindow* window, int w, int h)
	{
		void* userptr = glfwGetWindowUserPointer(window);
		Window* win = reinterpret_cast<Window*>(userptr);
		
		// TT : should be the right thing to do, but might break some old stuff
		win->viewport(Viewport(0.f, 0.f, (float)(w), (float)(h)));
	}

	static void glfwCursorPosCallback(GLFWwindow* /*window*/, double x, double y)
	{
		// We  pass the mouse position to our code iff the interface doesn't need it.
		if (!ImGui::GetIO().WantCaptureMouse) {
			sibr::Input::global().mousePosition(Vector2i((int)x, (int)y));
		} else {
			sibr::Input::global() = sibr::Input();
		}
		
	}

	static void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		// We only pass the mouse input to our code if the interface isn't currently using it.
		if (!ImGui::GetIO().WantCaptureMouse) {
			if (action == GLFW_PRESS) {
				sibr::Input::global().mouseButton().press((sibr::Mouse::Code)button);
			} else if (action == GLFW_RELEASE) {
				sibr::Input::global().mouseButton().release((sibr::Mouse::Code)button);
			}
		} else {
			// We have to pass release events in the case where we pressed while inside our views, and released outside.
			if(sibr::Input::global().mouseButton().isActivated((sibr::Mouse::Code)button)) {
				sibr::Input::global().mouseButton().release((sibr::Mouse::Code)button);
			}
		}
		
		ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
	}

	static void glfwMouseScrollCallback(GLFWwindow* window, double x, double y)
	{
		sibr::Input::global().mouseScroll(y);
		ImGui_ImplGlfw_ScrollCallback(window, x, y);
	}
	///////////////////////////////////////////////////////////////////////////

	static int windowCounter = 0;

	/*static*/ bool			Window::contextIsRunning( void )
	{
		return windowCounter > 0;
	}

	Window::AutoInitializer::AutoInitializer( const WindowArgs & args ) : _useGUI(!args.no_gui && !args.offscreen)
	{
		if (windowCounter == 0)
		{
			SIBR_LOG << "Initialization of GLFW" << std::endl;
			glfwSetErrorCallback(glfwErrorCallback);

			if (!glfwInit())
				SIBR_ERR << "cannot init glfw" << std::endl;
			if (!args.offscreen)
				sibr::Input::global().key().clearStates();

		}
		++windowCounter;
	}

	Window::AutoInitializer::~AutoInitializer( void )
	{
		--windowCounter;
		if (windowCounter == 0)
		{
			if(_useGUI) {
				ImGui_ImplGlfwGL3_Shutdown();	/// \todo TODO: not sure if safe with multi-context
				ImGui::DestroyContext();
			}
			glfwSetErrorCallback(nullptr);
			SIBR_LOG << "Deinitialization of GLFW" << std::endl;
			glfwTerminate();
		}
	}

	Window::Window(uint w, uint h, const std::string& title, const WindowArgs & args, const std::string& defaultSettingsFilename) 
		: _hiddenInit(args), _useGUI(!args.no_gui && !args.offscreen), _shouldClose(false) 
	{
		
		setup(w, h, title, args, defaultSettingsFilename);

		if (!(args.fullscreen)) {
			glfwSetWindowPos(_glfwWin.get(), 200, 200);
		}
	}

	Window::Window(const std::string& title, const WindowArgs & args, const std::string& defaultSettingsFilename)
		: Window(args.win_width, args.win_height, title, args, defaultSettingsFilename)
	{
	}

	Window::Window(const std::string& title, const sibr::Vector2i & margins, const WindowArgs & args, const std::string& defaultSettingsFilename)
		: _hiddenInit(args), _useGUI(!args.no_gui && !args.offscreen), _shouldClose(false)
	{
		sibr::Vector2i winSize;
		if (args.offscreen) {
			winSize = sibr::Vector2i(args.win_width, args.win_height);
		}
		else {
			winSize = desktopSize();
		}
		
		// Here autoInitializer is already initialized, thus glfwInit() has been called
		setup(winSize.x() - 2*margins.x(), winSize.y() - 2*margins.y(), title, args, defaultSettingsFilename);

		if (!(args.fullscreen)) {
			glfwSetWindowPos(_glfwWin.get(), margins.x(), margins.y());
		}

	}

	void Window::swapBuffer(void) {
		if (_useGUI) {
			glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "ImGui interface");
			ImGui::Render();
			ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
			glPopDebugGroup();
		}
		glfwSwapBuffers(_glfwWin.get());
		// Keep the call below in all cases to avoid accumulating all interfaces in one frame.
		if (_useGUI)
			ImGui_ImplGlfwGL3_NewFrame();
	}
	
	void Window::resetSettingsToDefault() {
		std::string iniFilename = ImGui::GetIO().IniFilename;
		if(iniFilename != "" && fileExists(iniFilename)) {
			if(remove(iniFilename.c_str()))
				SIBR_WRG << "Settings file " << iniFilename << " was not removed due to an error." << std::endl;
			else
				SIBR_LOG << "Settings file " << iniFilename << " was removed successfully." << std::endl;
		}
		else {
			SIBR_WRG << "Settings file " << iniFilename << " not found." << std::endl;
		}

		ImGuiContext& g = *ImGui::GetCurrentContext();

		for (int i = 0; i < g.SettingsWindows.Size; i++)
				IM_DELETE(g.SettingsWindows[i].Name);

		g.SettingsWindows.clear();

		loadSettings();

		for (ImGuiWindow * window: g.Windows) {
			if (ImGuiWindowSettings* settings = ImGui::FindWindowSettings(window->ID)) {
				SetWindowPos(window, settings->Pos, ImGuiCond_Always);
				SetWindowSize(window, settings->Size, ImGuiCond_Always);
				SetWindowCollapsed(window, settings->Collapsed, ImGuiCond_Always);
			}
		}
	}

	void Window::loadSettings() {
		// Load defaults from core
		LoadIniSettingsFromDisk(std::string(getResourcesDirectory() + "/core/" + _defaultImguiSettingsFilename).c_str());
		
		// Load defaults from Window constructor
		if(fileExists(_windowImguiSettingsFilename))
			LoadIniSettingsFromDisk(_windowImguiSettingsFilename.c_str());

		// Load user specific settings for this particular window
		LoadIniSettingsFromDisk(ImGui::GetIO().IniFilename);
	}

	void Window::setup(int width, int height, const std::string& title, const WindowArgs & args, const std::string& defaultSettingsFilename) {
		// IMPORTANT NOTE: if you got compatibility problem with old opengl function,
		// try to load compat 3.2 instead of core 4.2

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

#ifdef GLEW_EGL
		glfwWindowHint(GLFW_CONTEXT_CREATION_API, (args.offscreen) ?
													GLFW_EGL_CONTEXT_API :
													GLFW_NATIVE_CONTEXT_API);
#else
		if(args.offscreen) SIBR_WRG << "Offscreen enabled without EGL implementation. Using native context (Offscreen might run into issues if no real display is available)." << std::endl;
#endif

		glfwWindowHint(GLFW_RED_BITS, 8);
		glfwWindowHint(GLFW_GREEN_BITS, 8);
		glfwWindowHint(GLFW_BLUE_BITS, 8);
		glfwWindowHint(GLFW_ALPHA_BITS, 8);
		glfwWindowHint(GLFW_DEPTH_BITS, 24);
		glfwWindowHint(GLFW_STENCIL_BITS, 8);

		if (args.offscreen) {
			glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		}

		_glfwWin = GLFWwindowptr(
			glfwCreateWindow(
				width, height, title.c_str(),
				(args.fullscreen && !args.offscreen) ? glfwGetPrimaryMonitor() : NULL
				, NULL ), 
			glfwDestroyWindow
		);

		if (_glfwWin == nullptr)
			SIBR_ERR << "failed to create a glfw window (is your graphics driver updated ?)" << std::endl;

		makeContextCurrent();

		

		//SR, TT fix for image size non divisible by 4
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		//// Print available OpenGL version
		GLint major, minor;
		glGetIntegerv(GL_MAJOR_VERSION, &major);
		glGetIntegerv(GL_MINOR_VERSION, &minor);
		SIBR_LOG << "OpenGL Version: " << glGetString(GL_VERSION)
			<< "[major: " << major << ", minor: " << minor << "]" << std::endl;

		glewExperimental = GL_TRUE;
		GLenum err = glewInit();
#ifdef GLEW_EGL
//		if (err != GLEW_OK && (!args.offscreen || err != GLEW_ERROR_NO_GLX_DISPLAY)) // Small hack for glew, this error occurs but does not concern offscreen
		if (err != GLEW_OK && (!args.offscreen )) // Small hack for glew, this error occurs but does not concern offscreen
#else
		if (err != GLEW_OK)
#endif
			SIBR_ERR << "cannot initialize GLEW (used to load OpenGL function)" << std::endl;
		(void)glGetError(); // I notice that glew might do wrong things during its init()
							// some drivers complain about it. So I reset OpenGL's errors to discard this.

		glfwSetWindowUserPointer(_glfwWin.get(), this);
		/// \todo TODO: fix, width and height might be erroneous. SR
		viewport(Viewport(0.f, 0.f, (float)width, (float)height));	/// \todo TODO: bind both

		_useVSync = args.vsync;
		glfwSwapInterval(args.vsync);
		glfwSetKeyCallback(_glfwWin.get(), glfwKeyboardCallback);
		glfwSetScrollCallback(_glfwWin.get(), glfwMouseScrollCallback);
		glfwSetMouseButtonCallback(_glfwWin.get(), glfwMouseButtonCallback);
		glfwSetCursorPosCallback(_glfwWin.get(), glfwCursorPosCallback);
		glfwSetWindowSizeCallback(_glfwWin.get(), glfwResizeCallback);

		// SR: we don't use it by default because you won't get callstack/file/line info.
		if(args.gl_debug) {
			glEnable(GL_DEBUG_OUTPUT);
			glDebugMessageCallback(glErrorCallback, nullptr);
		}

		if(_useGUI) {
			//contextId
			++Window::contextId;

			// Setup ImGui binding
			ImGui::CreateContext();
			ImGui_ImplGlfwGL3_Init(_glfwWin.get(), false);
			glfwSetCharCallback(_glfwWin.get(), ImGui_ImplGlfw_CharCallback);

			ImGui_ImplGlfwGL3_NewFrame();

			_windowImguiSettingsFilename = defaultSettingsFilename;

			// Set user specific file for this particular window as default ini file
			std::string iniFilename = std::string(getAppDataDirectory() + "/" + std::regex_replace(title, std::regex("[^0-9A-Za-z\\-_]"), "_") + ".ini").c_str();
			char* iniFilenameCStr = new char[iniFilename.length()+1];
			strcpy(iniFilenameCStr, iniFilename.c_str());
			ImGui::GetIO().IniFilename = iniFilenameCStr;

			loadSettings();
		}

		if(!args.offscreen) {
			_oldPosition = position();
			_oldSize = size();

			// Support for HiDPI on Windows. The default is 96.
			// Compute the pixel density at the current definition.
			int widthmm, heightmm;
			glfwGetMonitorPhysicalSize(glfwGetPrimaryMonitor(), &widthmm, &heightmm);
			const float defaultDPI = 96.0f;
			sibr::Vector2i dsize = desktopSize();
			
			_scaling = sibr::clamp(std::round(dsize.x() / (widthmm / 25.4f) / defaultDPI), 1.0f, 2.0f);

			if (_useGUI && args.hdpi) {
				ImGui::GetStyle().ScaleAllSizes(scaling());
				ImGui::GetIO().FontGlobalScale = scaling();
			}
		}

		/** \todo
		TODO: fix issue on some HiDPI screens + interaction with GUI labels generation.
		// If we have a screen in HiDPI mode, scale the interface accordingly.
		if (_scaling > 1.0f) {
			ImGui::GetStyle().ScaleAllSizes(_scaling);
			ImGui::GetIO().FontGlobalScale = _scaling;
		}
		*/
	}

	Vector2i		Window::desktopSize( void )
	{
		const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		return Vector2i(mode->width, mode->height);
	}

	Vector2i		Window::size( void ) const
	{
		Vector2i s;
		glfwGetWindowSize(_glfwWin.get(), &s[0], &s[1]);
		return s;
	}

	void Window::position(const unsigned int x, const unsigned int y)
	{
		glfwSetWindowPos(_glfwWin.get(), x, y);
	}

	Vector2i Window::position() const {
		Vector2i s;
		glfwGetWindowPos(_glfwWin.get(), &s[0], &s[1]);
		return s;
	}

	bool			Window::isOpened( void ) const
	{
		return (!_shouldClose && !glfwWindowShouldClose(_glfwWin.get()));
	}

	void			Window::close( void )
	{
		_shouldClose = true;
		glfwSetWindowShouldClose(_glfwWin.get(), GL_TRUE);
	}

	bool Window::isFullscreen(void) const
	{
		return glfwGetWindowMonitor(_glfwWin.get()) != NULL;
	}

	void Window::setFullscreen(const bool fullscreen) {
		const bool currentState = isFullscreen();
		if((fullscreen && currentState) || (!fullscreen && !currentState)) {
			// Do nothing.
			return;
		}
		if (fullscreen) {
			_oldPosition = position();
			_oldSize = size();
			GLFWmonitor* monitor = glfwGetPrimaryMonitor();
			const GLFWvidmode* mode = glfwGetVideoMode(monitor);
			glfwSetWindowMonitor(_glfwWin.get(), monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
			// There is a bug in glfw (see https://github.com/glfw/glfw/issues/1072).
			// We have to manually re-set the swap interval.
			glfwSwapInterval(_useVSync ? 1 : 0);
		} else {
			glfwSetWindowMonitor(_glfwWin.get(), NULL, _oldPosition[0], _oldPosition[1], _oldSize[0], _oldSize[1], 0);
		}
	}

	void			Window::size( int w, int h )
	{
		glfwSetWindowSize(_glfwWin.get(), w, h);
		Vector2i s = size();

		if (s[0] != w || s[1] != h)
			SIBR_WRG << "Attempting to resize the window to an unsuported resolution "
			"(w = " << w << ", h = " << h << " ), using w = " << s[0] << ", h = " << s[1] << " instead." << std::endl;

		// TT : should be the right thing to do, but might brake some old stuff
		viewport(Viewport(0.f, 0.f, (float)(s[0]), (float)(s[1])));

		//viewport(Viewport(0.f, 0.f, (float)(s[0]-1), (float)(s[1]-1))); // TODO: bind both
	}

	void Window::setFrameRate(int fps)
	{
		if (fps == 60) {
			glfwSwapInterval(1);
		} else if (fps == 30) {
			glfwSwapInterval(2);
		} else if (fps == 15) {
			glfwSwapInterval(3);
		}
	}

	bool Window::isVsynced(void) const
	{
		return _useVSync;
	}

	bool Window::isGUIEnabled(void) const
	{
		return _useGUI;
	}

	void Window::setVsynced(const bool vsync) {
		_useVSync = vsync;
		glfwSwapInterval(_useVSync ? 1 : 0);
	}

	void				Window::enableCursor( bool enable )
	{
		glfwSetInputMode(_glfwWin.get(), GLFW_CURSOR, enable? GLFW_CURSOR_NORMAL : GLFW_CURSOR_HIDDEN);
	}

	GLFWwindow * Window::GLFW(void) {
		return _glfwWin.get();
	}



} // namespace sibr
