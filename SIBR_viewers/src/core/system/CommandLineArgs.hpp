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


#pragma once

# include <string>
# include <vector>
# include <map>

# include "Config.hpp"
# include <core/system/Vector.hpp>

namespace sibr
{
	/**
	\addtogroup sibr_system
	@{
	*/

	/// Used to wrap a toggle argument in the command line.
	struct Switch {};

	/// uint contexpr helper, defining the number of command line tokens required to init T
	template<typename T>
	constexpr uint NumberOfArg = 1;
	/// uint contexpr helper, defining the number of command line tokens required to init T
	template<>
	constexpr uint NumberOfArg<bool> = 0;
	/// uint contexpr helper, defining the number of command line tokens required to init T
	template<>
	constexpr uint NumberOfArg<Switch> = 0;
	/// uint contexpr helper, defining the number of command line tokens required to init T
	template<typename T, uint N>
	constexpr uint NumberOfArg<sibr::Vector<T, N>> = N * NumberOfArg<T>;

	/// Helper to extract values from a vector of strings.
	template<typename T> struct ValueGetter {

		/** Extract the N-th element from a vector of string representations.
		\param values a list of strings representing elements
		\param n the index of the element to query
		\return the element corresponding to the N-th string.
		*/
		static T get(const std::vector<std::string> & values, uint n);

		/** Convert an element to its string representation.
		\param value the element to convert
		\return the string representation
		*/
		static std::string toString(const T & value);
	};

	/** Available rendering modes for IBR views. */
	enum RenderingModes {
		RENDERMODE_MONO,
		RENDERMODE_STEREO_ANAGLYPH,
		RENDERMODE_STEREO_QUADBUFFER
	};



	/** @} */

	/* Parse and store the command line arguments specified by the user.
	* Only a static instance exists, that must be init with parseMainArgs(argc,argv) right after main(argc,argv)
	* Parses -key or --key with any number of value.
	* \ingroup sibr_system
	*/
	class SIBR_SYSTEM_EXPORT CommandLineArgs {

	public:

		/** Populate arguments list, should be called once at launch.
		 * \param argc argument count
		 * \param argv argument list
		 * */
		static void parseMainArgs(const int argc, const char* const* argv);

		/** Get the Nth parsed element following -key or --key as a T 
		* If not available (key not found or not enough token), return default_val argument
		* \param key the argument keyword
		* \param default_val the default value to use if not present
		* */
		template<typename T, uint N = 0>
		T get(const std::string & key, const T & default_val) const {
			T out;
			if (getInternal<T,N>(key, out)) {
				return out;
			} else {
				return default_val;
			}
		}

		/** Get the Nth parsed element following -key or --key as a T
		* If not available (key not found or not enough token), an error is raised.
		* \param key the argument keyword
		* */
		template<typename T, uint N = 0>
		T getRequired(const std::string & key) const {
			T out;
			if (!getInternal(key, out)) {
				SIBR_ERR;
			}
			return out;
		}

		/** Register a command for the help message.
		 * \param key the command
		 * \param description a string describing the use of the command
		 * \param defaultValue string representation of the default value
		 */
		void registerCommand(const std::string & key, const std::string & description, const std::string & defaultValue);

		/** Register a mandatory command for the help message.
		 * \param key the command
		 * \param description a string describing the use of the command
		 */
		void registerRequiredCommand(const std::string & key, const std::string & description);

		/** Check if a given argument was specified by the user.
		 *\param key the argument to look for
		 *\return whether the argument was specified
		 */
		bool contains(const std::string & key) const;

		/** Count how many parameters were specified by the suer for a given argument.
		 *\param key the argument to look for
		 *\return the number of parameters
		 */
		int numArguments(const std::string & key) const;

		/** Global instance getter. */
		static CommandLineArgs & getGlobal();

		/** Display an help message to stdout. */
		void displayHelp() const;

	protected:

		/// Default constructor.
		CommandLineArgs() = default;

		/** Get the Nth parsed element following -key or --key as a T
		* If not available (key not found or not enough token), an error is raised.
		* */
		template<typename T, uint N = 0>
		bool getInternal(const std::string & key, T & val) const {
			if (contains(key) && (N + 1)*NumberOfArg<T> <= args.at(key).size()) {
				val = ValueGetter<T>::get(args.at(key), N);

				return true;
			} else {
				return false;
			}
		}

		std::map<std::string, std::vector<std::string>> args; ///< List of arguments input by the user and their parameters.
		std::map<std::string, std::string> commands; ///< List of registered commands to display the help.
		bool init = false; ///< Have the arguments been parsed.

		static CommandLineArgs global;///< Singleton (because there is only one command line).
	};

	/** Getter for the command line args manager singleton.
	* \ingroup sibr_system
	 */
	SIBR_SYSTEM_EXPORT const CommandLineArgs & getCommandLineArgs();

	/** Internal argument based interface.
	* \ingroup sibr_system*/
	template<typename T>
	class ArgBase {
	public:

		/// \return a reference to the argument value
		operator const T &() const { return _value; }

		/// \return a reference to the argument value
		const T & get() const { return _value; }

		/** Copy operator.
		\param t the value to copy
		\return a reference to the argument value
		*/
		T & operator=(const T & t) { _value = t; return _value; }

	protected:

		T _value; ///< the argument value.
	};

	/** Template Arg class, will init itself in the defaut ctor using the command line args (ie. --key value)
	* Should be declared as some class/struct member using Arg<T> myArg = { "key", some_default_value };
	* is implicitly convertible to the template type
	* \note As multiple implicit conversion is not possible in cpp, you might have to use the .get() method to access the inner T value
	* \ingroup sibr_system
	* */
	template<typename T>
	class Arg : public ArgBase<T> {
	public:
		/** Constructor
		 *\param key the command argument
		 *\param default_value the default value
		 *\param description help message description
		 */
		Arg(const std::string & key, const T & default_value, const std::string & description = "") {
			this->_value = CommandLineArgs::getGlobal().get<T>(key, default_value);
			// \todo We could display default values if we had a common stringization method.
			CommandLineArgs::getGlobal().registerCommand(key, description, ValueGetter<T>::toString(default_value));
		}
		using ArgBase<T>::operator=;
	};

	/// Specialization of Arg for Switch, default value get flipped if arg is present
	/// \ingroup sibr_system
	template<>
	class Arg<Switch> : public ArgBase<bool> {
	public:
		/** Constructor
		 *\param key the command argument
		 *\param default_value the default boolean value
		 *\param description help message description
		 */
		Arg(const std::string & key, const bool & default_value, const std::string & description = "") {
			const bool arg_is_present = CommandLineArgs::getGlobal().get<bool>(key, false);
			if (arg_is_present) {
				_value = !default_value;
			} else {
				_value = default_value;
			}
			const std::string defaultDesc = (default_value ? "enabled" : "disabled");
			CommandLineArgs::getGlobal().registerCommand(key, description, defaultDesc);
		}

		using ArgBase<bool>::operator=;
	};
	using ArgSwitch = Arg<Switch>;

	/// Specialization of Arg for bool, value is true if key is present and false otherwise
	/// \ingroup sibr_system
	template<>
	class Arg<bool> : public ArgBase<bool> {
	public:
		/** Constructor
		 *\param key the command argument
		 *\param description help message description
		 *\note Will default to false
		 */
		Arg(const std::string & key, const std::string & description = "") {
			const bool arg_is_present = CommandLineArgs::getGlobal().get<bool>(key, false);
			_value = arg_is_present;
			CommandLineArgs::getGlobal().registerCommand(key, description, "disabled");
		}

		using ArgBase<bool>::operator=;
	};

	/// Represent a mandatory argument
	/// \ingroup sibr_system
	template<typename T>
	class RequiredArgBase {
	public:
		/** Constructor
		 *\param _key the command argument
		 *\param description help message description
		 */
		RequiredArgBase(const std::string & _key, const std::string & description = "") : key(_key) {
			if (CommandLineArgs::getGlobal().contains(key)) {
				_value = CommandLineArgs::getGlobal().get<T>(key, _value);
				wasInit = true;
			}
			CommandLineArgs::getGlobal().registerRequiredCommand(key, description);
		}

		/// \return a reference to the argument value
		operator const T &() const { checkInit(); return _value; }

		/// \return a reference to the argument value
		const T & get() const { checkInit(); return _value; }

		/** Copy operator.
		\param t the value to copy
		\return a reference to the argument value
		*/
		T & operator=(const T & t) { _value = t; wasInit = true; return _value; }

		/// \return true if the argument was given
		const bool & isInit() const { return wasInit; }

	protected:

		/** Check if the argument was init.If not, as it is a required argument we display the help message and raise an error.*/
		void checkInit() const {
			if (!wasInit) {
				CommandLineArgs::getGlobal().displayHelp();
				SIBR_ERR << "Argument \"" << key << "\" is required." << std::endl;
			}
		}

		std::string key; ///< Argument key.
		T _value; ///< Argument value.
		bool wasInit = false; ///< Was the argument initialized.
	};

	/// Similar to Arg, except this one will crash if attempt to use the value while not initialized
	/// initialization can be done using the command line or manually
	/// \ingroup sibr_system
	template<typename T>
	class RequiredArg : public RequiredArgBase<T> {
		using RequiredArgBase<T>::RequiredArgBase;
	};

	/// Specialization required for std::string as const string & key constructor and const T & constructor are ambiguous. 
	/// TT : no const T & ctor anymore but operator const char*() const operator added
	/// \ingroup sibr_system
	template<>
	class RequiredArg<std::string> : public RequiredArgBase<std::string> {
		
	public:
		using RequiredArgBase<std::string>::RequiredArgBase;
		std::string & operator=(const std::string & t) { _value = t; wasInit = true; return _value; }

		operator const char*() const { checkInit(); return _value.c_str(); }
	};

	/// Hierarchy of Args classes that can be seens as modules, and can be combined using virtual inheritance, with no duplication of code so derived Args has no extra work to do
	/// Assuming CommandLineArgs::parseMainArgs() was called once, Args arguments will be automatically initialized with the value from the command line by the constructor
	/// Existing Args structs should cover most of the existing IBR apps
	/// To add a new argument like --my-arg 5 on top of existing arguments and
	/// to add a new required argument like --important-param "on" on top of existing arguments, do the following:
	///
	/// struct SIBR_SYSTEM_EXPORT MyArgs : virtual ExistingArg1, virtual ExistingArgs2, ... {
	///		Arg<int> myParameter = { "my-arg", some_default_value };
	///		RequiredArg<std::string> myRequiredParameter = { "important-param" };
	/// }
	/// \ingroup sibr_system
	struct SIBR_SYSTEM_EXPORT AppArgs {
		/// Constructor
		AppArgs();

		std::string appName;
		std::string appPath;
		Arg<std::string> custom_app_path = { "appPath", "./", "define a custom app path" };
		Arg<bool> showHelp = {"help", "display this help message"};

		/// Helper to print the help message if the help argument was passed.
		void displayHelpIfRequired() const;

		// offline path rendering options
		Arg<bool> noExit = {"noExit", "dont exit after rendering path "};
		Arg<std::string> pathFile = { "pathFile", "", "filename of path to render offline; app renders path and exits" }; // app needs to handle this; if it does default behavior is to render the path and exit
		Arg<std::string> outPath = { "outPath", "pathOutput", "Path of directory to store path output default relative the input path directory " }; // app needs to handle this; if it does default behavior is to render the path and exit

	};

	/// Arguments related to a window.
	/// \ingroup sibr_system
	struct SIBR_SYSTEM_EXPORT WindowArgs {
		Arg<int> win_width = { "width", 720, "initial window width" };
		Arg<int> win_height = { "height", 480, "initial window height" };
		Arg<int> vsync = { "vsync", 1, "enable vertical sync" };
		Arg<bool> fullscreen = { "fullscreen", "set the window to fullscreen" };
		Arg<bool> hdpi = { "hd", "rescale UI elements for high-density screens" };
		Arg<bool> no_gui = { "nogui", "do not use ImGui" };
		Arg<bool> gl_debug = { "gldebug", "enable OpenGL error callback" };
		Arg<bool> offscreen = { "offscreen", "do not open window" };
	};

	/// Combination of window and application arguments.
	/// \ingroup sibr_system
	struct SIBR_SYSTEM_EXPORT WindowAppArgs :
		virtual AppArgs, virtual WindowArgs {
	};

	/// Common rendering settings.
	/// \ingroup sibr_system
	struct SIBR_SYSTEM_EXPORT RenderingArgs {
		Arg<std::string> scene_metadata_filename = { "scene", "scene_metadata.txt", "scene metadata file" };
		Arg<Vector2i> rendering_size = { "rendering-size", { 0, 0 }, "size at which rendering is performed" };
		Arg<int> texture_width = { "texture-width", 0 , "size of the input data in memory"};
		Arg<float> texture_ratio = { "texture-ratio", 1.0f };
		Arg<int> rendering_mode = { "rendering-mode", RENDERMODE_MONO, "select mono (0) or stereo (1) rendering mode" };
		Arg<sibr::Vector3f> focal_pt = { "focal-pt", {0.0f, 0.0f, 0.0f} };
		Arg<Switch> colmap_fovXfovY_flag = { "colmap_fovXfovY_flag", false };
		Arg<Switch> force_aspect_ratio = { "force-aspect-ratio", false };
	};

	/// Dataset related arguments.
	/// \ingroup sibr_system
	struct SIBR_SYSTEM_EXPORT BasicDatasetArgs {
		RequiredArg<std::string> dataset_path = { "path", "path to the dataset root" };
		Arg<std::string> dataset_type = { "dataset_type", "", "type of dataset" };
	};

	/// "Default" set of arguments.
	/// \ingroup sibr_system
	struct SIBR_SYSTEM_EXPORT BasicIBRAppArgs :
		virtual WindowAppArgs, virtual BasicDatasetArgs, virtual RenderingArgs {
	};

	/// Specialization of value getter for strings.
	/// \ingroup sibr_system
	template<>
	struct ValueGetter<std::string> {
		static std::string get(const std::vector<std::string> & values, uint n) {
			return values[n];
		}
		static std::string toString(const std::string & value) {
			return "\"" + value + "\"";
		}
	};

	/// Specialization of value getter for booleans.
	/// \ingroup sibr_system
	template<>
	struct ValueGetter<bool> {
		static bool get(const std::vector<std::string> & values, uint n) {
			return true;
		}
		static std::string toString(const bool & value) {
			return value ? "true" : "false";
		}
	};

	/// Specialization of value getter for doubles.
	/// \ingroup sibr_system
	template<>
	struct ValueGetter<double> {
		static double get(const std::vector<std::string> & values, uint n) {
			return std::stod(values[n]);
		}
		static std::string toString(const double & value) {
			return std::to_string(value);
		}
	};

	/// Specialization of value getter for floats.
	/// \ingroup sibr_system
	template<>
	struct ValueGetter<float> {
		static float get(const std::vector<std::string> & values, uint n) {
			return std::stof(values[n]);
		}
		static std::string toString(const float & value) {
			return std::to_string(value);
		}
	};

	/// Specialization of value getter for integers.
	/// \ingroup sibr_system
	template<>
	struct ValueGetter<int> {
		static int get(const std::vector<std::string> & values, uint n) {
			return std::stoi(values[n]);
		}
		static std::string toString(const int & value) {
			return std::to_string(value);
		}
	};

	/// Specialization of value getter for chars.
	/// \ingroup sibr_system
	template<>
	struct ValueGetter<char> {
		static char get(const std::vector<std::string> & values, uint n) {
			return static_cast<char>(std::stoi(values[n]));
		}
		static std::string toString(const char & value) {
			return std::to_string(value);
		}
	};

	/// Specialization of value getter for unsigned integers.
	/// \ingroup sibr_system
	template<>
	struct ValueGetter<uint> {
		static uint get(const std::vector<std::string> & values, uint n) {
			return static_cast<uint>(std::stoi(values[n]));
		}
		static std::string toString(const uint & value) {
			return std::to_string(value);
		}
	};

	/// Specialization of value getter for arrays.
	/// \ingroup sibr_system
	template<typename T, uint N>
	struct ValueGetter<std::array<T, N>> {
		static std::array<T, N> get(const std::vector<std::string> & values, uint n) {
			std::array<T, N> out;
			for (uint i = 0; i < N; ++i) {
				out[i] = ValueGetter<T>::get(values, n*N*NumberOfArg<T> + i);
			}
			return out;
		}
		static std::string toString(const std::array<T, N> & value) {
			std::string res = "(";
			for (uint i = 0; i < N; ++i) {
				res.append(ValueGetter<T>::toString(value[i]));
				if (i != N - 1) {
					res.append(",");
				}
			}
			return res + ")";
		}
	};

	/// Specialization of value getter for sibr::Vectors & eigen matrices.
	/// \ingroup sibr_system
	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	struct ValueGetter<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
		static Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> get(const std::vector<std::string> & values, uint n) {
			Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> out;
			for (uint i = 0; i < _Rows*_Cols; ++i) {
				out[i] = ValueGetter<_Scalar>::get(values, n*_Rows*_Cols*NumberOfArg<_Scalar> + i);
			}
			return out;
		}
		static std::string toString(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & value) {
			std::string res = "(";
			for (uint i = 0; i < _Rows*_Cols; ++i) {
				res.append(ValueGetter<_Scalar>::toString(value[i]));
				if(i != _Rows*_Cols-1) {
					res.append(",");
				}
			}
			return res + ")";
		}
	};

} // namespace sibr
