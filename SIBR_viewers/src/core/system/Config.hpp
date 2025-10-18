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

//// Default includes ////
# ifndef _USE_MATH_DEFINES
# define _USE_MATH_DEFINES // for C++
# endif
# include <cmath>
# include <cassert>
# include <iostream>
# include <string>
# include <memory>
# include <numeric>
# include <algorithm>
# include <mutex>
# include <stdint.h>
#include <boost/filesystem.hpp>

//// Determine the operating system ////
# if defined(_WIN32)
#  define SIBR_OS_WINDOWS
// Windows define macro for 'far' and 'near'...
// http://stackoverflow.com/questions/118774/is-there-a-clean-way-to-prevent-windows-h-from-creating-a-near-far-macro
// We could use other names than far and near but because we work in
// computer graphics, I am sure that future guys will also try to
// declare variables called far/near and loose time until finding
// this is all because windows.

// Edit: I wanted to do something about it (a warning message) but it
// slow the compilation time (~5sec on my machine), so I let this code
// but disabled by default.
#  if SIBR_UNDEF_WINDOWMACROS
// The strategy here is to undef macros AFTER including
// headers that use them.
#   pragma warning(push, 0)

// Note including this file increase the compilation time
// of the core libs by 5 additional seconds.
#    include <windows.h>
#    include <shlguid.h>
#    include <commctrl.h>
#    include <isguids.h>
#    include <ShlObj.h>
#   pragma warning(pop)
#   undef far
#   undef near
#  endif // SIBR_UNDEF_WINDOWMACROS
# elif defined(__unix__)
#  define SIBR_OS_UNIX
# elif defined(__APPLE__) && defined(__MACH__)
#  define SIBR_OS_MAC
# else
#  error This operating system might be not supported.
# endif

//# undef NDEBUG /// \todo By undefining NDEBUG, I enable the assert system definition. (TODO RELEASE: remove this)
// (it certainly not a good pratice but it reveal previous assert already in the code)

# ifdef SIBR_OS_WINDOWS

#   pragma warning(disable:4503) // decorated name length exceeded, name was truncated
//   The two following lines disable warning concerning 'inconsistent dll linkage'.
//   MSVC doesn't like exporting STL containers because their implementation (their 'dll')
//   can be different from one Windows to another. Unix garantees to provide a universal
//   implementation and doesn't have this problem.
//   My point of view is:
//   - Make the code compliants with Windows' dlls will make us:
//     1) lose lots of time (if we need to wrap STL containers each time we use them...)
//     2) break the beauty of the code (we want to keep simple code).
//   - Once we will release this code (for a large public), we should:
//     1) Either explicitely export EVERY template/stl containers we use.
//     2) Or provide msvc's dll (redistribuable) that contains the same stl implementation
#   pragma warning(disable:4251)
#   pragma warning(disable:4273)

//// Export Macro (used for creating DLLs) ////
#  ifdef SIBR_STATIC_DEFINE
#    define SIBR_EXPORT
#    define SIBR_NO_EXPORT
#  else
#    ifndef SIBR_SYSTEM_EXPORT
#      ifdef SIBR_SYSTEM_EXPORTS
          /* We are building this library */
#        define SIBR_SYSTEM_EXPORT __declspec(dllexport)
#      else
          /* We are using this library */
#        define SIBR_SYSTEM_EXPORT __declspec(dllimport)
#      endif
#    endif
#    ifndef SIBR_NO_EXPORT
#      define SIBR_NO_EXPORT
#    endif
#  endif
# else
#  define SIBR_SYSTEM_EXPORT
# endif

//// Deprecator Macro (used to flag as 'deprecated' some functionalities)  ////
#ifndef SIBR_DEPRECATED
#  define SIBR_DEPRECATED __declspec(deprecated)
#endif

//// Int To String Macro (used to convert int into string at compile-time) ////
# define SIBR_MACROINTTOSTR_IMPL(x) #x		// small trick to get __LINE__ into a string
# define SIBR_MACROINTTOSTR(x) SIBR_MACROINTTOSTR_IMPL(x)
//// Concatenate Macro (used to concatenate two things, whatever it is.    ////
//// See SIBR_PROFILESCOPE for an example of use).                         ////
# define SIBR_CATMACRO_IMPL(x, y) x ## y
# define SIBR_CATMACRO(x, y) SIBR_CATMACRO_IMPL(x, y)

//# if SIBR_OS_WINDOWS
#  define __FUNCTION_STR__ __FUNCTION__
//# else
//#  define __FUNCTION_STR__ SIBR_MACROINTTOSTR(__FUNCTION__)
//# endif

// Macro used for
// Use: #pragma message WARN("My message")
# if _MSC_VER
#  define FILE_LINE_LINK __FILE__ "(" SIBR_MACROINTTOSTR(__LINE__) ") : "
#  define PRAGMAWARN(exp) (FILE_LINE_LINK "WARNING: " exp)
# else//__GNUC__ - may need other defines for different compilers
#  define PRAGMAWARN(exp) ("WARNING: " exp)
# endif

//// Math Macro ////
# define SIBR_PI	3.14159265358979323846
# define SIBR_2PI (SIBR_PI * 2.0)

# define SIBR_PI_DIV_180	0.01745329251
# define SIBR_180_DIV_PI	57.2957795131

# define SIBR_RADTODEG(x)	((x) * (float)SIBR_180_DIV_PI) // ( (x) * (180.0f / PI) )
# define SIBR_DEGTORAD(x)	((x) * (float)SIBR_PI_DIV_180) // ( (x) * (PI / 180.0f) )

//// Class Attribute Macro ////
# define SIBR_DISALLOW_COPY( classname )		\
	private:									\
	classname( const classname& );				\
	classname& operator =( const classname& );

# define SIBR_CLASS_PTR( classname )			\
	public:										\
	typedef std::shared_ptr<classname>	Ptr;	\
	typedef std::unique_ptr<classname>	UPtr;

namespace sibr
{
	/** Ensure that all logs are output before exiting when an error or exception is raised. 
	\ingroup sibr_system
	*/
	struct SIBR_SYSTEM_EXPORT LogExit
	{
		/// Constructor.
		LogExit( void );

		/** Throw an exception and trigger exit.
		\param stream the log stream.
		*/
		void operator <<=( const std::ostream& stream );

		std::lock_guard<std::mutex>		lock; ///< Sync lock.
	};
}


#ifdef NDEBUG
# define SIBR_MAXIMIZE_INLINE
#endif

# ifdef SIBR_MAXIMIZE_INLINE
#  define SIBR_OPT_INLINE	inline
# else
#  define SIBR_OPT_INLINE
# endif


//// Log Macro ////
# define SIBR_LOG	std::cout << "[SIBR] --  INFOS  --:\t"			// Must be replaced by a true log system
# define SIBR_WRG	std::cout << "[SIBR] !! WARNING !!:\tFILE " << __FILE__  << "\n\t\t\tLINE " << __LINE__ << ", FUNC " << __FUNCTION_STR__ << "\n\t\t\t"
# define SIBR_ERR ::sibr::LogExit() <<= \
					std::cerr << "[SIBR] ##  ERROR  ##:\tFILE " << __FILE__  << "\n\t\t\tLINE " << __LINE__ << ", FUNC " << __FUNCTION_STR__ << "\n\t\t\t"		// Could be augmented for exiting

// One drawback of using the standard assert is that you MUST catch the exception
// it throws in order to display its message and know the error. Not everyone thinks
// to do this (or want to add try/catch block in their code). Thus the solution here
// is to, first display the message (btw we inform on the precise location where it
// happens using __FILE__ and __LINE__) and then throw an exception (using the std
// assert) so that we can retrieve the callstack easily for debugging).


#ifdef NDEBUG
# define SIBR_ASSERT(condition) ((void)0)
# define SIBR_ASSERT_LOGIC(condition) (condition)	// This assertion can contain code logic (this code will also be included at release)
#else
# define SIBR_ASSERT(condition)			do { if(!(condition)) { SIBR_WRG << "ASSERT FAILED: " #condition << std::endl; assert(condition); } } while(0)
# define SIBR_ASSERT_LOGIC(condition)	do { if(!(condition)) { SIBR_WRG << "ASSERT FAILED: " #condition << std::endl; assert(condition); } } while(0)
#endif

// Small variants for adding function name
# define SIBR_FLOG SIBR_LOG "[" << __FUNCTION_STR__ << "]"
// Some code parts are written to manage additional or future features. They might remain untested
// until they are required (avoiding losing time to test code that could be useless at the end).
# define SIBR_UNTESTED	\
	SIBR_LOG << "!Warning! Using an untested code flagged as potentially "		\
	"unstable. (if something goes wrong, check over here - " __FILE__ ":" << __LINE__ << ")" << std::endl;
# define SIBR_DEBUG(var) std::cout << __FILE__ ":\n" "[Debug] " #var " = "<< (var) << std::endl		// No access to debug mode for now (so I made this tmp tool)

// Note Visual studio is bugged with multiple statements macro (I avoided them):
// http://stackoverflow.com/questions/22212737/strange-syntax-error-reported-in-a-range-based-for-loop

//// TYPEDEF ////
typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef unsigned uint;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

using Path = boost::filesystem::path;

// This stuff should be in a file gathering all debug tools
//# if !defined(NDEBUG)
#  include <ctime>
namespace sibr
{
	/// Used for quickly measuring time for completing a scope.
	/// \ingroup sibr_system
	struct SIBR_SYSTEM_EXPORT DebugScopeProfiler
	{
		/** Constructor.
		\param name the display name of the profiling session
		*/
		DebugScopeProfiler( const std::string& name );
		
		/// Destructor.
		~DebugScopeProfiler( void );
	
	private:
		clock_t _t0; ///< Timing.
		std::string _name; ///< Name.
	};

# define SIBR_PROFILESCOPE_EXPAND(x, y) sibr::DebugScopeProfiler x(y);
	// its a bit weird (because of macro's tricks) but that just create an instance of DebugScopeProfiler (with a generated var name)
# define SIBR_PROFILESCOPE	\
	SIBR_PROFILESCOPE_EXPAND(SIBR_CATMACRO(debugScopeProfiler,__COUNTER__), std::string(__FUNCTION_STR__) + std::string(" (File: " __FILE__ ":" SIBR_MACROINTTOSTR(__LINE__) ")") );
# define SIBR_PROFILESCOPE_NAME(name) \
	SIBR_PROFILESCOPE_EXPAND(SIBR_CATMACRO(debugScopeProfiler,__COUNTER__), name );
} // namespace sibr
//# endif

//// Define the init behavior ////
# define SIBR_INITZERO
// Initializing with a default value (zero) can be slightly slower but
// make the code safe. In the current case, we don't have performance
// problems (of this level).
# if defined(SIBR_INITZERO)
#  ifndef EIGEN_INITIALIZE_MATRICES_BY_ZERO
#   define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#  endif
# endif

// I didn't use the Plugin system for Eigen's MatrixBase
// because I wanted also custom ctor
// EDIT:
// Now that libslmini leaved out, we don't need this ctor
// anymore.
# define EIGEN_MATRIXBASE_PLUGIN "core/system/MatrixBasePlugin.hpp"
# define EIGEN_MATRIX_PLUGIN "core/system/MatrixPlugin.hpp"
# include <Eigen/Core>
# include <Eigen/Geometry>

# define SIBR_USE_CHOLMOD_EIGEN



namespace sibr
{
	/** Rounding operation.
	\param x the value to round
	\return the rounded value
	\todo Compare behaviour with std::round
	\ingroup sibr_system
	*/
	inline float round(float x) {
		return x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
	}

} // namespace sibr
