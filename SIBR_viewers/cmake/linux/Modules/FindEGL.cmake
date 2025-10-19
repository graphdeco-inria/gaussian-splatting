#.rst:
# FindEGL
# -------
#
# Try to find EGL.
#
# This will define the following variables:
#
# ``EGL_FOUND``
#     True if (the requested version of) EGL is available
# ``EGL_VERSION``
#     The version of EGL; note that this is the API version defined in the
#     headers, rather than the version of the implementation (eg: Mesa)
# ``EGL_LIBRARIES``
#     This can be passed to target_link_libraries() instead of the ``EGL::EGL``
#     target
# ``EGL_INCLUDE_DIRS``
#     This should be passed to target_include_directories() if the target is not
#     used for linking
# ``EGL_DEFINITIONS``
#     This should be passed to target_compile_options() if the target is not
#     used for linking
#
# If ``EGL_FOUND`` is TRUE, it will also define the following imported target:
#
# ``EGL::EGL``
#     The EGL library
#
# In general we recommend using the imported target, as it is easier to use.
# Bear in mind, however, that if the target is in the link interface of an
# exported library, it must be made available by the package config file.
#
# Since pre-1.0.0.

#=============================================================================
# Copyright 2014 Alex Merry <alex.merry@kde.org>
# Copyright 2014 Martin Gräßlin <mgraesslin@kde.org>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# Use pkg-config to get the directories and then use these values
# in the FIND_PATH() and FIND_LIBRARY() calls
find_package(PkgConfig)
pkg_check_modules(PKG_EGL QUIET egl)

set(EGL_DEFINITIONS ${PKG_EGL_CFLAGS_OTHER})

find_path(EGL_INCLUDE_DIR
    NAMES
        EGL/egl.h
    HINTS
        ${PKG_EGL_INCLUDE_DIRS}
)
find_library(EGL_LIBRARY
    NAMES
        EGL
    HINTS
        ${PKG_EGL_LIBRARY_DIRS}
)

# NB: We do *not* use the version information from pkg-config, as that
#     is the implementation version (eg: the Mesa version)
if(EGL_INCLUDE_DIR)
    # egl.h has defines of the form EGL_VERSION_x_y for each supported
    # version; so the header for EGL 1.1 will define EGL_VERSION_1_0 and
    # EGL_VERSION_1_1.  Finding the highest supported version involves
    # finding all these defines and selecting the highest numbered.
    file(READ "${EGL_INCLUDE_DIR}/EGL/egl.h" _EGL_header_contents)
    string(REGEX MATCHALL
        "[ \t]EGL_VERSION_[0-9_]+"
        _EGL_version_lines
        "${_EGL_header_contents}"
    )
    unset(_EGL_header_contents)
    foreach(_EGL_version_line ${_EGL_version_lines})
        string(REGEX REPLACE
            "[ \t]EGL_VERSION_([0-9_]+)"
            "\\1"
            _version_candidate
            "${_EGL_version_line}"
        )
        string(REPLACE "_" "." _version_candidate "${_version_candidate}")
        if(NOT DEFINED EGL_VERSION OR EGL_VERSION VERSION_LESS _version_candidate)
            set(EGL_VERSION "${_version_candidate}")
        endif()
    endforeach()
    unset(_EGL_version_lines)
endif()

cmake_push_check_state(RESET)
list(APPEND CMAKE_REQUIRED_LIBRARIES "${EGL_LIBRARY}")
list(APPEND CMAKE_REQUIRED_INCLUDES "${EGL_INCLUDE_DIR}")

check_cxx_source_compiles("
#include <EGL/egl.h>

int main(int argc, char *argv[]) {
    EGLint x = 0; EGLDisplay dpy = 0; EGLContext ctx = 0;
    eglDestroyContext(dpy, ctx);
}" HAVE_EGL)

cmake_pop_check_state()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EGL
    FOUND_VAR
        EGL_FOUND
    REQUIRED_VARS
        EGL_LIBRARY
        EGL_INCLUDE_DIR
        HAVE_EGL
    VERSION_VAR
        EGL_VERSION
)

if(EGL_FOUND AND NOT TARGET EGL::EGL)
    add_library(EGL::EGL UNKNOWN IMPORTED)
    set_target_properties(EGL::EGL PROPERTIES
        IMPORTED_LOCATION "${EGL_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${EGL_DEFINITIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${EGL_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(EGL_LIBRARY EGL_INCLUDE_DIR HAVE_EGL)

# compatibility variables
set(EGL_LIBRARIES ${EGL_LIBRARY})
set(EGL_INCLUDE_DIRS ${EGL_INCLUDE_DIR})
set(EGL_VERSION_STRING ${EGL_VERSION})

include(FeatureSummary)
set_package_properties(EGL PROPERTIES
    URL "https://www.khronos.org/egl/"
    DESCRIPTION "A platform-agnostic mechanism for creating rendering surfaces for use with other graphics libraries, such as OpenGL|ES and OpenVG."
)
