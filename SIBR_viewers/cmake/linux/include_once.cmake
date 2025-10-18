# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


macro(include_once file)
    get_filename_component(INCLUDE_ONCE_FILEPATH ${file} REALPATH)
    string(REGEX REPLACE "(\\.|\\/+|\\:|\\\\+)" "_" INCLUDE_ONCE_FILEPATH ${INCLUDE_ONCE_FILEPATH})
    get_property(INCLUDED_${INCLUDE_ONCE_FILEPATH}_LOCAL GLOBAL PROPERTY INCLUDED_${INCLUDE_ONCE_FILEPATH})
    if (INCLUDED_${INCLUDE_ONCE_FILEPATH}_LOCAL)
        return()
    else()
        set_property(GLOBAL PROPERTY INCLUDED_${INCLUDE_ONCE_FILEPATH} true)

        include(${file})
    endif()
endmacro()