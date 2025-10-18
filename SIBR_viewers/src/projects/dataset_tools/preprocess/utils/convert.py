# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


#!/usr/bin/env python
#! -*- encoding: utf-8 -*-

import os
import re

def updateStringFromDict(string, map, format='${%s}', fix_paths=True):
    newstring = string

    for keyword, value in map.items():
        newstring = newstring.replace(format % keyword, str(value))
        # if it's a path, get absolute path
        if fix_paths and re.match(r"^(?:\w:[\\\/]*|[@A-Za-z_.0-9-]*[\\\/]+|\.{1,2}[\\\/])(?:[\\\/]|[@A-Za-z_.0-9-]+)*$", newstring):
            newstring = os.path.abspath(newstring)

    return newstring

def fixMeshEol(meshPath, newMeshPath):
    with open(meshPath,"rb") as meshFile, open(newMeshPath, "wb") as newMeshFile:
        meshBytes = meshFile.read()
        endBytes = b"end_header"
        badEol = b"\r\n"
        newEol = b"\n"

        index = meshBytes.find(endBytes) + len(endBytes) + len(badEol)

        newMeshBytes = meshBytes[0:index].replace(badEol, newEol)
        newMeshBytes += meshBytes[index:]

        newMeshFile.write(newMeshBytes)