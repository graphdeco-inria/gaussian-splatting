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

import subprocess
import os, sys
from shutil import which
from utils.paths import getBinariesPath, getColmapPath, getMeshlabPath, getRCPath

def getProcess(programName, binaryPath = getBinariesPath()):
    suffixes = [ '', '_msr', '_rwdi', '_d']

    print("BINARIES ", binaryPath)
    for suffix in suffixes:
        binary = os.path.join(binaryPath, programName + suffix + (".exe" if os.name == 'nt' else ''))

        if os.path.isfile(binary) or which(binary) is not None:
            print("Program '%s' found in '%s'." % (programName, binary))
            return binary

def getRCprocess(binaryPath = getRCPath()):
    programName = "RealityCapture"
    binary = os.path.join(binaryPath, programName + ".exe")

    if os.path.isfile(binary):
        print("Program '%s' found in '%s'." % (programName, binary))
        return binary


def runCommand(binary, command_args):
#    print("Running process '%s'" % (' '.join([binary, *command_args])))
    sys.stdout.flush()
    completedProcess = subprocess.run([binary, *command_args])

    if completedProcess.returncode == 0:
        print("Process %s completed." % binary)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        print("Process %s failed with code %d." % (binary, completedProcess.returncode))

    return completedProcess

def getColmap(colmapPath = getColmapPath()):
    colmapBinary = os.path.join(colmapPath, "COLMAP.bat" if os.name == 'nt' else 'colmap')

    if os.path.isfile(colmapBinary) or which(colmapBinary) is not None:
        print("Program '%s' found in '%s'." % (colmapBinary, colmapPath))
        return colmapBinary
    else:
        print("Program '%s' not found in '%s'. Aborting." % (colmapBinary, colmapPath))
        return None

def getMeshlabServer(meshlabPath = getMeshlabPath()):
    meshlabserverBinary = os.path.join(meshlabPath, "meshlabserver" + ('.exe' if os.name == 'nt' else ''))

    if os.path.isfile(meshlabserverBinary) or which(meshlabserverBinary) is not None:
        print("Program '%s' found in '%s'." % (meshlabserverBinary, meshlabPath))
        return meshlabserverBinary
    else:
        print("Program '%s' not found in '%s'. Aborting." % (meshlabserverBinary, meshlabPath))
        return None
