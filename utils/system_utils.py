#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
import re

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = []
    for fname in os.listdir(folder):
        match = re.fullmatch(r"iteration_([0-9]+)", fname)
        if match and path.isdir(path.join(folder, fname)):
            saved_iters.append(int(match.group(1)))
    if not saved_iters:
        raise ValueError(f"No iteration directories found in {folder}")
    return max(saved_iters)
