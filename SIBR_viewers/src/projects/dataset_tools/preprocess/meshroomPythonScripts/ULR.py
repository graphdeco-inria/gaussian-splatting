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
__version__ = "3.0"

from meshroom.core import desc
import os

class ULR(desc.CommandLineNode):
    commandLine = 'SIBR_ulrv2_app_rwdi {allParams}'

    print(os.path.abspath('.'))
    cpu = desc.Level.INTENSIVE
    ram = desc.Level.INTENSIVE

    inputs = [
        desc.ListAttribute(
            elementDesc = desc.File(
                name = "path",
                label = "Cache folder",
                description = "",
                value = desc.Node.internalFolder + "../..",
                uid=[0],
            ),
            name='path',
            label='Input Folder',
            description='MeshroomCache folder containing the StructureFromMotion folder, PrepareDenseScene folder, and Texturing folder.'
        ),
        desc.ChoiceParam(
            name='texture-width',
            label='Texture Width',
            description='''Output texture size''',
            value=1024,
            values=(256, 512, 1024, 2048, 4096),
            exclusive=True,
            uid=[0],
        ),
    ]

    outputs = [
    ]
