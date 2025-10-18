# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, sys, os
from PIL import Image, UnidentifiedImageError

def generateListImages(imagesPath, outputPath = None, filename = "list_images.txt"):
    if not os.path.exists(imagesPath):
        print("Path '%s' does not exists. Aborting." % imagesPath)
        sys.exit(1)

    if not outputPath:
        outputPath = imagesPath
    elif not os.path.exists(outputPath):
        print("Path '%s' does not exists. Aborting." % outputPath)
        sys.exit(1)

    files = os.listdir(imagesPath)

    if not files:
        print("No files found in directory '%s'. Aborting." % imagesPath)
        sys.exit(1)

    with open(os.path.join(outputPath, filename), "w") as list_images:
        for file in files:
            try:
                if os.path.isdir(os.path.join(imagesPath, file)):
                    raise UnidentifiedImageError()
                with Image.open(os.path.join(imagesPath, file)) as image:
                    list_images.write("%s %s %s\n" % (file, image.width, image.height))
            except UnidentifiedImageError:
                print("File '%s' is not a recognizable image. Skipping." % file)



def main():
    parser = argparse.ArgumentParser()

    # common arguments
    parser.add_argument("--imagesPath", type=str, required=True, help="path to your images folder")
    parser.add_argument("--outputPath", type=str, default=None, help="output path where to place the list_images.txt")
    parser.add_argument("--filename", type=str, default=None, help="filename for the image list")

    args = vars(parser.parse_args())

    if not args["outputPath"]:
        args["outputPath"] = args["imagesPath"]
    elif not os.path.isdir(args["outputPath"]) and os.path.basename(args["outputPath"]) and not args["filename"]:
        args["outputPath"], args["filename"] = os.path.split(args["outputPath"])
    
    if not args["filename"]:
        args["filename"] = "list_images.txt"

    print("Generating '%s' file from images in '%s' and saving to '%s'." % (args["filename"], args["imagesPath"], args["outputPath"]))

    generateListImages(os.path.abspath(args["imagesPath"]), os.path.abspath(args["outputPath"]), args["filename"])

    print("'%s' generated successfully." % args["filename"])
    sys.exit(0)

if __name__ == "__main__":
    main()