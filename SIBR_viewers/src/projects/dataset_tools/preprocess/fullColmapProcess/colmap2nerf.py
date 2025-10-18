#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument("--video_in", default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
    parser.add_argument("--video_fps", default=2)
    parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
    parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
    parser.add_argument("--images", default="images", help="input path to the images")
    parser.add_argument("--text", default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
    parser.add_argument("--aabb_scale", default=16, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--skip_early", default=0, help="skip this many images from the start")
    parser.add_argument("--out", default="transforms.json", help="output path")
    parser.add_argument("--path", default="", help="top level dataset")
    args = parser.parse_args()
    return args

def do_system(arg):
    print(f"==== running: {arg}")
    err=os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


def run_ffmpeg(args):
    if not os.path.isabs(args.images):
        args.images = os.path.join(os.path.dirname(args.video_in), args.images)
    images=args.images
    video=args.video_in
    fps=float(args.video_fps) or 1.0
    print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
        sys.exit(1)
    try:
        shutil.rmtree(images)
    except:
        pass
    do_system(f"mkdir {images}")
    do_system(f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}\" {images}/%04d.jpg")

def run_colmap(args):
    db=args.colmap_db
    images=args.images
    db_noext=str(Path(db).with_suffix(""))

    if args.text=="text":
        args.text=db_noext+"_text"
    text=args.text
    sparse=db_noext+"_sparse"
    print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
    if (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
        sys.exit(1)
    if os.path.exists(db):
        os.remove(db)
    do_system(f"colmap feature_extractor --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
    do_system(f"colmap {args.colmap_matcher}_matcher --database_path {db}")
    try:
        shutil.rmtree(sparse)
    except:
        pass
    do_system(f"mkdir {sparse}")
    do_system(f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse}")
    do_system(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
    try:
        shutil.rmtree(text)
    except:
        pass
    do_system(f"mkdir {text}")
    do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da=da/np.linalg.norm(da)
    db=db/np.linalg.norm(db)
    c=np.cross(da,db)
    denom=(np.linalg.norm(c)**2)
    t=ob-oa
    ta=np.linalg.det([t,db,c])/(denom+1e-10)
    tb=np.linalg.det([t,da,c])/(denom+1e-10)
    if ta>0:
        ta=0
    if tb>0:
        tb=0
    return (oa+ta*da+ob+tb*db)*0.5,denom



def convert(AABB_SCALE, SKIP_EARLY, IMAGE_FOLDER, TEXT_FOLDER, OUT_PATH, totp=-1, totw=-1, avglen = -1, rMat = np.array([])):
    print(f"outputting to {OUT_PATH}...")
    with open(os.path.join(TEXT_FOLDER,"cameras.txt"), "r") as f:
        angle_x=math.pi/2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            # 
            if line[0]=="#":
                continue
            els=line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w/2
            cy = h/2
            if (els[1]=="SIMPLE_RADIAL"):
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif (els[1]=="RADIAL"):
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif (els[1]=="OPENCV"):
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            elif (els[1]=="PINHOLE"):
                cx = float(els[6])
                cy = float(els[7])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x= math.atan(w/(fl_x*2))*2
            angle_y= math.atan(h/(fl_y*2))*2
            fovx=angle_x*180/math.pi
            fovy=angle_y*180/math.pi

    #print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    with open(os.path.join(TEXT_FOLDER,"images.txt"), "r") as f:
        i=0
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        out={
            "camera_angle_x":angle_x,
            "camera_angle_y":angle_y,
            "fl_x":fl_x,
            "fl_y":fl_y,
            "k1":k1,
            "k2":k2,
            "p1":p1,
            "p2":p2,
            "cx":cx,
            "cy":cy,
            "w":w,
            "h":h,
            "aabb_scale":AABB_SCALE,"frames":[]
        }

        up=np.zeros(3)
        for line in f:
            line=line.strip()
            if len(line)!=0 and line[0]=="#":
                continue
            i=i+1
            if i < SKIP_EARLY*2:
                continue
            if  i%2==1 :
                elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9 is filename
                #name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGE_FOLDER)
                name = str(f"./{image_rel}/{elems[9]}")
                if not os.path.exists(name):
                    name = name + ".png"
                    print("opening ", name)
                b=sharpness(name)
                #print(name, "sharpness=",b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                c2w[0:3,2] *= -1 # flip the y and z axis
                c2w[0:3,1] *= -1
                c2w=c2w[[1,0,2,3],:] # swap y and z
                c2w[2,:] *= -1 # flip whole world upside down

                up += c2w[0:3,1]

                #s=str(os.path.splitext(os.path.basename(elems[9]))[0])
                s=str(os.path.basename(elems[9]))
                #print("BASENAME ", s)
                name = "images/"+ s # os.path.join("images", s)
                frame={"file_path":name,"sharpness":b,"transform_matrix": c2w}
                #print("OUTPUT ", name)
                out["frames"].append(frame)
    nframes = len(out["frames"])
    if len(rMat) == 0:
        up = up / np.linalg.norm(up)
        print("rMat is None up vector was ", up)
        R=rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R=np.pad(R,[0,1])
        R[-1,-1]=1
        rMat = R
    else:
        R = rMat


    for f in out["frames"]:
        f["transform_matrix"]=np.matmul(R,f["transform_matrix"]) # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")

    print("TOTP {} TOTW {}".format(totp, totw))
    if totw < 0 :
        totw=0
        totp=[0,0,0]
        for f in out["frames"]:
            mf=f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg=g["transform_matrix"][0:3,:]
                p,w=closest_point_2_lines(mf[:,3],mf[:,2],mg[:,3],mg[:,2])
                if w>0.01:
                    totp+=p*w
                totw+=w
        if totw >0:
            totp/=totw

    print("AFTER TOTP {} TOTW {}".format(totp, totw))
    print(totp) # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3,3]-=totp

    if avglen < 0:
        avglen=0.
        for f in out["frames"]:
            avglen+=np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen/=nframes

    print("avg camera distance from origin ", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3,3]*=4./avglen     # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"]=f["transform_matrix"].tolist()
    print(nframes,"frames")
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)

    return totp, totw, avglen, rMat

def createNerf(path, hires=False):
    AABB_SCALE=int(16)
    SKIP_EARLY=int(0)
    print("Path is ", path, str(path))
    if hires:
        print("DOING HIRES !!")
        colmappath = os.path.join(os.path.join(str(path), "sibr"), "colmap")
    else:
        colmappath = os.path.join(os.path.join(str(path), "colmap_1000"), "colmap")
    TEXT_FOLDER=os.path.join(os.path.join(colmappath,  "stereo"), "sparse")
    IMAGE_FOLDER=os.path.join(os.path.join(colmappath,  "stereo"), "images")
    OUT_PATH= os.path.join(os.path.join(colmappath,  "stereo"), "transforms.json")

    totp, totw, avglen, rMat = convert(AABB_SCALE, SKIP_EARLY, IMAGE_FOLDER, TEXT_FOLDER, OUT_PATH, -1, -1, -1, np.array([]))

    colmappath = os.path.join(os.path.join(str(path), "colmap_1000"), "validation_colmap") 
    TEXT_FOLDER=os.path.join(os.path.join(colmappath,  "stereo"), "sparse")
    IMAGE_FOLDER=os.path.join(os.path.join(colmappath,  "stereo"), "images")
    OUT_PATH= os.path.join(os.path.join(colmappath,  "stereo"), "transforms.json")
    totp, totw, avglen, rMat = convert(AABB_SCALE, SKIP_EARLY, IMAGE_FOLDER, TEXT_FOLDER, OUT_PATH, totp, totw, avglen, rMat)

    colmappath = os.path.join(os.path.join(str(path), "colmap_1000"), "test_path_colmap") 
    TEXT_FOLDER=os.path.join(os.path.join(colmappath,  "stereo"), "sparse")
    IMAGE_FOLDER=os.path.join(os.path.join(colmappath,  "stereo"), "images")
    OUT_PATH= os.path.join(os.path.join(colmappath,  "stereo"), "transforms.json")
    totp, totw, avglen, rMat = convert(AABB_SCALE, SKIP_EARLY, IMAGE_FOLDER, TEXT_FOLDER, OUT_PATH, totp, totw, avglen, rMat)

    # if test2_path exists
    colmappath = os.path.join(os.path.join(str(path), "colmap_1000"), "test_path2") 

    if os.path.exists(colmappath):
        TEXT_FOLDER=os.path.join(os.path.join(colmappath,  "stereo"), "sparse")
        IMAGE_FOLDER=os.path.join(os.path.join(colmappath,  "stereo"), "images")
        OUT_PATH= os.path.join(os.path.join(colmappath,  "stereo"), "transforms.json")
        totp, totw, avglen, rMat = convert(AABB_SCALE, SKIP_EARLY, IMAGE_FOLDER, TEXT_FOLDER, OUT_PATH, totp, totw, avglen, rMat)

if __name__ == "__main__":
    args = parse_args()
    if args.video_in != "":
        run_ffmpeg(args)
    if args.run_colmap:
        run_colmap(args)
    AABB_SCALE=int(args.aabb_scale)
    SKIP_EARLY=int(args.skip_early)
    IMAGE_FOLDER=args.images
    TEXT_FOLDER=args.text
    OUT_PATH=args.out
    convert(AABB_SCALE, SKIP_EARLY, IMAGE_FOLDER, TEXT_FOLDER, OUT_PATH, -1, -1)
