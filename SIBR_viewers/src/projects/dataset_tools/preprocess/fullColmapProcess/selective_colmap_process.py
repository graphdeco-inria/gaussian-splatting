import os
import os.path
import sys
import argparse
import shutil
import sqlite3
import read_write_model as rwm


import cv2
print(cv2.__version__)


def extract_images(pathIn, pathOut, videoName, maxNumFrames = -1, resize=False):
    EVERY_NTH = 2
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    total_frames = vidcap.get(7)/EVERY_NTH
    print("FPS = ", fps)
    success,image = vidcap.read()
    success = True
    print("Extracting ", total_frames, " Frames" )
    fileNames = []
    newFolder = pathOut + "\\%s" % (videoName)
    if not os.path.exists(newFolder):
      print( "Creating: ", newFolder)
      os.makedirs(newFolder, exist_ok=True)

    for frame in range(round(total_frames)):
        # every Nth frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,(EVERY_NTH*frame))
        success,image = vidcap.read()
        if not success:
           break
        resized = image
        if resize :
            #print('Original Dimensions : ',image.shape)
            scale_percent = 52 # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        print( "Writing: frame # ", EVERY_NTH*frame, " " , pathOut + "\\%s\\frame%04d.png" % (videoName, count))     
        fileNames.append("%s\\frame%04d.png" % (videoName, count))     
        cv2.imwrite( pathOut + "\\%s\\frame%04d.png" % (videoName, count), resized)     # save frame as PNG file
 
        if maxNumFrames == count:
           break;

        count = count + 1

    return fileNames

def extract_images_with_name(imageName, images_data, new_images_data):
   add_next = False
   new_images_data.append(images_data[0])
   new_images_data.append(images_data[1])
   new_images_data.append(images_data[2])
   new_images_data.append(images_data[3])

   img_cnt = 0
   # create list with photo-only images
   for line in images_data:
       if line.split():
           if imageName in line.split()[-1]:
              new_images_data.append(line)
              img_cnt = img_cnt+1
              add_next = True
           elif add_next:
              new_images_data.append(line)
              add_next = False
           else:
              add_next = False

   return new_images_data, img_cnt

def remove_lines_from_file(fname, match, nextDel=False):
    newdata = []
    prevMatch = False
    with open(fname, 'r') as imagesfile:
        data = imagesfile.read().splitlines()
    for line in data:
        if match in line:
           if nextDel:
              prevMatch = True
        elif (not match in line) and (not prevMatch):
           newdata.append(line)
           prevMatch = False
        else:
           prevMatch = False

    # overwrite
    with open(fname, 'w') as outfile:
         for line in newdata:
            outfile.write(line + "\n")
    


def remove_video_images(path, photoName="MG_"):
   # make backups
   oldb = os.path.abspath(os.path.join(path, "colmap\\dataset.db" )) # will be modified
   backuppath = dstpath = os.path.join(path, "backups\\two_cams_all_images\\")
   if not os.path.exists(dstpath):
      os.makedirs(dstpath, exist_ok=True)

   dbfile = os.path.abspath(os.path.join(dstpath,"dataset.db"))
   oldb = os.path.abspath(os.path.join(path, "colmap\\dataset.db" )) # will be modified
   if not os.path.exists(dbfile):
       shutil.copyfile(oldb, dbfile)


   # Read images.txt & cameras.txt
   backup_images = os.path.abspath(os.path.join(dstpath,"images.txt"))
   images_fname = os.path.abspath(os.path.join(path, "colmap\\sparse\\")) + "\\images.txt"
   if not os.path.exists(backup_images):
       shutil.copyfile(images_fname, backup_images)

   backup_cameras = os.path.abspath(os.path.join(dstpath,"cameras.txt"))
   cameras_fname = os.path.abspath(os.path.join(path, "colmap\\sparse\\")) + "\\cameras.txt"
   if not os.path.exists(backup_cameras):
       shutil.copyfile(cameras_fname, backup_cameras)

   # extract photos only for images.txt
   with open(images_fname, 'r') as imagesfile:
       images_data = imagesfile.read().splitlines()

   videoDirList = []
   imagespath = os.path.abspath(os.path.join(path, "images"))
   for filename in os.listdir(imagespath):
       if "Video" in filename:
          videoDirList.append(filename)

   new_images_data = []
   new_images_data, img_cnt = extract_images_with_name(photoName, images_data, new_images_data)

   # remaining images
   print("Remaining images ", img_cnt)
   
   new_images_data[3] = ' '.join(images_data[3].split()[0:4]) + " " + str(img_cnt) +" " +  ' '.join(images_data[3].split()[5:-1] )

   # overwrite current calibration
   dstpath = os.path.abspath(os.path.join(path, "colmap\\sparse"))
   with open(dstpath+"\\images.txt", 'w') as outfile:
     for line in new_images_data:
       outfile.write(str(line) + "\n")

   outfile.close()
   print("Writing fixed images ", dstpath + "\\images.txt")
   ims = rwm.read_images_text(dstpath + "\\images.txt")
   rwm.write_images_binary(ims, dstpath + "\\images.bin")

   # fix the database

   # open the database
   db = sqlite3.connect(oldb)
   cursor = db.cursor()
   cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
   tables = cursor.fetchall()
   # debug
   delImagesQuery = """DELETE from images WHERE name LIKE '%Video%'"""
   cursor.execute(delImagesQuery)

   db.commit()

   # write out database 
   db.close()

   # create GT path directories
   dstpath = os.path.abspath(os.path.join(path, "paths_GT"))

   if not os.path.exists(dstpath):
      print("Creating ", dstpath)
      os.makedirs(dstpath)


   # backup original (distorted) video images & move undistorted images to special directories
   # then create the colmap data for each

   for currVideoName in videoDirList:
      # move the original videos to backup
      imagespath = os.path.abspath(os.path.join(path, "images"))
      shutil.move(imagespath + "\\" + currVideoName, backuppath+ "\\" + currVideoName)

      # create GT_path dir
      dstpath = os.path.abspath(os.path.join(path, "paths_GT"))
      curr_GTpath_dir = dstpath + "\\" + currVideoName
      print("Creating ", curr_GTpath_dir)
      os.makedirs(curr_GTpath_dir)
      os.makedirs(curr_GTpath_dir+"\\images")

      # move undistorted mages to GT_path dir
      imagespath = path +  "\\colmap\\stereo\\images\\"+ currVideoName
      shutil.move(imagespath, curr_GTpath_dir + "\\images")

      video_images_list = []
      video_images_list, img_cnt = extract_images_with_name(currVideoName, images_data, video_images_list)
      video_images_list[2] = ' '.join(images_data[2].split()[0:4]) + " " + str(img_cnt) +" " +  ' '.join(images_data[2].split()[5:-1] )

      # create colmap data
      dstpath = os.path.abspath(os.path.join(curr_GTpath_dir, "text"))
      os.makedirs(dstpath)

      with open(dstpath+"\\images.txt", 'w') as outfile:
         for line in video_images_list:
           outfile.write(str(line) + "\n")

      cameras_fname = os.path.abspath(os.path.join(path, "colmap\\sparse\\")) + "\\cameras.txt"
      shutil.copyfile(cameras_fname, dstpath+"\\cameras.txt")

      points_fname = os.path.abspath(os.path.join(path, "colmap\\sparse\\")) + "\\points3D.txt"
      shutil.copyfile(points_fname, dstpath+"\\points3D.txt")

      fname = os.path.abspath(os.path.join(path, "colmap\\stereo\\stereo\\fusion.cfg")) 
      remove_lines_from_file(fname, "Video", False)
      fname = os.path.abspath(os.path.join(path, "colmap\\stereo\\stereo\\patch-match.cfg")) 
      remove_lines_from_file(fname, "Video", True)
      # all done


def fix_cameras(path, photoName="MG_", sparseSubdir=""):
    if sparseSubdir == "":
        sparse_subdir = os.path.join("colmap", "sparse")
    else:
        sparse_subdir = sparseSubdir

    # Read images.txt
    images_fname = os.path.abspath(os.path.join(path, sparse_subdir)) + "\\images.txt"
    with open(images_fname, 'r') as imagesfile:
        images_data = imagesfile.read().splitlines()

    # Read cameras.txt
    cameras_fname = os.path.abspath(os.path.join(path, sparse_subdir)) + "\\cameras.txt"
    with open(cameras_fname, 'r') as camerasfile:
        cameras_data = camerasfile.read().splitlines()

    # find the first camera index for a photo
    photoCamIndex = -1
    for line in images_data:
        if line.split():
            if (photoName in line.split()[-1])  and (int(line.split()[0]) > 2):
              photoCamIndex = line.split()[0]
              print("Found Photo Camera Index ", photoCamIndex, " for camera ", line.split()[-1])
              break

    # find the first camera index for a video
    videoCamIndex = -1
    for line in images_data:
        if line.split():
            if ("Video" in line.split()[-1]) and ( int(line.split()[0]) > 2):
              videoCamIndex = line.split()[0]
              print("Found Video Camera Index ", videoCamIndex, " for camera ", line.split()[-1])
              break

    # make backups of original files
    dstpath = os.path.join(path, "backups\\")
    if not os.path.exists(dstpath):
       os.makedirs(dstpath, exist_ok=True)
    # 
    dstpath = os.path.join(path, "backups\\orig\\")
    if not os.path.exists(dstpath):
       os.makedirs(dstpath, exist_ok=True)

    # make backups of original files
    if not os.path.exists(dstpath+"\\images.txt"):
       shutil.copyfile(images_fname, dstpath +"\\images.txt")
    if not os.path.exists(dstpath+"\\cameras.txt"):
       shutil.copyfile(cameras_fname, dstpath +"\\cameras.txt")

    dbfile = os.path.abspath(os.path.join(dstpath,"dataset.db"))
    oldb = os.path.abspath(os.path.join(path, "colmap\\dataset.db" )) # will be modified
    if os.path.exists(oldb): # only do DB processing if it exists
        print("Old ", oldb, " new ", dbfile, " path ", path)
        if not os.path.exists(dbfile):
           shutil.copyfile(oldb, dbfile)

        # open the database
        db = sqlite3.connect(oldb)
        cursor = db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        # debug
        #  table = pd.read_sql_query("SELECT * from %s" % 'images', db)

        # delete all cameras except videoCamEntry and photoCamEntry from database
        delCamQuery = """DELETE from cameras WHERE camera_id != '%s' and camera_id != '%s'""" % (videoCamIndex, photoCamIndex)
        cursor.execute(delCamQuery )

        # change photo cam id to 1 and video cam id to 2
        setQuery = "UPDATE cameras SET camera_id = '%s' WHERE  camera_id = '%s'" % ("1", photoCamIndex)
        cursor.execute(setQuery)

        setQuery = "UPDATE cameras SET camera_id = '%s' WHERE  camera_id = '%s'" % ("2", videoCamIndex)
        cursor.execute(setQuery)

        # change photo cam id to 1 for all images of photos
        setQuery = "UPDATE images SET camera_id = '1' WHERE name LIKE '%MG_%'"
        cursor.execute(setQuery)

        # change video cam id to 2 for all images
        setQuery = "UPDATE images SET camera_id = '2' WHERE name LIKE '%Video%'"
        cursor.execute(setQuery)

        db.commit()

        # write out database ; next step re-exports the result to TXT and BIN
        db.close()
      
    # replace all camera indices in images.txt
    new_images_data = images_data
    cnt = 0
    for line in new_images_data:
        if line.split():
            if photoName in line.split()[-1]:
               new_images_data[cnt] = ' '.join(line.split()[0:-2]) + " 1 " + line.split()[-1] 
               print("replace ", line.split()[0],  " by 1 in ", new_images_data[cnt] )
            elif "Video" in line.split()[-1]:
               new_images_data[cnt] = ' '.join(line.split()[0:-2]) + " 2 " + line.split()[-1] 

        cnt = cnt + 1

    videoCamEntry = ""
    for line in cameras_data:
        if line.split():
            if photoCamIndex == line.split()[0]:
              photoCamEntry = "1 " + ' '.join(line.split()[1:])
            if videoCamIndex == line.split()[0]:
              videoCamEntry = "2 " + ' '.join(line.split()[1:])

    # create two element camera file one for photos one for video
    dstpath = os.path.abspath(os.path.join(path, "colmap\\sparse\\"))
    dst = dstpath + "\\cameras_two.txt"
    with open(dst, 'w') as outfile:
        outfile.write(photoCamEntry + "\n")
        if( videoCamEntry != "" ):
            outfile.write(videoCamEntry + "\n")
    outfile.close()

    # write out new file
    dst = dstpath + "\\images_two.txt"
    with open(dst, 'w') as outfile:
         for line in new_images_data:
            outfile.write(line + "\n")
    outfile.close()

    # replace files
    dstpath = os.path.abspath(os.path.join(path, "colmap\\sparse\\"))
    shutil.move(dstpath + "\\images_two.txt", images_fname)
    shutil.move(dstpath + "\\cameras_two.txt", cameras_fname)

    dstpath = os.path.abspath(os.path.join(path, "colmap\\sparse\\"))
    print("Writing cam/im binary ", dstpath + "\\cameras.bin")
    cams = rwm.read_cameras_text(dstpath + "\\cameras.txt")
    ims = rwm.read_images_text(dstpath + "\\images.txt")
    rwm.write_cameras_binary(cams, dstpath + "\\cameras.bin")
    rwm.write_images_binary(ims, dstpath + "\\images.bin")

    ptsbin = dstpath+"\\0\\points3D.bin"
    print("Pts bin ", ptsbin, " Exists " ,os.path.exists(ptsbin))
    if os.path.exists(ptsbin):
       shutil.copyfile(ptsbin, dstpath+"\\points3D.bin")

    # overwrite 0 as well
    dstpath = os.path.abspath(os.path.join(path, "colmap\\sparse\\0\\"))
    print("Writing cam/im binary ", dstpath + "\\cameras.bin")
    rwm.write_cameras_binary(cams, dstpath + "\\cameras.bin")
    rwm.write_images_binary(ims, dstpath + "\\images.bin")
    rwm.write_cameras_text(cams, dstpath + "\\cameras.txt")
    rwm.write_images_text(ims, dstpath + "\\images.txt")

    return True

def extract_video_frames(pathIn, pathOut):
    cnt = 0
    fileNames = []
    for filename in os.listdir(pathIn):
      if ("MP4" in filename) or ("mp4" in filename):
        with open(os.path.join(pathIn, filename), 'r') as f:
          print("Extracting Video from File: ", f.name)
#          fileNames  = fileNames + extract_images(f.name, pathOut, "Video%d" % cnt, maxNumFrames=3, resize=True)
          fileNames  = fileNames + extract_images(f.name, pathOut, "Video%d" % cnt, resize=True)
#          extract_images(f.name, pathOut, videoName="Video%d" % cnt)
          cnt = cnt+1

    with open(os.path.dirname(pathIn) + "\\videos\\Video_Frames.txt", 'w') as f:
       for item in fileNames:
          f.write("%s\n" % item.replace("\\", "/"))
       f.close()

