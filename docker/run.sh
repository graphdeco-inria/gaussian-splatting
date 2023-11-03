#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


#cd ..

mkdir -p $1-data/input
mkdir -p $1-data/output

FPS="5" #<-change this to change framerate


if [ -f $1-data/input/0001.jpg ]
then
  echo "File $1 appears to have already been split .."
else
  ffmpeg -i $1 -qscale:v 1 -qmin 1 -vf fps=$FPS $1-data/input/%04d.jpg
fi


python3.10 convert.py -s $1-data/ --camera SIMPLE_RADIAL --no_gpu #GPU produces worse results (?)
python3.10 train.py -s $1-data/ -r 1 --model_path=$1-data/output/ --position_lr_init 0.000016 --scaling_lr 0.001 --iterations 35000 #Test more training budget

python3.10 3dgsconverter/3dgsconverter.py -i $1-data/output/point_cloud/iteration_30000/point_cloud.ply -o $1-data/output/point_cloud/iteration_30000/output_cc.ply -f cc --rgb --density_filter --remove_flyers
python3.10 3dgsconverter/3dgsconverter.py -i $1-data/output/point_cloud/iteration_30000/output_cc.ply  -o $1-data/output/point_cloud/iteration_30000/point_cloud_clean.ply -f 3dgs

#pack it in
#tar cvfjh "$1.tar.bz2" $1-data/


exit 0
