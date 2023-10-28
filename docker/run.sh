#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


cd ..

mkdir -p $1-data/input
mkdir -p $1-data/output

FPS="5" #<-change this to change framerate


if [ -f $1-data/input/0001.jpg ]
then
  echo "File $1 appears to have already been split .."
else
  ffmpeg -i $1 -qscale:v 1 -qmin 1 -vf fps=$FPS $1-data/input/%04d.jpg
fi


python3.10 convert.py -s $1-data/ #--no_gpu
python3.10 train.py -s $1-data/ -r 1 --model_path=$1-data/output/

exit 0
