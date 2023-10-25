#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


cd ..

mkdir -p $1-data/input
mkdir -p $1-data/output

ffmpeg -i $1 -qscale:v 1 -qmin 1 -vf fps=5 $1-data/input/%04d.jpg


python3.10 convert.py -s $1-data/ --no_gpu
python3.10  train.py -s $1-data/ -r 1 --model_path=$1-data/output/

exit 0
