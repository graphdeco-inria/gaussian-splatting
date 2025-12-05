#!/bin/bash

set -e 

# for dataset in bicycle bonsai counter garden kitchen stump # train truck
# do
#     python3 train_l2.py -s ../datasets/${dataset} --start_checkpoint saved_output/chkpnt_${dataset}/chkpnt15000.pth --iterations 37000 \
#         --test_iterations 15000 16000 17000 18000 19000 20000 21000 22000 23000 \
#                           24000 25000 26000 27000 28000 29000 30000 31000 32000 \
#                           33000 34000 35000 36000 37000 --eval
# 
#     # Find last dir in output/
#     output_dir=$(ls -td output/*/ | head -1)
#     mv ${output_dir} saved_output/adam-${dataset}-huber
# done

# for dataset in train truck
# do
#     python3 train_l2.py -s ../datasets/tandt/${dataset} --start_checkpoint saved_output/chkpnt_${dataset}/chkpnt15000.pth --iterations 37000 \
#         --test_iterations 15000 16000 17000 18000 19000 20000 21000 22000 23000 \
#                           24000 25000 26000 27000 28000 29000 30000 31000 32000 \
#                           33000 34000 35000 36000 37000 --eval
# 
#     # Find last dir in output/
#     output_dir=$(ls -td output/*/ | head -1)
#     mv ${output_dir} saved_output/adam-${dataset}-huber
# done


for dataset in train truck
do
    python3 train_l2.py -s ../datasets/tandt/${dataset} --start_checkpoint saved_output/chkpnt_${dataset}/chkpnt15000.pth --iterations 37000 \
        --reset_optimizer \
        --test_iterations 15000 16000 17000 18000 19000 20000 21000 22000 23000 \
                          24000 25000 26000 27000 28000 29000 30000 31000 32000 \
                          33000 34000 35000 36000 37000 --eval

    # Find last dir in output/
    output_dir=$(ls -td output/*/ | head -1)
    mv ${output_dir} saved_output/adam-${dataset}-huber-reset
done

for dataset in bicycle bonsai counter garden kitchen stump # train truck
do
    python3 train_l2.py -s ../datasets/${dataset} --start_checkpoint saved_output/chkpnt_${dataset}/chkpnt15000.pth --iterations 37000 \
        --reset_optimizer \
        --test_iterations 15000 16000 17000 18000 19000 20000 21000 22000 23000 \
                          24000 25000 26000 27000 28000 29000 30000 31000 32000 \
                          33000 34000 35000 36000 37000 --eval

    # Find last dir in output/
    output_dir=$(ls -td output/*/ | head -1)
    mv ${output_dir} saved_output/adam-${dataset}-huber-reset
done
