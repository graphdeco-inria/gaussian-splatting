# python train.py --source_path ../../Dataset/3DGS_Dataset/linggongtang --model_path output/linggongtang --data_device 'cpu' --eval --resolution 1
# scene: {'kejiguan': 'cuda', 'wanfota': 'cuda', 'zhiwu': 'cuda', 'linggongtang': 'cpu', 'xiangjiadang': 'cpu', 'sipingguzhai': 'cpu'}
# device = cuda: 科技馆、万佛塔、植物
#        = cpu:  凌公塘、湘家荡、寺平古宅

import os

for cuda, scene in enumerate({'linggongtang': 'cpu', 'xiangjiadang': 'cpu', 'sipingguzhai': 'cpu'}.items()):
    print('---------------------------------------------------------------------------------')
    one_cmd = f'python train.py --source_path ../../Dataset/3DGS_Dataset/{scene[0]} --model_path output/{scene[0]} --data_device "{scene[1]}" --resolution 1 --eval'
    os.system(one_cmd)