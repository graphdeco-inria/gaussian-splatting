# python train.py --source_path ../../Dataset/3DGS_Dataset/linggongtang --model_path output/linggongtang --data_device 'cpu' --eval --resolution 1
# scene: {'kejiguan': 'cuda', 'wanfota': 'cuda', 'zhiwu': 'cuda', 'linggongtang': 'cuda', 'xiangjiadang': 'cuda', 'town-train-cpy': 'cuda', 'town2-train-cpy': 'cuda', 'sipingguzhai': 'cpu'}
# device = cuda: 科技馆、万佛塔、植物
#        = cpu:  凌公塘、湘家荡、寺平古宅

import os

# for idx, scene in enumerate({'town-train': 'cuda', 'town2-train': 'cuda', 'building1-train': 'cuda'}.items()):
#     print('---------------------------------------------------------------------------------')
#     one_cmd = f'python train.py --source_path /data2/lpl/data/carla-dataset/{scene[0]} --model_path output/{scene[0]} --data_device "{scene[1]}" --resolution 1 --checkpoint_iterations 30000'
#     print(one_cmd)
#     os.system(one_cmd)
#
# # python render.py -m <path to trained model>
# for idx, scene in enumerate(['town-train-cpy', 'town2-train-cpy', 'building1-train']):
#     print('---------------------------------------------------------------------------------')
#     one_cmd = f'python render.py -m output/{scene}'
#     print(one_cmd)
#     os.system(one_cmd)
#
# # python metrics.py -m <path to trained model>
# for idx, scene in enumerate(['town-train-cpy', 'town2-train-cpy', 'building1-train']):
#     print('---------------------------------------------------------------------------------')
#     one_cmd = f'python metrics.py -m output/{scene}'
#     print(one_cmd)
#     os.system(one_cmd)

for idx, scene in enumerate({'building2-train': 'cpu',  'building3-train': 'cuda'}.items()):
    print('---------------------------------------------------------------------------------')
    one_cmd = f'python train.py --source_path /data2/lpl/data/carla-dataset/{scene[0]} --model_path output/{scene[0]} --data_device "{scene[1]}" --resolution 1 --checkpoint_iterations 30000 --port 6009'
    print(one_cmd)
    os.system(one_cmd)

# python render.py -m <path to trained model>
for idx, scene in enumerate(['building2-train', 'building3-train']):
    print('---------------------------------------------------------------------------------')
    one_cmd = f'python render.py -m output/{scene}'
    print(one_cmd)
    os.system(one_cmd)

# python metrics.py -m <path to trained model>
for idx, scene in enumerate(['building2-train', 'building3-train']):
    print('---------------------------------------------------------------------------------')
    one_cmd = f'python metrics.py -m output/{scene}'
    print(one_cmd)
    os.system(one_cmd)