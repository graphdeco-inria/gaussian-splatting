# python train.py --source_path ../../Dataset/3DGS_Dataset/凌公塘 --model_path output/linggongtang --eval --resolution 1
# scene: {'科技馆': 'kejiguan', '万佛塔': 'wanfota', '植物': 'zhiwu', '凌公塘': 'linggongtang', '湘家荡': 'xiangjiadang', '寺平古宅': 'sipingguzhai'}
#      : {'科技馆': ['kejiguan', 'cuda'], '万佛塔': ['wanfota', 'cuda'], '植物': ['zhiwu', 'cuda'], '凌公塘': ['linggongtang', 'cpu'], '湘家荡': ['xiangjiadang', 'cpu'], '寺平古宅': ['sipingguzhai', 'cpu']}
# device = cuda: 科技馆、万佛塔、植物
#        = cpu:  凌公塘、湘家荡、寺平古宅

import os

for cuda, scene in enumerate({'科技馆': ['kejiguan', 'cuda'], '湘家荡': ['xiangjiadang', 'cpu'], '凌公塘': ['linggongtang', 'cpu'], '寺平古宅': ['sipingguzhai', 'cpu'],}.items()):
    print('---------------------------------------------------------------------------------')
    one_cmd = f'python train.py --source_path ../../Dataset/3DGS_Dataset/{scene[0]} --model_path output/{scene[1][0]} --data_device "{scene[1][1]}" --resolution 1 --eval'
    os.system(one_cmd)