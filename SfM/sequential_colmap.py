import os
from tqdm import tqdm
from natsort import natsorted
def process(proj_path, img_path):
    sparse_path = os.path.join(proj_path,'sparse')
    os.system(f'COLMAP.bat feature_extractor --database_path {proj_path}/ddd.db --image_path {img_path} --ImageReader.camera_model PINHOLE')
    os.system(f'COLMAP.bat exhaustive_matcher --database_path {proj_path}/ddd.db')
    os.makedirs(sparse_path,exist_ok=True)
    os.system(f'COLMAP.bat mapper --database_path {proj_path}/ddd.db --image_path {img_path} --output_path {sparse_path}')

data_path = '.\816'
print(data_path)
ll = natsorted(os.listdir(data_path))

for i in tqdm(range(0,37)):
    i = '{:0>2d}'.format(i)
    proj_path = os.path.join(data_path, f'colmap_{i}')
    print(proj_path)
    if not os.path.exists(proj_path):
        os.makedirs(proj_path)
    img_path = os.path.join(data_path, f'{i}')
    process(proj_path,img_path)