import os
from tqdm import tqdm
from natsort import natsorted
def process(proj_path, img_path):
    sparse_path = os.path.join(proj_path,'sparse')
    os.system(f'colmap feature_extractor --database_path {proj_path}/ddd.db --image_path {img_path} --ImageReader.camera_model PINHOLE')
    os.system(f'colmap exhaustive_matcher --database_path {proj_path}/ddd.db')
    os.makedirs(sparse_path,exist_ok=True)
    os.system(f'colmap mapper --database_path {proj_path}/ddd.db --image_path {img_path} --output_path {sparse_path}')

data_path = './img_822_sm'
print(data_path)
ll = natsorted(os.listdir(data_path))

for i in tqdm(range(35)):
    i = '{:0>2d}'.format(i)
    proj_path = os.path.join(data_path, f'colmap_{i}')
    print(proj_path)
    if not os.path.exists(proj_path):
        os.makedirs(proj_path)
    #img_path = os.path.join(data_path, f'{i}')
    img_path = os.path.join(proj_path, 'images')
    process(proj_path,img_path)