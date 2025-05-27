# copy from street GS script/waymo/generate_sky_mask.py

import os, sys

sys.path.append(os.getcwd())
import argparse
import os
import copy
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from termcolor import colored
from glob import glob

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

def setup(args):
    # ======================== Load Grounding DINO model ========================
    print(colored('Load Grounding DINO model', 'green'))
    def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file) 
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model   

    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    global groundingdino_model 
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    # ======================== Load Segment Anything model ========================
    print(colored('Load SAM model', 'green'))
    # sam_checkpoint = '/nas/home/yanyunzhi/segment-anything/sam_vit_h_4b8939.pth.1'
    sam = build_sam(checkpoint=args.sam_checkpoint)
    sam.cuda()
    global sam_predictor
    sam_predictor = SamPredictor(sam)

image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])

def add_to_mask_dict(masks_dict, mask_path):
    basename = os.path.basename(mask_path)
    cam = image_filename_to_cam(basename)
    frame = image_filename_to_frame(basename)
    mask = cv2.imread(mask_path) 
    if frame not in masks_dict:
        masks_dict[frame] = [None] * 3 # FRONT_LEFT, FRONT, FRONT_RIGHT 1, 0, 2
    if cam == 1:
        masks_dict[frame][0] = mask
    elif cam == 0:
        masks_dict[frame][1] = mask
    elif cam == 2:
        masks_dict[frame][2] = mask
    


def segment_with_text_prompt(datadir, BOX_TRESHOLD, TEXT_TRESHOLD, ignore_exists):
    save_dir = os.path.join(datadir, 'sky_mask')
    os.makedirs(save_dir, exist_ok=True)

    image_dir = os.path.join(datadir, 'images')
    image_files = glob(image_dir + "/*.jpg") 
    image_files += glob(image_dir + "/*.png")
    image_files = sorted(image_files)
    
    masks_dict = dict()
    for image_path in tqdm(image_files):
        image_base_name = os.path.basename(image_path)
        output_mask = os.path.join(save_dir, image_base_name)
                        
        if os.path.exists(output_mask) and ignore_exists:
            add_to_mask_dict(masks_dict, output_mask)
            print(f'{output_mask} exists, skip')
            continue
        
        cam = image_filename_to_cam(image_base_name)
        box_threshold = BOX_TRESHOLD[cam]
        
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption='sky', 
            box_threshold=box_threshold, 
            text_threshold=TEXT_TRESHOLD
        )

        print(f'detecting {boxes.shape[0]} boxed of sky in {image_path}, box_threshold: {box_threshold}, logits: {logits}')
        if boxes.shape[0] != 0:
            H, W, _ = image_source.shape
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
            # assume that the box prompt for sky should be close to the top edge of the image
            #  --------------  top edge 
            # | x ----       |
            # | |    |       |
            # |  ----x       |
            #  --------------
            boxes_mask = boxes_xyxy[:, 1] < 100 # 100 pixels
            boxes_xyxy = boxes_xyxy[boxes_mask]
        else:
            boxes_xyxy = []
        
        num_boxes = len(boxes_xyxy)

        if num_boxes == 0:                
            mask = np.zeros_like(image_source[..., 0])
        else:
            sam_predictor.set_image(image_source)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
            masks, _, _ = sam_predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes,
                        multimask_output = False,
                    )
            
            torch.cuda.empty_cache()
            mask_final = torch.zeros_like(masks[0, 0]).bool()
            for mask in masks[:, 0]:
                mask_final = mask_final | mask.bool()
                
            mask = mask_final.cpu().numpy()
            
        cv2.imwrite(output_mask, mask * 255)
        add_to_mask_dict(masks_dict, output_mask)

    print('saving sky mask video')
    masks_dict = dict(sorted(masks_dict.items(), key=lambda x: x[0]))
    merge_masks = []
    for frame, masks in masks_dict.items():
        merge_mask = np.concatenate(masks, axis=1)
        text = f'frame: {frame}'
        cv2.putText(merge_mask, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2) 
        merge_masks.append(merge_mask)
    merge_masks_path = os.path.join(save_dir, 'mask.mp4')
    imageio.mimwrite(merge_masks_path, merge_masks, fps=24)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, type=str)
    parser.add_argument('--box_threshold', nargs='+', type=float, default=[0.3]) # Change this to your threshold
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--ignore_exists", action='store_true')
    parser.add_argument("--sam_checkpoint", type=str)
    
    args = parser.parse_args()
    setup(args)

    assert isinstance(args.box_threshold, list)
    if len(args.box_threshold) == 1:
        box_threshold = [args.box_threshold[0]] * 5
    else:
        assert len(args.box_threshold) == 5
        box_threshold = args.box_threshold
    print('box_threshold: ', box_threshold)
        
    segment_with_text_prompt(
        datadir=args.datadir, 
        BOX_TRESHOLD=box_threshold,
        TEXT_TRESHOLD=args.text_threshold,
        ignore_exists=args.ignore_exists,
    )