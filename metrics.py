#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from DISTS_pytorch import DISTS
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from wasserstein_distortion import VGG16WassersteinDistortion
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    D = DISTS().cuda()

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                print("check point")
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                print("check point")

                ssims = []
                psnrs = []
                lpipss = []
                dists =[]
                wloss_pytorch = VGG16WassersteinDistortion().to("cuda")
                wd_s2_l2 =[]
                wd_s3_l2=[]
                wd_s2_l3=[]
                wd_s3_l3=[]
                log2_sigma=[2,2,3,3]
                scale=[2,3,2,3]
                log2_scale_pair=list(zip(log2_sigma,scale))
                wloss_pytorch.eval()
                with torch.no_grad():
                    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                        dists.append(D(renders[idx], gts[idx]).item())
                        for (log2, scale) in log2_scale_pair:
                            log2sigma=log2
                            image_4d = renders[idx].to("cuda", non_blocking=True)
                            gt_image_4d = gts[idx].to("cuda", non_blocking=True)
                            log2_sigma = (torch.zeros_like(image_4d[:, 0:1, ...]) + log2sigma).to("cuda", non_blocking=True)
                            if log2==2 and scale==2:
                                wd_s2_l2.append(wloss_pytorch(image_4d, gt_image_4d, log2_sigma, num_scales=scale).item())
                            elif log2==2 and scale==3:
                                wd_s3_l2.append(wloss_pytorch(image_4d, gt_image_4d, log2_sigma, num_scales=scale).item())
                            elif log2==3 and scale==2:
                                wd_s2_l3.append(wloss_pytorch(image_4d, gt_image_4d, log2_sigma, num_scales=scale).item())
                            elif log2==3 and scale==3:
                                wd_s3_l3.append(wloss_pytorch(image_4d, gt_image_4d, log2_sigma, num_scales=scale).item())


                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  DISTS: {:>12.7f}".format(torch.tensor(dists).mean(), ".5"))
                print("  WD s2 l2: {:>12.7f}".format(torch.tensor(wd_s2_l2).mean(), ".5"))
                print("  WD s3 l2: {:>12.7f}".format(torch.tensor(wd_s3_l2).mean(), ".5"))
                print("  WD s2 l3: {:>12.7f}".format(torch.tensor(wd_s2_l3).mean(), ".5"))
                print("  WD s3 l3: {:>12.7f}".format(torch.tensor(wd_s3_l3).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "DISTS": torch.tensor(dists).mean().item(),
                                                        "WD_s2_l2": torch.tensor(wd_s2_l2).mean().item(),
                                                        "WD_s3_l2": torch.tensor(wd_s3_l2).mean().item(),
                                                        "WD_s2_l3": torch.tensor(wd_s2_l3).mean().item(),
                                                        "WD_s3_l3": torch.tensor(wd_s3_l3).mean().item()
                                                        })
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "DISTS": {name: d for d, name in zip(torch.tensor(dists).tolist(), image_names)},
                                                            "WD_s2_l2": {name: w for w, name in zip(torch.tensor(wd_s2_l2).tolist(), image_names)},
                                                            "WD_s3_l2": {name: w for w, name in zip(torch.tensor(wd_s3_l2).tolist(), image_names)},
                                                            "WD_s2_l3": {name: w for w, name in zip(torch.tensor(wd_s2_l3).tolist(), image_names)},
                                                            "WD_s3_l3": {name: w for w, name in zip(torch.tensor(wd_s3_l3).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
