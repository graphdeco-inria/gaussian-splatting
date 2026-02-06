#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
from scene import Scene
import numpy as np
from PIL import Image
import io
import wandb

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def get_effective_rank(scale, temp=1):
    D = (scale*scale)**(1/temp)
    _sum = D.sum(dim=1, keepdim=True)
    pD = D / _sum
    try:
        entropy = -torch.sum(pD*torch.log(pD), dim=1)
        erank = torch.exp(entropy)
    except Exception as e:
        print(e)
        pass
    return erank

def get_volume(scale):
    V = scale[:,0]*scale[:,1]*scale[:,2]
    return V

def get_ordered_scale_multiple(scale):
    ordered_scale, _ = torch.sort(scale, descending=True)
    ordered_scale_multiple = ordered_scale / ordered_scale[:,2:3]
    return ordered_scale_multiple, ordered_scale

def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad


def get_histograms(log_dict, iteration, scene, erank, volume, opacity, ordered_scale_multiple):
    erank_np = erank.detach().cpu().numpy()
    volume_np = volume.detach().cpu().numpy()
    opacity_np = opacity.detach().cpu().numpy()
    ordered_scale_multiple_np = ordered_scale_multiple.detach().cpu().numpy()

    # erank histogram
    plt.clf()
    mean_erank = torch.mean(erank).item()
    #print(f'iteration: {iteration}  erank: {mean_erank}')
    fig, ax1 = plt.subplots(figsize=(8,6))

    plt.ylim(0, 30000)
    plt.xticks([1, 1.5, 2.0, 2.5, 3.0])  # Specify x-axis tick marks
    plt.yticks([5000,15000,25000])
    ax1.tick_params(axis='both', which='major', labelsize=24)

    cmap=plt.cm.Greens
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0., vmax=3.0)
    #ax1.hist(erank_np, bins=50, range=(1.,3.), color='green', alpha=0.5)  # You can adjust the number of bins to your preference
    n,bins,patches=ax1.hist(erank_np, bins=50, range=(1.,3.), alpha=0.5)  # You can adjust the number of bins to your preference
    for patch, value in zip(patches, bins):
        color = cmap(norm(value))
        patch.set_facecolor(color)

    ax1.text(0.55, 0.9, f'iteration: {iteration}', fontsize=20,transform=plt.gca().transAxes)
    ax1.text(0.65, 0.77, f'total: {len(scene.gaussians.get_xyz)}', fontsize=20, transform=plt.gca().transAxes)
    ax1.text(0.55, 0.83, f'mean: {mean_erank:.3f}', fontsize=20, transform=plt.gca().transAxes)

    #ax1.set_xlabel('effective rank', fontsize=24)
    #ax1.set_ylabel('count', fontsize=24)
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.9)


    erank_bin_edges = np.histogram_bin_edges(erank_np, bins=50, range=(1.,3.))
    # avg_volume_erank
    average_volume_erank = []
    average_opacity_erank = []
    erank_num = []
    for i in range(50):
        if i == 49:
            erank_ind = np.where((erank_np >= erank_bin_edges[i]) & (erank_np <= erank_bin_edges[i+1]))[0]
        else:
            erank_ind = np.where((erank_np >= erank_bin_edges[i]) & (erank_np < erank_bin_edges[i+1]))[0]
        erank_num.append(len(erank_ind))
        volume_in_erank_bin = [volume_np[idx] for idx in erank_ind]
        opacity_in_erank_bin = [opacity_np[idx] for idx in erank_ind]
        _average_volume_erank = np.mean(volume_in_erank_bin)
        _average_opacity_erank = np.mean(opacity_in_erank_bin)
        average_volume_erank.append(_average_volume_erank)
        average_opacity_erank.append(_average_opacity_erank)
    average_volume_erank = np.array(average_volume_erank)
    average_opacity_erank = np.array(average_opacity_erank)
    average_volume_erank[np.isnan(average_volume_erank)] = 0
    average_opacity_erank[np.isnan(average_opacity_erank)] = 0

    smoothed_volume_erank = []
    smoothed_opacity_erank = []
    for i in range(50):
        if i == 0:
            smoothed_volume_erank.append(average_volume_erank[i])
            smoothed_opacity_erank.append(average_opacity_erank[i])
        elif i ==1:
            if erank_num[i] + erank_num[i-1] != 0:
                sv = (average_volume_erank[i]*erank_num[i] + average_volume_erank[i-1]*erank_num[i-1]) / (erank_num[i] + erank_num[i-1])
                so = (average_opacity_erank[i]*erank_num[i] + average_opacity_erank[i-1]*erank_num[i-1]) / (erank_num[i] + erank_num[i-1])
            else:
                sv = so = 0
            smoothed_volume_erank.append(sv)
            smoothed_opacity_erank.append(so)

        else:
            if erank_num[i] + erank_num[i-1] + erank_num[i-2] != 0:
                sv = (average_volume_erank[i]*erank_num[i] + average_volume_erank[i-1]*erank_num[i-1] + average_volume_erank[i-2]*erank_num[i-2]) / (erank_num[i] + erank_num[i-1] + erank_num[i-2])
                so = (average_opacity_erank[i]*erank_num[i] + average_opacity_erank[i-1]*erank_num[i-1] + average_opacity_erank[i-2]*erank_num[i-2]) / (erank_num[i] + erank_num[i-1] + erank_num[i-2])
            else:
                sv = so = 0
            smoothed_volume_erank.append(sv)
            smoothed_opacity_erank.append(so)



    os.makedirs(os.path.join(scene.model_path, 'stats', 'eranks'), exist_ok=True)
    plt.savefig(os.path.join(scene.model_path, 'stats', 'eranks', f'{iteration:05}.png'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    erank_array = np.array(Image.open(buf))
    # wandb
    erank_histogram = torch.tensor(erank_array[:,:,:3]/255.).permute(2,0,1)
    log_dict['erank_histogram'] = wandb.Image(erank_histogram)


    ########################
    # opacity histogram
    ########################
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(8,6))
    plt.xticks([0.,0.2,0.4,0.6,0.8,1.0])  # Specify x-axis tick marks
    plt.ylim(0, 30000)

    ax1.hist(opacity_np, bins=50, range=(0.,1.), color='green', alpha=0.5)  # You can adjust the number of bins to your preference
    title_dir = os.path.basename(os.path.dirname(scene.model_path))
    title_base = os.path.basename(scene.model_path)
    plt.title(f"{title_dir}/{title_base}", loc='center')
    # plt.title(f"{scene.model_path.split('/')[-2]}/{scene.model_path.split('/')[-1]}", loc='center')

    ax1.text(0.65, 0.9, f'iter: {iteration}', fontsize=12,transform=plt.gca().transAxes)
    ax1.text(0.65, 0.83, f'# gaussians: {len(scene.gaussians.get_xyz)}', fontsize=12, transform=plt.gca().transAxes)
    ax1.text(0.65, 0.77, f'mean_erank: {mean_erank:.3f}', fontsize=12, transform=plt.gca().transAxes)

    ax1.set_xlabel('opacity')
    ax1.set_ylabel('opacity hist', color='green')

    opacity_bin_edges = np.histogram_bin_edges(opacity_np, bins=50, range=(0.,1.))
    average_volume_opacity = []
    average_erank_opacity = []
    opacity_num = []

    for i in range(50):
        if i == 49:
            opacity_ind = np.where((opacity_np >= opacity_bin_edges[i]) & (opacity_np <= opacity_bin_edges[i+1]))[0]
        else:
            opacity_ind = np.where((opacity_np >= opacity_bin_edges[i]) & (opacity_np < opacity_bin_edges[i+1]))[0]
        opacity_num.append(len(opacity_ind))
        volume_in_opacity_bin = [volume_np[idx] for idx in opacity_ind]
        erank_in_opacity_bin = [erank_np[idx] for idx in opacity_ind]
        _average_volume_opacity = np.mean(volume_in_opacity_bin)
        _average_erank_opacity = np.mean(erank_in_opacity_bin)
        average_volume_opacity.append(_average_volume_opacity)
        average_erank_opacity.append(_average_erank_opacity)
    average_volume_opacity = np.array(average_volume_opacity)
    average_erank_opacity = np.array(average_erank_opacity)
    average_volume_opacity[np.isnan(average_volume_opacity)] = 0
    average_erank_opacity[np.isnan(average_erank_opacity)] = 0

    smoothed_volume_opacity = []
    smoothed_erank_opacity = []
    for i in range(50):
        if i == 0:
            smoothed_volume_opacity.append(average_volume_opacity[i])
            smoothed_erank_opacity.append(average_erank_opacity[i])
        elif i == 1:
            if opacity_num[i] + opacity_num[i-1] != 0:
                svo = (average_volume_opacity[i]*opacity_num[i] + average_volume_opacity[i-1]*opacity_num[i-1]) / (opacity_num[i]+opacity_num[i-1])
                seo = (average_erank_opacity[i]*opacity_num[i] + average_erank_opacity[i-1]*opacity_num[i-1]) / (opacity_num[i]+opacity_num[i-1])
            else:
                svo = seo = 0
            smoothed_volume_opacity.append(svo)
            smoothed_erank_opacity.append(seo)
        else:
            if opacity_num[i] + opacity_num[i-1] + opacity_num[i-2] != 0:
                svo = (average_volume_opacity[i]*opacity_num[i] + average_volume_opacity[i-1]*opacity_num[i-1] + average_volume_opacity[i-2]*opacity_num[i-2]) / (opacity_num[i]+opacity_num[i-1]+opacity_num[i-2])
                seo = (average_erank_opacity[i]*opacity_num[i] + average_erank_opacity[i-1]*opacity_num[i-1] + average_erank_opacity[i-2]*opacity_num[i-2]) / (opacity_num[i]+opacity_num[i-1]+opacity_num[i-2])
            else:
                svo = seo = 0
            smoothed_volume_opacity.append(svo)
            smoothed_erank_opacity.append(seo)

    ax2 = ax1.twinx()
    ax2.set_ylabel('volume', color='red')
    ax2.set_ylim([0, 1e-6])

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('erank', color='purple')
    ax3.set_ylim([1,3.])

    x_values = []
    for i in range(len(opacity_bin_edges)-1):
        x_values.append((opacity_bin_edges[i]+opacity_bin_edges[i+1])/2.)

    ax2.plot(x_values, smoothed_volume_opacity, marker='o', linestyle='--', color='red', alpha=0.5)
    ax3.plot(x_values, smoothed_erank_opacity, marker='o', linestyle='-', color='purple',alpha=0.5)

    plt.subplots_adjust(right=0.8)

    os.makedirs(os.path.join(scene.model_path, 'stats', 'opacity'), exist_ok=True)
    plt.savefig(os.path.join(scene.model_path, 'stats', 'opacity', f'{iteration:05}.png'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    opacity_histogram = np.array(Image.open(buf))

    #wandb
    opacity_histogram = torch.tensor(opacity_histogram[:,:,:3]/255.).permute(2,0,1)
    log_dict['opacity_histogram'] = wandb.Image(opacity_histogram)


    ########################
    # ordered_scale_multiple
    ########################
    plt.clf()
    fig, axs = plt.subplots(1,2,figsize=(12,5))

    x_mult = np.clip(ordered_scale_multiple_np[:,0], a_min=0, a_max=100)
    y_mult = np.clip(ordered_scale_multiple_np[:,1], a_min=0, a_max=100)
    mappable = axs[0].hist2d(x_mult, y_mult, bins=50, cmap='BuPu', range=[[1,100],[1,100]])


    # Add labels and title
    axs[0].set_xlabel('1st / 3rd')
    axs[0].set_ylabel('2nd / 3rd')
    axs[0].set_title('scale multiplier w.r.t. 3rd scale')

    axs[0].text(0.2, 0.9, f'iter: {iteration}', fontsize=8,transform=plt.gca().transAxes)
    axs[0].text(0.2, 0.83, f'# gaussians: {len(scene.gaussians.get_xyz)}', fontsize=8, transform=plt.gca().transAxes)
    axs[0].text(0.2, 0.77, f'mean_erank: {mean_erank:.3f}', fontsize=8, transform=plt.gca().transAxes)

    # Add color bar
    fig.colorbar(mappable[3], ax=axs[0])

    mult_12 = np.clip(ordered_scale_multiple_np[:,0] / ordered_scale_multiple_np[:,1], a_min=0, a_max=100)
    axs[1].hist(mult_12, bins=50, range=[1,100])
    axs[1].set_xlabel('1st / 2nd')
    axs[1].set_title('scale multiplier w.r.t. 2nd scale')

    os.makedirs(os.path.join(scene.model_path, 'stats', 'scale'), exist_ok=True)
    plt.savefig(os.path.join(scene.model_path, 'stats', 'scale', f'{iteration:05}.png'))
    plt.close()


    return log_dict

def generate_histograms(dataset: ModelParams, iteration: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        print(f"\nLoaded model:")
        print(f"  path = {scene.model_path}")
        print(f"  iteration = {iteration}")

        scale = gaussians.get_scaling
        erank = get_effective_rank(scale)
        volume = get_volume(scale)
        ordered_scale_multiple, _ = get_ordered_scale_multiple(scale)
        opacity = gaussians.get_opacity

        get_histograms({}, iteration, scene, erank, volume, opacity, ordered_scale_multiple)

        print("\nHistograms saved to:")
        print(f"  {scene.model_path}/stats/eranks/{iteration:05}.png")
        print(f"  {scene.model_path}/stats/opacity/{iteration:05}.png")
        print(f"  {scene.model_path}/stats/scale/{iteration:05}.png\n")

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate histogram for pretrained 3DGS model")

    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iterations", default=-1, type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)

    generate_histograms(model.extract(args), args.iterations)
