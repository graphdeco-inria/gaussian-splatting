from utils.scheduler_utils import ImageClustering, GroupScheduler
import os
import torch
import sys
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


if __name__ == "__main__":

    clustering = ImageClustering("/mnt/disk2/360/bicycle/sparse/0")
    scheduler = GroupScheduler(None, clustering.ordered_cluster_names,
                               densify_until_iter = 15000,
                               densify_from_iter = 500,
                               debug = True)
    
    for iteration in range(1, 30000+1):
        scheduler.scheduled_training_index(iteration)
        if scheduler.densify_and_prune_flag:
            print("densify_and_prune")
            scheduler.densify_and_prune_flag = False
        if scheduler.reset_opacity_flag:
            print("reset_opacity")
            scheduler.reset_opacity_flag = False