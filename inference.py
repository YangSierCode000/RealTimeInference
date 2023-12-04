from enum import unique
import gc
import argparse
from posixpath import dirname
from os.path import join, exists
import os
from datetime import datetime
import pandas as pd
import copy
from tqdm import tqdm
import gin
import torch
import torch.nn.functional as F
import torchmetrics
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import numpy as np
import wandb
from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.models import get_model
from src.data import get_data_module
from src.utils.metric import per_class_iou
import src.data.transforms as T
from src.mbox import com
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# TODO not done yet, currently achieve plotting functions

@gin.configurable
def inference(data_module_name, model_name):
    data_module = get_data_module(data_module_name)()
    data_module.setup("val")
    val_loader = data_module.val_dataloader()
    for i, batch in enumerate(val_loader):
        if i == 1:
            break
        print("Log: number of points:", batch["coordinates"].shape)
        plot_pc(batch["coordinates"][batch["labels"] != 255], batch["labels"][batch["labels"] != 255])
        create_ply_file(np.array(batch["coordinates"]), "point_cloud.ply")

def create_ply_file(coordinates, file_path):
    """
    Creates a PLY file from a numpy array of XYZ coordinates.
    
    Args:
    coordinates (numpy.ndarray): A numpy array of shape (n, 3) where each row is an XYZ coordinate.
    file_path (str): Path where the PLY file will be saved.
    """
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(coordinates)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]

    with open(file_path, 'w') as ply_file:
        # Write the header
        ply_file.write('\n'.join(header) + '\n')
        
        # Write the vertex data
        for _, x, y, z in coordinates:
            ply_file.write(f"{x} {z} {y}\n")

def plot_pc(xyz, hue):
    # Extracting x, y, and z coordinates
    x_coords = xyz[:, 1]
    y_coords = xyz[:, 2]
    z_coords = xyz[:, 3]

    # Creating a new matplotlib figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the 3D scatter plot
    ax.scatter(z_coords, x_coords, y_coords, s=1, c=hue, cmap='viridis', alpha=0.1)

    # Setting labels for axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Showing the plot
    plt.savefig('3d_scatter_plot.png', dpi=600)  # Save as PNG with high resolution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/semantic_kitti/inference_main.gin")              # train_res16unet34c_prob | train_res16unet34c | train_fpt.gin
    args = parser.parse_args()
    
    gin.parse_config_file(args.config)
    
    inference(data_module_name="SemanticKITTIDataModule")
