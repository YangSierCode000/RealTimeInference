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

def print_results(classnames, confusion_matrix):    # (21, 21)
    # results
    ious = per_class_iou(confusion_matrix) * 100    # (21,)
    accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100         # (21,)
    miou = np.nanmean(ious)
    macc = np.nanmean(accs)
    print(f'miou:{miou:.4f}')
    print(f'macc:{macc:.4f}')
    print(f'ious:{ious}')
    df = pd.DataFrame(data=ious)
    df.to_csv('baseline.csv', header=0, float_format='%.4f')

    # print results
    console = Console()
    table = Table(show_header=True, header_style="bold")

    columns = ["mAcc", "mIoU"]
    num_classes = len(classnames)
    for i in range(num_classes):
        columns.append(classnames[i])
    for col in columns:
        table.add_column(col)
    ious = ious.tolist()
    row = [macc, miou, *ious]
    table.add_row(*[f"{x:.2f}" for x in row])
    # console.print(table)

 
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

def plot_pc(xyz, hue, file_name='3d_scatter_plot.png'):
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
    plt.savefig(file_name, dpi=600)  # Save as PNG with high resolution


@torch.no_grad()
def infer(model, batch, device):
    in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
    pred = model(in_data).cpu()
    return pred
    # if type(pred) is tuple:
    #     pred = pred[0].squeeze(0).cpu()
    # else:
    #     pred = pred.cpu()
    # return pred



@gin.configurable
def inference(
    checkpoint_path,
    model_name,
    data_module_name,
    save_report=False
):

    inference_dir = join(dirname(checkpoint_path), f"inference_{datetime.now().strftime('%m%d_%H%M%S')}")
    meta_dir = join(inference_dir, 'meta')
    if not exists(meta_dir):
        os.makedirs(meta_dir)

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    ckpt = torch.load(checkpoint_path, map_location=device)

    def remove_prefix(k, prefix):
        return k[len(prefix):] if k.startswith(prefix) else k

    state_dict = {remove_prefix(k, "model."): v for k, v in ckpt["state_dict"].items()}
    model = get_model(model_name)()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    data_module = get_data_module(data_module_name)()
    data_module.setup("test")
    val_loader = data_module.val_dataloader()

    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=data_module.dset_val.NUM_CLASSES) # FIXME , compute_on_step=False

    outputs = []
    ignore_portion = []
    if 'Prob' in model_name:
        print('CUE inference route')
        with torch.inference_mode(mode=True):
            for index, batch in enumerate(track(val_loader)):
                if index == 1:
                    break
                
                print("Log: number of points:", batch["coordinates"].shape)
                
                plot_pc(batch["coordinates"][batch["labels"] != 255], batch["labels"][batch["labels"] != 255], join(meta_dir, '3d_scatter_plot.png'))
                create_ply_file(np.array(batch["coordinates"]), join(meta_dir, "point_cloud.ply"))

                in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
                # print("[Model] ", model) # TODO if you want to see the model structure
                # print("[data] ", in_data.shape) # TODO if you want to see the input shape
                logits, emb_mu, emb_sigma = model(in_data)                     # ([Nr, 13])
                logits = logits.mean(dim=0)
                pred_dense = logits.argmax(dim=1, keepdim=False)
                emb_mu_dense = emb_mu.slice(in_data)           # TensorField
                emb_sigma_dense = emb_sigma.slice(in_data)     # TensorField
                xyz_dense = batch["coordinates"]
                label_dense = batch["labels"]

                xyz_sparse, unique_map = ME.utils.sparse_quantize(xyz_dense, return_index=True)
                labels_sparse = label_dense[unique_map]
                emb_mu_sparse = emb_mu_dense.F[unique_map]
                emb_sigma_sparse = emb_sigma_dense.F[unique_map]
                logits_sparse = logits[unique_map]   # logits[:,unique_map,:]
                rgb_sparse = batch["features"][unique_map]
                pred_sparse = pred_dense[unique_map]
                bin_precisions, bin_counts, precisions = com.get_bins_precision(emb_sigma_sparse, pred_sparse, labels_sparse)
                outputs.append([bin_precisions, bin_counts, precisions])
                
                if save_report:
                    np.save(join(meta_dir, f'{index}_seg_logit.npy'), logits_sparse.cpu().numpy())     # ([m, N, 13])
                    np.save(join(meta_dir, f'{index}_pred.npy'), pred_sparse.cpu().numpy())
                    np.save(join(meta_dir, f'{index}_sigma.npy'), emb_sigma_sparse.cpu().numpy())
                    np.save(join(meta_dir, f'{index}_xyz.npy'), xyz_sparse[:,1:].cpu().numpy())
                    np.save(join(meta_dir, f'{index}_rgb.npy'), rgb_sparse.cpu().numpy())
                    np.save(join(meta_dir, f'{index}_label.npy'), labels_sparse.cpu().numpy())

                    np.save(join(meta_dir, f'{index}_seg_logit_dense.npy'), logits.cpu().numpy())     # ([m, Nr, 13])
                    np.save(join(meta_dir, f'{index}_pred_dense.npy'), pred_dense.cpu().numpy())
                    np.save(join(meta_dir, f'{index}_sigma_dense.npy'), emb_sigma_dense.F.cpu().numpy())
                    np.save(join(meta_dir, f'{index}_xyz_dense.npy'), xyz_dense[:,1:].cpu().numpy())
                    np.save(join(meta_dir, f'{index}_rgb_dense.npy'), batch["features"].cpu().numpy())
                    np.save(join(meta_dir, f'{index}_label_dense.npy'), label_dense.cpu().numpy())

                mask = batch["labels"] != data_module.dset_val.ignore_label
                ignore_portion.append(mask.float().mean())
                pred = logits.argmax(dim=1).cpu()          # mean(dim=0).
                
                plot_pc(batch["coordinates"], pred, file_name=join(meta_dir, '3d_pred_scatter_plot.png'))
                
                create_ply_file(np.array(batch["coordinates"][batch["labels"] == 0]), join(meta_dir, "point_cloud_car.ply"))
                create_ply_file(np.array(batch["coordinates"][pred == 0]), join(meta_dir, "point_cloud_pred_car.ply"))

                
                confmat(pred[mask], batch["labels"][mask])
                torch.cuda.empty_cache()

    confmat = confmat.compute().numpy() # (21, 21)
    cnames = data_module.dset_val.get_classnames()
    print_results(cnames, confmat)

    if 'Prob' in model_name:
        bin_precisions = np.array([x[0] for x in outputs]).mean(axis=0)
        bin_counts = np.array([x[1] for x in outputs]).mean(axis=0)
        precision = np.array([x[2] for x in outputs]).mean()
        ece_s = com.cal_ece(bin_precisions, bin_counts)
        print(f'ece_s:{ece_s:.3f}')

    ignore_portion = np.array(ignore_portion)
    print(f'labeled points/all points:{ignore_portion.mean():.3f}')
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/semantic_kitti/inference_main.gin")              # train_res16unet34c_prob | train_res16unet34c | train_fpt.gin
    parser.add_argument("--ckpt_path", type=str, default=None)              # train_res16unet34c_prob | train_res16unet34c | train_fpt.gin
    args = parser.parse_args()
    
    gin.parse_config_file(args.config)
    
    inference(checkpoint_path=args.ckpt_path, data_module_name="SemanticKITTIDataModule")
