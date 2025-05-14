"""
Adapted from the official implementation of:
"Intriguing Properties of Synthetic Images: From GANs to Diffusion Models"
by GRIP-UNINA. Used for academic, non-commercial research purposes only.
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import os
import argparse
import random

from rescale import rescale_area
from denoiser import get_denoiser


def fft2_area(img, siz):
    img = np.fft.fft2(img, axes=(0, 1), norm='ortho')
    img_energy = np.abs(img) ** 2
    img_energy = rescale_area(rescale_area(img_energy, siz, 0), siz, 1)
    img_energy = np.fft.fftshift(img_energy, axes=(0, 1))
    return img_energy


def imread(filename):
    return np.asarray(Image.open(filename).convert('RGB')) / 256.0


def compute_power_spectrum(files_path, output_dir, output_name):
    print(f"Generating average power spectrum for images in: {files_path}")
    filenames = glob.glob(os.path.join(files_path, "*"))
    random.shuffle(filenames)
    filenames = filenames[:1000]

    denoiser = get_denoiser(1, cuda=torch.cuda.is_available())
    siz = 222
    res_fft2 = [fft2_area(denoiser(imread(f)), siz) for f in tqdm(filenames)]
    res_fft2_mean = np.mean(res_fft2, axis=0)
    energy_norm = np.mean(res_fft2_mean)
    res_fft2_mean /= 4 * energy_norm

    # Save figure
    output_path = os.path.join(output_dir, output_name)
    os.makedirs(output_path, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(np.mean(res_fft2_mean, axis=-1).clip(0, 1),
               clim=[0, 1], extent=[-0.5, 0.5, 0.5, -0.5])
    plt.xticks([])
    plt.yticks([])
    fig.savefig(os.path.join(output_path, 'fft2_gray.png'),
                bbox_inches='tight', pad_inches=0.0)


if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser(description="Compute average power spectrum of synthetic images")
    parser.add_argument("--files_path", type=str, required=True, help="Folder containing synthetic images")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for saving plots")
    parser.add_argument("--out_name", type=str, required=True, help="Name of subfolder for output files")
    args = parser.parse_args()

    compute_power_spectrum(args.files_path, args.out_dir, args.out_name)
