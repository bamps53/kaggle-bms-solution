import os
from multiprocessing import Pool, cpu_count
import functools

import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import click
import cv2
import pandas as pd
from tqdm.auto import tqdm


def get_train_file_path(image_id):
    return "../input/bms-molecular-translation/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id
    )


class Denoise(nn.Module):
    def __init__(
        self,
        kernel_size=5,
        thresh=1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(
            1, 1,
            (kernel_size, kernel_size),
            padding=padding,
            bias=False,
        )
        self.conv.weight = torch.nn.Parameter(
            torch.ones((1, 1, kernel_size, kernel_size))
        )
        self.conv.require_grad = False
        self.thresh = thresh

    @torch.no_grad()
    def forward(self, image):
        inv_image = (~image.bool()).float()
        mask = self.conv(inv_image[None]).squeeze(0) > self.thresh
        denoised = inv_image * mask
        denoised = (~denoised.bool()).float()
        return denoised


class SaltAndPepperNoise(nn.Module):
    def __init__(
        self,
        salt_amount=0.3,
        pepper_amount=0.001,
    ):
        super().__init__()
        self.salt_amount = salt_amount
        self.pepper_amount = pepper_amount

    @torch.no_grad()
    def forward(self, image):
        salt_amount = torch.rand(1) * self.salt_amount
        salt = torch.rand(image.shape) < salt_amount
        image = torch.where(salt, torch.full_like(image, 1.0), image)
        pepper_amount = torch.rand(1) * self.pepper_amount
        pepper = torch.rand(image.shape) < pepper_amount
        image = torch.where(pepper, torch.zeros_like(image), image)
        return image


def preprocess(img, height, width):
    salt_and_pepper = SaltAndPepperNoise()
    denoise = Denoise()
    resize = A.Resize(height, width)

    assert (np.unique(img) == np.array([0, 255], dtype=np.uint8)).all()
    img = img[None, :, :]
    img = torch.Tensor(img / 255.0)
    img = salt_and_pepper(img)
    img = denoise(img)
    img = (img.numpy()[0, :, :] * 255).astype('uint8')
    img = resize(image=img)['image']
    return img


def read_and_save(fp, save_dir, height, width):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    img = preprocess(img, height, width)
    file_name = os.path.basename(fp)
    cv2.imwrite(os.path.join(save_dir, file_name), img)


@click.command()
@click.option('--height', '-h', type=int)
@click.option('--width', '-w', type=int)
def main(height, width):
    save_dir = f'data/resized{height}x{width}_noise_denoise'
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv('data/folds.csv')
    df['file_path'] = df['image_id'].apply(get_train_file_path)

    map_func = functools.partial(
        read_and_save, save_dir=save_dir, height=height, width=width)
    with tqdm(total=len(df)) as t:
        with Pool(cpu_count()) as p:
            for _ in p.imap_unordered(map_func, df.file_path):
                t.update(1)


if __name__ == '__main__':
    main()
