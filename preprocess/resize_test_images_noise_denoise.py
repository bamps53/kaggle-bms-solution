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


def get_test_file_path(image_id):
    return "../input/bms-molecular-translation/test/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id
    )


def fix_rotate_image(img):
    h, w = img.shape
    if h > w:
        transpose_flip = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        img = transpose_flip(image=img)['image']
    return img


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


def read_and_save(fp, save_dir, height, width):
    resize = A.Resize(height, width,)
    denoise = Denoise()

    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    img = fix_rotate_image(img)

    img = img[None, :, :]
    img = torch.Tensor(img / 255.0)
    img = denoise(img)
    img = (img.numpy()[0, :, :] * 255).astype('uint8')

    img = resize(image=img)['image']
    file_name = os.path.basename(fp)
    cv2.imwrite(os.path.join(save_dir, file_name), img)


@click.command()
@click.option('--height', '-h', type=int)
@click.option('--width', '-w', type=int)
def main(height, width):
    save_dir = f'data/resized{height}x{width}_noise_denoise_test/'
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(
        '../input/bms-molecular-translation/sample_submission.csv')
    df['file_path'] = df['image_id'].apply(get_test_file_path)

    map_func = functools.partial(
        read_and_save, save_dir=save_dir, height=height, width=width)
    with tqdm(total=len(df)) as t:
        with Pool(cpu_count()) as p:
            for _ in p.imap_unordered(map_func, df.file_path):
                t.update(1)


if __name__ == '__main__':
    main()
