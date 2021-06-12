import os
from multiprocessing import Pool, cpu_count
import functools

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
    h, w, _ = img.shape
    if h > w:
        transpose_flip = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        img = transpose_flip(image=img)['image']
    return img


def read_and_save(fp, transforms, save_dir):
    img = cv2.imread(fp)
    assert img is not None
    img = fix_rotate_image(img)
    img = transforms(image=img)['image']
    file_name = os.path.basename(fp)
    cv2.imwrite(os.path.join(save_dir, file_name), img)


@click.command()
@click.option('--height', '-h', type=int)
@click.option('--width', '-w', type=int)
def main(height, width):
    save_dir = f'data/resized{height}x{width}_test/'
    os.makedirs(save_dir, exist_ok=True)

    transforms = A.Resize(height, width,)

    df = pd.read_csv(
        '../input/bms-molecular-translation/sample_submission.csv')
    df['file_path'] = df['image_id'].apply(get_test_file_path)

    map_func = functools.partial(
        read_and_save, transforms=transforms, save_dir=save_dir)
    with tqdm(total=len(df)) as t:
        with Pool(cpu_count()) as p:
            for _ in p.imap_unordered(map_func, df.file_path):
                t.update(1)


if __name__ == '__main__':
    main()
