import pandas as pd
from multiprocessing import Pool
import os
import click

from lib.tfrecords_util import create_test_tfrecords


@click.command()
@click.option('--df_path', '-d', type=str)
@click.option('--image_dir', '-i', type=str)
@click.option('--save_dir', '-s', type=str)
def main(df_path, image_dir, save_dir):
    df = pd.read_csv(df_path)
    df['file_path'] = df['image_id'].map(
        lambda x: os.path.join(image_dir, f'{x}.png'))
    create_test_tfrecords(df, save_dir)


if __name__ == '__main__':
    main()
