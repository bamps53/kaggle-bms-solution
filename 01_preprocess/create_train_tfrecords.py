import pandas as pd
from multiprocessing import Pool
import os
import click
import functools

from lib.tokenizer import Tokenizer
from lib.tfrecords_util import create_train_tfrecords

NUM_FOLDS = 20


@click.command()
@click.option('--df_path', '-d', type=str)
@click.option('--image_dir', '-i', type=str)
@click.option('--save_dir', '-s', type=str)
def main(df_path, image_dir, save_dir):

    df = pd.read_csv(df_path)
    df['file_path'] = df['image_id'].map(
        lambda x: os.path.join(image_dir, f'{x}.png'))
    dfs = [df.query('fold==@fold').reset_index(drop=True)
           for fold in range(NUM_FOLDS)]

    tokenizer = Tokenizer()

    map_func = functools.partial(
        create_train_tfrecords, save_dir=save_dir, tokenizer=tokenizer)
    with Pool(20) as p:
        p.starmap(map_func, zip(dfs, range(NUM_FOLDS)))


if __name__ == '__main__':
    main()
