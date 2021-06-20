import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom

from lib.tokenizer import Tokenizer
tokenizer = Tokenizer()

AUTO = tf.data.experimental.AUTOTUNE

TFRECORDS_PATHS = {
    'tfrecords013': 'gs://kds-58ed18930131dc67bac2c0bd3f88332161cf5ce6936e2a06638eaba7',
    'tfrecords016': 'gs://kds-ade8722870002e6b333930c93d5294fd143c14064293116453ba9832',
    'tfrecords018': 'gs://kds-f090a95044a5e32eda53024e47ef1fe88853b522598a09272e887cc4',
    'tfrecords020': 'gs://kds-13109b3c9ee7b300b9440b1797c8a7abbf8eaca9015034bac331412a',
    'tfrecords031-2': 'gs://kds-629a00274e5eb59b41e21e2d6f6f058f2a02ad06264a5199df8c164c',
    'tfrecords032': 'gs://kds-3c6742d4e2d876cb67576c058fc94ee6d81d558d719060f68a25f252',
    'tfrecords033': 'gs://kds-99c634a8cd4dffcfa3e52d32c5a993d8fdb64ad28bab1eb933f215e8',
}

GCS_PATHS = {
    '300x600_no_pad': TFRECORDS_PATHS['tfrecords013'],
    '300x600_no_pad_test': TFRECORDS_PATHS['tfrecords016'],
    '416x736_no_pad': TFRECORDS_PATHS['tfrecords018'],
    '416x736_no_pad_test': TFRECORDS_PATHS['tfrecords020'],
    '416x736_no_pad_pseudo060': TFRECORDS_PATHS['tfrecords031-2'],
    '416x736_no_pad_noise_denoise': TFRECORDS_PATHS['tfrecords032'],
    '416x736_no_pad_noise_denoise_test': TFRECORDS_PATHS['tfrecords033'],
}

def random_rotate(img, angle, batch_size):
    angle *= np.pi/180
    angle = tf.random.uniform(
        shape=[batch_size], minval=-angle, maxval=angle)
    return tfa.image.rotate(img, angle, fill_value=255)


def salt_and_pepper(img, labels, height, width):
    pepper_amount = tf.random.uniform([], 0, 0.001, tf.float32)
    pepper = tf.random.uniform(
        [height, width], 0, 1, tf.float32) < pepper_amount
    img = tf.where(pepper[:, :, None], tf.zeros_like(img), img)

    salt_amount = tf.random.uniform([], 0, 0.3, tf.float32)
    salt = tf.random.uniform([height, width], 0, 1, tf.float32) < salt_amount
    img = tf.where(salt[:, :, None], tf.ones_like(img)*255, img)
    return img, labels


def _reshape_image(img, height, width, gray_scale):
    if gray_scale:
        img = tf.reshape(img, (height, width))
        img = tf.stack([img, img, img], axis=-1)
    else:
        img = tf.reshape(img, (height, width, 3))
    return img


def read_tfrecord(example, height, width, seq_len, label_dtype, gray_scale=False):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = tf.image.decode_png(example['image'])
    img = _reshape_image(img, height, width, gray_scale)

    label = tf.io.decode_raw(example['label'], label_dtype)
    label = tf.reshape(label, (277,))
    label = label[:seq_len]
    label = (label + 1) % 193
    label = tf.cast(label, tf.int32)
    return img, label


def read_test_tfrecord(example, height, width, gray_scale=False):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = tf.image.decode_png(example['image'])
    img = _reshape_image(img, height, width, gray_scale)
    image_id = example['image_id']
    return img, image_id


def imagenet_normalize(img, labels, dtype_):
    IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = tf.cast(img, dtype_)
    return img, labels


def _get_tfrecords_path(mode, gcs_dir, fold, pseudo_gcs_dir=None):
    lengths = [
        121210, 121210, 121210, 121210, 121210, 121210,
        121209, 121209, 121209, 121209, 121209, 121209,
        121209, 121209, 121209, 121209, 121209, 121209,
        121209, 121209
    ]

    if mode == 'train':
        files = tf.io.gfile.glob(f'{gcs_dir}/*.tfrec')
        files = [f for f in files if f'fold{fold}' not in f]
        length = sum([length for i, length in enumerate(lengths) if i != fold])
    else:
        files = tf.io.gfile.glob(f'{gcs_dir}/*.tfrec')
        files = [f for f in files if f'fold{fold}' in f]
        length = lengths[fold]

    if pseudo_gcs_dir:
        pseudo_files = files = tf.io.gfile.glob(f'{pseudo_gcs_dir}/*.tfrec')
        print('pseudo labels:', pseudo_files)
        files += pseudo_files
        length += 1616018
        random.shuffle(files)

    return files, length


def get_train_dataset(gcs_dir, batch_size, fold, dtype_, height, width, seq_len, label_dtype, gray_scale, rotate_angle, zoom_range, do_salt_and_pepper, pseudo_gcs_dir=None):
    files, length = _get_tfrecords_path('train', gcs_dir, fold, pseudo_gcs_dir)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.prefetch(AUTO)

    def _read_func(example):
        return read_tfrecord(example, height, width, seq_len, label_dtype, gray_scale)
    ds = ds.map(_read_func, num_parallel_calls=AUTO)

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    ds = ds.batch(batch_size, drop_remainder=True)

    if do_salt_and_pepper:
        def _sp_func(image, labels):
            return salt_and_pepper(image, labels, height, width)
        ds = ds.map(_sp_func, num_parallel_calls=AUTO)

    if zoom_range > 0:
        zoom = RandomZoom((-zoom_range, zoom_range))

        def _zoom_func(images, labels):
            return zoom(images), labels
        ds = ds.map(_zoom_func, num_parallel_calls=AUTO)

    if rotate_angle > 0:
        def _rotate_func(images, labels):
            images = random_rotate(images, rotate_angle, batch_size)
            return images, labels
        ds = ds.map(_rotate_func, num_parallel_calls=AUTO)

    ds = ds.map(lambda x, y: imagenet_normalize(
        x, y, dtype_), num_parallel_calls=AUTO)
    ds = ds.with_options(ignore_order)
    ds = ds.shuffle(512, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.prefetch(1)
    return ds, length


def get_val_dataset(gcs_dir, batch_size, fold, dtype_, height, width, seq_len, label_dtype, gray_scale):
    files, length = _get_tfrecords_path('val', gcs_dir, fold)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.prefetch(AUTO)

    def _read_func(example):
        return read_tfrecord(example, height, width, seq_len, label_dtype, gray_scale)
    ds = ds.map(_read_func, num_parallel_calls=AUTO)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(lambda x, y: imagenet_normalize(
        x, y, dtype_), num_parallel_calls=AUTO)
    ds = ds.prefetch(1)
    return ds, length


def get_test_dataset(gcs_dir, batch_size, dtype_, height, width, gray_scale):
    length = 1616107
    files = tf.io.gfile.glob(f'{gcs_dir}/*.tfrec')

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.prefetch(AUTO)
    ds = ds.repeat()

    def _read_func(example):
        return read_test_tfrecord(example, height, width, gray_scale)
    ds = ds.map(_read_func, num_parallel_calls=AUTO)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(lambda x, y: imagenet_normalize(
        x, y, dtype_), num_parallel_calls=AUTO)
    ds = ds.prefetch(1)
    return ds, length

def read_image(file_name, label, height=416, width=736, seq_len=200):
    bits = tf.io.read_file(file_name)
    img = tf.image.decode_png(bits, channels=3)
    img = tf.reshape(img, (height, width, 3))
    label = label[:seq_len]
    label = (label + 1) % 193  # to set pad token as 0
    label = tf.cast(label, tf.int64)
    return img, label

def read_local_image(file_name, label, height, width, seq_len, gray_scale):
    bits = tf.io.read_file(file_name)
    if gray_scale:
        img = tf.image.decode_png(bits, channels=1)
        img = tf.reshape(img, (height, width))
        img = tf.stack([img, img, img], axis=-1)
    else:
        img = tf.image.decode_png(bits, channels=3)
        img = tf.reshape(img, (height, width, 3))
    label = label[:seq_len]
    label = (label + 1) % 193  # to set pad token as 0
    label = tf.cast(label, tf.int32)
    return img, label

def get_local_candidate_dataset(df_path, img_dir, batch_size, height, width, seq_len, gray_scale):
    df = pd.read_csv(df_path)
    df = df.rename(columns={'normed_InChI': 'InChI'})
    df = df[['image_id', 'InChI']]

    file_paths = df['image_id'].map(
        lambda x: f'{img_dir}/{x}.png').tolist()
    labels = df['InChI'].swifter.apply(tokenizer.tokenize).tolist()

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    def _read_func(file_name, label):
        return read_local_image(file_name, label, height, width, seq_len, gray_scale)
    ds = ds.map(_read_func, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(imagenet_normalize, num_parallel_calls=AUTO)
    ds = ds.prefetch(1)
    ds_len = len(df)

    return ds, ds_len