import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers.experimental.preprocessing import RandomZoom

AUTO = tf.data.experimental.AUTOTUNE

GCS_PATHS = {
    '300x600_no_pad': 'gs://kds-979a6e5e09f2e5daa41b272745a5724d0ba787f09b84bb30f25f81c5',
    '300x600_no_pad_test': 'gs://kds-f2e338c5eaa7c880d3f99c69221f1fe1ca5f39c4fc9d91ac907af560',
    '416x736_no_pad': 'gs://kds-1636aeb30aaca51ddfeec6cfa7a2339bf6f70968eb7a7d1efc8665f2',
    '416x736_no_pad_noise_denoise': 'gs://kds-9a4900604f031e0764c4d8170125b0e8705bb09a846a634da16dd807',
    '416x736_no_pad_test': 'gs://kds-d8a59fcdfcfe49b0c44db9c27d60f8f1f80f8233054c59f1f61e2c08',
    '416x736_no_pad_noise_denoise_test': 'gs://kds-28668c53d609f95ef872b5c343ff79179b559925bd4ebf3e724b60c4',
    '416x736_no_pad_pseudo060': 'gs://kds-be866cd81681c2f7d50550f36e017e72e2179f1f00be94bb2a6d0a41',
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


def read_tfrecord(example, height, width, seq_len, gray_scale=False):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = tf.image.decode_png(example['image'])
    img = _reshape_image(img, height, width, gray_scale)

    label = tf.io.decode_raw(example['label'], tf.int32)
    label = tf.reshape(label, (277,))
    label = label[:seq_len]
    label = (label + 1) % 193
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


def get_train_dataset(gcs_dir, batch_size, fold, dtype_, height, width, seq_len, gray_scale, rotate_angle, zoom_range, do_salt_and_pepper, pseudo_gcs_dir=None):
    files, length = _get_tfrecords_path('train', gcs_dir, fold, pseudo_gcs_dir)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.prefetch(AUTO)

    def _read_func(example):
        return read_tfrecord(example, height, width, seq_len, gray_scale)
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


def get_val_dataset(gcs_dir, batch_size, fold, dtype_, height, width, seq_len, gray_scale):
    files, length = _get_tfrecords_path('val', gcs_dir, fold)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.prefetch(AUTO)

    def _read_func(example):
        return read_tfrecord(example, height, width, seq_len, gray_scale)
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
