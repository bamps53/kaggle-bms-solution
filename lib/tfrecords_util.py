import os

import tensorflow as tf
from tqdm.auto import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_train_example(image, label, image_id):
    feature = {
        'image': _bytes_feature(image),
        'image_id': _bytes_feature(image_id),
        'label': _bytes_feature(label),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_test_example(image, image_id):
    feature = {
        'image': _bytes_feature(image),
        'image_id': _bytes_feature(image_id),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def read_image(fp):
    with tf.io.gfile.GFile(fp, "rb") as fid:
        encoded_img = fid.read()
    return encoded_img


def get_image_id(fp):
    return os.path.basename(fp).strip('.png')


def get_train_chunk_data(file_paths, labels, num_chunk, chunk_size):
    start = num_chunk * chunk_size
    end = start + chunk_size
    chunk_file_paths = file_paths[start:end]
    chunk_labels = labels[start:end]
    return chunk_file_paths, chunk_labels


def get_test_chunk_data(file_paths, num_chunk, chunk_size):
    start = num_chunk * chunk_size
    end = start + chunk_size
    chunk_file_paths = file_paths[start:end]
    return chunk_file_paths


def create_train_tfrecords(df, fold, save_dir, tokenizer, chunk_size=100000):
    os.makedirs(save_dir, exist_ok=True)
    file_paths = df['file_path'].values
    labels = df['InChI_text'].map(tokenizer.tokenize).values

    num_chunks = len(file_paths) // chunk_size + 1
    print(f'num_chunks:', num_chunks)

    for num_chunk in range(num_chunks):
        chunk_file_paths, chunk_labels = get_train_chunk_data(
            file_paths, labels, num_chunk, chunk_size)
        with tf.io.TFRecordWriter(f'{save_dir}/tfrecords_fold{fold}_{num_chunk:04}.tfrec') as writer:
            for file_path, label in tqdm(zip(chunk_file_paths, chunk_labels), total=len(chunk_file_paths)):
                img = read_image(file_path)
                image_id = str.encode(get_image_id(file_path))
                label = label.tostring()
                example = serialize_train_example(img, label, image_id)
                writer.write(example)


def create_test_tfrecords(df, save_dir, chunk_size=100000):
    os.makedirs(save_dir, exist_ok=True)
    file_paths = df['file_path'].values

    chunk_size = 100000
    num_chunks = len(file_paths) // chunk_size + 1
    print(f'num_chunks:', num_chunks)

    for num_chunk in range(num_chunks):
        chunk_file_paths = get_test_chunk_data(
            file_paths, num_chunk, chunk_size)
        with tf.io.TFRecordWriter(f'{save_dir}/tfrecords_{num_chunk:04}.tfrec') as writer:
            for file_path in tqdm(chunk_file_paths):
                img = read_image(file_path)
                image_id = str.encode(get_image_id(file_path))
                example = serialize_test_example(img, image_id)
                writer.write(example)
