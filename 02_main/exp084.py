import io
import json
import math
import os
import pickle
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Union

import efficientnet.tfkeras as efn
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import torch
import wandb
from tensorflow.keras.mixed_precision import experimental as mixed_precision
# from kaggle_datasets import KaggleDatasets
from tqdm.auto import tqdm

# seed everything
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

GCS_PATHS = {
    '300x600_no_pad': 'gs://kds-979a6e5e09f2e5daa41b272745a5724d0ba787f09b84bb30f25f81c5',
    '300x600_no_pad_test': 'gs://kds-f2e338c5eaa7c880d3f99c69221f1fe1ca5f39c4fc9d91ac907af560',
    '416x736_no_pad': 'gs://kds-1636aeb30aaca51ddfeec6cfa7a2339bf6f70968eb7a7d1efc8665f2',
    '416x736_no_pad_noise_denoise': 'gs://kds-9a4900604f031e0764c4d8170125b0e8705bb09a846a634da16dd807',
    '416x736_no_pad_test': 'gs://kds-d8a59fcdfcfe49b0c44db9c27d60f8f1f80f8233054c59f1f61e2c08',
    '416x736_no_pad_noise_denoise_test': 'gs://kds-28668c53d609f95ef872b5c343ff79179b559925bd4ebf3e724b60c4',
    '416x736_no_pad_pseudo060': 'gs://kds-be866cd81681c2f7d50550f36e017e72e2179f1f00be94bb2a6d0a41',
}


# Detect hardware, set appropriate distribution strategy (GPU/TPU)
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', TPU.master())
except ValueError:
    print('Running on GPU')
    TPU = None

if TPU:
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    # default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

# # set half precision policy
mixed_precision.set_policy('mixed_bfloat16' if TPU else 'float32')

# enable XLA optmizations
tf.config.optimizer.set_jit(True)

print(f'Compute dtype: {mixed_precision.global_policy().compute_dtype}')
print(f'Variable dtype: {mixed_precision.global_policy().variable_dtype}')

#########################################################################################################
# CONFIG
#########################################################################################################


@dataclass
class CFG:
    exp_id = 'exp084_416_736_transformer_b4_2layers_seqlen200_fix_pos_random_crop_focal'
    debug: bool = False
    inference: bool = False

    steps_per_epoch: int = 20000
    batch_size_base: int = 16
    batch_size: int = batch_size_base * REPLICAS
    test_batch_size_base: int = 16
    test_batch_size: int = batch_size_base * REPLICAS
    num_epochs: int = 50
    steps_per_execution: int = 1

    warmup_steps: int = 500
    verbose_freq: int = 100
    save_freq: int = 5000
    total_steps: int = num_epochs * steps_per_epoch

    save_dir: str = f'{YOUR_GCS_DIR}/{exp_id}'

    # target data type, bfloat16 when using TPU to improve throughput
    image_height: int = 416
    image_width: int = 736
    row_size: int = image_height // 32
    col_size: int = image_width // 32

    dtype: str = tf.bfloat16 if TPU else tf.float32
    # dtype: str = tf.float32
    train_gcs_dir: str = GCS_PATHS['416x736_no_pad']
    val_gcs_dir: str = GCS_PATHS['416x736_no_pad']
    test_gcs_dir: str = GCS_PATHS['416x736_no_pad_test']
    val_size: int = 121210
    val_steps: int = val_size // batch_size
    rotate_angle: int = 5

    # configure model
    vocab_size: int = 193
    seq_len: int = 200
    attention_units: int = 256
    encoder_dim: int = 1792
    start_token: int = 191
    end_token: int = 192
    pad_token: int = 0
    drop_rate: float = 0.3
    num_layers: int = 2
    d_model: int = 512
    num_heads: int = 8
    dff: int = 2048
    encoder_drop_rate: float = 0.1
    decoder_drop_rate: float = 0.1

    resume: bool = True
    resume_from_last: bool = True
    resume_epoch: int = None
    resume_steps: int = None


opt = CFG()
os.makedirs(opt.save_dir, exist_ok=True)

PAD_TOKEN = tf.constant(opt.pad_token, dtype=tf.int64)
START_TOKEN = tf.constant(opt.start_token, dtype=tf.int64)
END_TOKEN = tf.constant(opt.end_token, dtype=tf.int64)

if opt.debug:
    opt.steps_per_epoch = 10
    opt.val_steps = 10
    opt.num_epochs = 5
    opt.steps_per_execution = 1
#########################################################################################################
# DATASET
#########################################################################################################
 # Tensorflow AUTO flag, used in datasets
AUTO = tf.data.experimental.AUTOTUNE


def random_rotate(img, angle=opt.rotate_angle):
    angle *= np.pi/180
    angle = tf.random.uniform(
        shape=[opt.batch_size], minval=-angle, maxval=angle)
    return tfa.image.rotate(img, angle, fill_value=255)


def read_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = tf.image.decode_png(example['image'])
    img = tf.reshape(img, (opt.image_height, opt.image_width, 3))
    label = tf.io.decode_raw(example['label'], tf.int64)
    label = tf.reshape(label, (277,))
    label = label[:opt.seq_len]
    label = (label + 1) % 193
    return img, label


def read_test_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = tf.image.decode_png(example['image'])
    img = tf.reshape(img, (opt.image_height, opt.image_width, 3))
    image_id = example['image_id']
    # image_id = tf.io.decode_raw(example['image_id'], tf.int64)
    return img, image_id


def imagenet_normalize(img, labels):
    IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = tf.cast(img, opt.dtype)
    return img, labels


def get_dataset(mode, batch_size=64, data_root='./', fold=0):
    lengths = [
        121210, 121210, 121210, 121210, 121210, 121210,
        121209, 121209, 121209, 121209, 121209, 121209,
        121209, 121209, 121209, 121209, 121209, 121209,
        121209, 121209
    ]
    if mode == 'train':
        files = tf.io.gfile.glob(f'{opt.train_gcs_dir}/*.tfrec')
        files = [f for f in files if f'fold{fold}' not in f]
        length = sum([length for i, length in enumerate(lengths) if i != fold])
    else:
        files = tf.io.gfile.glob(f'{opt.val_gcs_dir}/*.tfrec')
        files = [f for f in files if f'fold{fold}' in f]
        length = lengths[fold]

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.prefetch(AUTO)
    ds = ds.map(read_tfrecord, num_parallel_calls=AUTO)

    if mode == 'train':
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False

        ds = ds.batch(batch_size, drop_remainder=True)
        zoom = tf.keras.layers.experimental.preprocessing.RandomZoom(
            (-0.1, 0.1))
        ds = ds.map(lambda x, y: (zoom(x), y),
                    num_parallel_calls=AUTO)
        ds = ds.map(lambda x, y: (random_rotate(x), y),
                    num_parallel_calls=AUTO)

        ds = ds.map(imagenet_normalize, num_parallel_calls=AUTO)
        ds = ds.with_options(ignore_order)
        ds = ds.shuffle(512, reshuffle_each_iteration=True)
        ds = ds.repeat()
    else:
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(imagenet_normalize, num_parallel_calls=AUTO)
    ds = ds.prefetch(1)
    return ds, length


def get_test_dataset(batch_size=64):
    length = 1616107
    files = tf.io.gfile.glob(f'{opt.test_gcs_dir}/*.tfrec')

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.prefetch(AUTO)
    ds = ds.repeat()
    ds = ds.map(read_test_tfrecord, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(imagenet_normalize, num_parallel_calls=AUTO)
    ds = ds.prefetch(1)
    return ds, length


#########################################################################################################
# MODEL
#########################################################################################################
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding_1d(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=opt.dtype)


def positional_encoding_2d(row, col, d_model):
    assert d_model % 2 == 0
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0),
                        row, axis=0).reshape(-1, 1)

    angle_rads_row = get_angles(row_pos, np.arange(
        d_model//2)[np.newaxis, :], d_model//2)
    angle_rads_col = get_angles(col_pos, np.arange(
        d_model//2)[np.newaxis, :], d_model//2)

    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])
    pos_encoding = np.concatenate([angle_rads_row, angle_rads_col], axis=1)[
        np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=opt.dtype)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), opt.dtype)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.cast(mask, opt.dtype)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], opt.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, q_pos=None, k_pos=None, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        if q_pos is not None:
            q = q + q_pos
        if k_pos is not None:
            k = k + k_pos

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # (batch_size, seq_len_q,      num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        return output, attention_weights


class Encoder(tf.keras.Model):
    def __init__(self, d_model, drop_rate):
        super(Encoder, self).__init__()
        self.d_model = d_model

        self.backbone = efn.EfficientNetB4(
            include_top=False, weights='noisy-student')
        self.reshape = tf.keras.layers.Reshape(
            [-1, self.d_model], name='reshape_featuere_maps')

        self.embedding = tf.keras.layers.Dense(self.d_model, activation='relu')
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training):
        x = self.backbone(x, training=training)  # (B, H, W, 1792)
        x = self.embedding(x, training=training)  # (B, H, W, 512)
        x = self.reshape(x, training=training)  # (B, H*W, 512)
        x = self.dropout(x, training=training)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, max_len, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, enc_pos, dec_pos, training, look_ahead_mask=None, padding_mask=None):
        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(
            x, x, x, q_pos=dec_pos, k_pos=dec_pos, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            out1, enc_output, enc_output, q_pos=dec_pos, k_pos=enc_pos)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_len, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding_1d = positional_encoding_1d(max_len, d_model)
        self.pos_encoding_2d = positional_encoding_2d(
            opt.row_size, opt.col_size, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, max_len, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        dec_pos = self.pos_encoding_1d[:, :seq_len, :]
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, opt.dtype))

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, self.pos_encoding_2d, dec_pos, training, look_ahead_mask, padding_mask)

        predictions = self.final_layer(x)
        return predictions


#########################################################################################################
# LOSS
#########################################################################################################

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        """Initializes `FocalLoss`.
        Args:
            alpha: The `alpha` weight factor for binary class imbalance.
            gamma: The `gamma` focusing parameter to re-weight loss.
            reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
                loss. Default value is `AUTO`. `AUTO` indicates that the reduction
                option will be determined by the usage context. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`. When used with
                `tf.distribute.Strategy`, outside of built-in training loops such as
                `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
                will raise an error. Please see this custom training [tutorial](
                https://www.tensorflow.org/tutorials/distribute/custom_training) for
                more details.
          name: Optional name for the op. Defaults to 'retinanet_class_loss'.
        """
        self._alpha = alpha
        self._gamma = gamma
        super(FocalLoss, self).__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        """Invokes the `FocalLoss`.
        Args:
            y_true: A tensor of size [batch, num_anchors, num_classes]
            y_pred: A tensor of size [batch, num_anchors, num_classes]
        Returns:
            Summed loss float `Tensor`.
        """
        with tf.name_scope('focal_loss'):
            y_true = tf.one_hot(y_true, opt.vocab_size)
            y_true = tf.cast(y_true, dtype=tf.float32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)
            positive_label_mask = tf.equal(y_true, 1.0)
            cross_entropy = (
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
            probs = tf.sigmoid(y_pred)
            probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
            # With small gamma, the implementation could produce NaN during back prop.
            modulator = tf.pow(1.0 - probs_gt, self._gamma)
            loss = modulator * cross_entropy
            weighted_loss = tf.where(positive_label_mask, self._alpha * loss,
                                     (1.0 - self._alpha) * loss)

        return weighted_loss

    def get_config(self):
        config = {
            'alpha': self._alpha,
            'gamma': self._gamma,
        }
        base_config = super(FocalLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#########################################################################################################
# OPTMIZER
#########################################################################################################


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """
    Adam enables L2 weight decay and clip_by_global_norm on gradients. Just adding the square of the weights to the
    loss function is *not* the correct way of using L2 regularization/weight decay with Adam, since that will interact
    with the m and v parameters in strange ways as shown in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.
    Instead we want ot decay the weights in a manner that doesn't interact with the m/v parameters. This is equivalent
    to adding the square of the weights to the loss with plain (non-momentum) SGD.
    Args:
        learning_rate (:obj:`Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]`, `optional`, defaults to 1e-3):
            The learning rate to use or a schedule.
        beta_1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 parameter in Adam, which is the exponential decay rate for the 1st momentum estimates.
        beta_2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 parameter in Adam, which is the exponential decay rate for the 2nd momentum estimates.
        epsilon (:obj:`float`, `optional`, defaults to 1e-7):
            The epsilon parameter in Adam, which is a small constant for numerical stability.
        amsgrad (:obj:`bool`, `optional`, default to `False`):
            Whether to apply AMSGrad variant of this algorithm or not, see `On the Convergence of Adam and Beyond
            <https://arxiv.org/abs/1904.09237>`__.
        weight_decay_rate (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply.
        include_in_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters by default (unless they are in :obj:`exclude_from_weight_decay`).
        exclude_from_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to exclude from applying weight decay to. If a
            :obj:`include_in_weight_decay` is passed, the names in it will supersede this list.
        name (:obj:`str`, `optional`, defaults to 'AdamWeightDecay'):
            Optional name for the operations created when applying gradients.
        kwargs:
            Keyward arguments. Allowed to be {``clipnorm``, ``clipvalue``, ``lr``, ``decay``}. ``clipnorm`` is clip
            gradients by norm; ``clipvalue`` is clip gradients by value, ``decay`` is included for backward
            compatibility to allow time inverse decay of learning rate. ``lr`` is included for backward compatibility,
            recommended to use ``learning_rate`` instead.
    """

    def __init__(
        self,
        learning_rate: Union[float,
                             tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        weight_decay_rate: float = 1.0e-4,
        include_in_weight_decay: Optional[List[str]] = None,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name: str = "AdamWeightDecay",
        **kwargs
    ):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {"WarmUp": WarmUp}
        return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(
            var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.constant(
            self.weight_decay_rate, name="adam_weight_decay_rate"
        )

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var *
                apply_state[(var.device, var.dtype.base_dtype)
                            ]["weight_decay_rate"],
                use_locking=self._use_locking,
            )
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        grads, tvars = list(zip(*grads_and_vars))
        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name, **kwargs)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients["lr_t"], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(
            var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(
            var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"weight_decay_rate": self.weight_decay_rate})
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


def lrfn(step, WARMUP_LR_START, LR_START, LR_FINAL, DECAYS):
    # exponential warmup
    if step < opt.warmup_steps:
        warmup_factor = (step / opt.warmup_steps) ** 2
        lr = WARMUP_LR_START + (LR_START - WARMUP_LR_START) * warmup_factor
    # staircase decay
    else:
        power = (step - opt.warmup_steps) // ((opt.total_steps -
                                               opt.warmup_steps) / (DECAYS + 1))
        decay_factor = ((LR_START / LR_FINAL) ** (1 / DECAYS)) ** power
        lr = LR_START / decay_factor

    return round(lr, 8)


class LRReduce():
    def __init__(self, optimizer, lr_schedule):
        self.opt = optimizer
        self.lr_schedule = lr_schedule
        # assign initial learning rate
        self.lr = lr_schedule[0]
        self.opt.learning_rate.assign(self.lr)

    def step(self, step, loss=None):
        self.lr = self.lr_schedule[step]
        # assign learning rate to optimizer
        self.opt.learning_rate.assign(self.lr)

    def get_counter(self):
        return self.c

    def get_lr(self):
        return self.lr

#########################################################################################################
# UTIL
#########################################################################################################


def log(batch, loss, t_start_batch, val_loss, val_ls_distance, val_acc, lr):
    # training metrics
    print(
        f'Step %s|' % f'{batch}/{opt.steps_per_epoch}'.ljust(9, ' '),
        f'loss: %.3f,' % loss,
        f'acc: %.3f, ' % train_accuracy.result(),
        end='')

    # plot validation metrics if given
    if val_loss is not None and val_ls_distance is not None and val_acc is not None:
        print(
            f'val_loss: %.3f, ' % val_loss,
            f'val lsd: %s,' % ('%.3f' % val_ls_distance).ljust(5, ' '),
            f'val_acc: %.3f, ' % val_acc,
            end='')
    # always end with learning rate, batch duration and line break
    print(
        f'lr: %s,' % ('%.3E' % lr).ljust(7),
        f't: %s sec' % int(time.time() - t_start_batch),
    )


#########################################################################################################
# METRIC
#########################################################################################################


class Tokenizer(object):

    def __init__(self):
        self.stoi = {
            '(': 0, ')': 1, '+': 2, ',': 3, '-': 4, '/b': 5, '/c': 6, '/h': 7, '/i': 8, '/m': 9, '/s': 10,
            '/t': 11, '0': 12, '1': 13, '10': 14, '100': 15, '101': 16, '102': 17, '103': 18, '104': 19, '105': 20,
            '106': 21, '107': 22, '108': 23, '109': 24, '11': 25, '110': 26, '111': 27, '112': 28, '113': 29, '114': 30,
            '115': 31, '116': 32, '117': 33, '118': 34, '119': 35, '12': 36, '120': 37, '121': 38, '122': 39, '123': 40,
            '124': 41, '125': 42, '126': 43, '127': 44, '128': 45, '129': 46, '13': 47, '130': 48, '131': 49, '132': 50,
            '133': 51, '134': 52, '135': 53, '136': 54, '137': 55, '138': 56, '139': 57, '14': 58, '140': 59, '141': 60,
            '142': 61, '143': 62, '144': 63, '145': 64, '146': 65, '147': 66, '148': 67, '149': 68, '15': 69, '150': 70,
            '151': 71, '152': 72, '153': 73, '154': 74, '155': 75, '156': 76, '157': 77, '158': 78, '159': 79, '16': 80,
            '161': 81, '163': 82, '165': 83, '167': 84, '17': 85, '18': 86, '19': 87, '2': 88, '20': 89, '21': 90,
            '22': 91, '23': 92, '24': 93, '25': 94, '26': 95, '27': 96, '28': 97, '29': 98, '3': 99, '30': 100,
            '31': 101, '32': 102, '33': 103, '34': 104, '35': 105, '36': 106, '37': 107, '38': 108, '39': 109, '4': 110,
            '40': 111, '41': 112, '42': 113, '43': 114, '44': 115, '45': 116, '46': 117, '47': 118, '48': 119, '49': 120,
            '5': 121, '50': 122, '51': 123, '52': 124, '53': 125, '54': 126, '55': 127, '56': 128, '57': 129, '58': 130,
            '59': 131, '6': 132, '60': 133, '61': 134, '62': 135, '63': 136, '64': 137, '65': 138, '66': 139, '67': 140,
            '68': 141, '69': 142, '7': 143, '70': 144, '71': 145, '72': 146, '73': 147, '74': 148, '75': 149, '76': 150,
            '77': 151, '78': 152, '79': 153, '8': 154, '80': 155, '81': 156, '82': 157, '83': 158, '84': 159, '85': 160,
            '86': 161, '87': 162, '88': 163, '89': 164, '9': 165, '90': 166, '91': 167, '92': 168, '93': 169, '94': 170,
            '95': 171, '96': 172, '97': 173, '98': 174, '99': 175, 'B': 176, 'Br': 177, 'C': 178, 'Cl': 179, 'D': 180,
            'F': 181, 'H': 182, 'I': 183, 'N': 184, 'O': 185, 'P': 186, 'S': 187, 'Si': 188, 'T': 189, '<sos>': 190,
            '<eos>': 191, '<pad>': 192}
        self.itos = {v: k for k, v in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            i = (i + 192) % 193
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            elif i == self.stoi['<sos>']:
                continue
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions


tokenizer = Tokenizer()


def dense_to_sparse(dense):
    ones = tf.ones(dense.shape)
    indices = tf.where(ones)
    values = tf.gather_nd(dense, indices)
    sparse = tf.SparseTensor(indices, values, dense.shape)
    return sparse

# computes the levenshtein distance between the predictions and labels


def get_levenshtein_distance(preds, lbls, return_preds=True):
    preds = tf.cast(preds, tf.int64)
    # preds = tf.where(tf.not_equal(preds, START_TOKEN) & tf.not_equal(
    # preds, END_TOKEN) & tf.not_equal(preds, PAD_TOKEN), preds, y=0)

    lbls = strategy.gather(lbls, axis=0)
    lbls = tf.cast(lbls, tf.int64)
    # lbls = tf.where(tf.not_equal(lbls, START_TOKEN) & tf.not_equal(
    #     lbls, END_TOKEN) & tf.not_equal(lbls, PAD_TOKEN), lbls, y=0)

    y_trues = tokenizer.predict_captions(lbls.numpy())
    y_preds = tokenizer.predict_captions(preds.numpy())

    scores = []
    for true, pred in zip(y_trues, y_preds):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)

    if return_preds:
        return avg_score, y_trues, y_preds, scores
    return avg_score


def check_preds(preds, lbls):
    preds = tf.cast(preds, tf.int64)

    lbls = strategy.gather(lbls, axis=0)
    lbls = tf.cast(lbls, tf.int64)

    y_true = tokenizer.predict_captions(lbls.numpy())
    y_pred = tokenizer.predict_captions(preds.numpy())
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        print('*'*100)
        print('preds :', pred)
        print('labels:', true)
        print('score:', Levenshtein.distance(true, pred))
        if i == 30:
            break


def check_test_preds(preds):
    preds = tf.cast(preds, tf.int64)
    y_pred = tokenizer.predict_captions(preds.numpy())
    for i, pred in enumerate(y_pred):
        print('*'*100)
        print('preds :', pred)
        if i == 30:
            break
#########################################################################################################
# LOOP
#########################################################################################################


class Trainer:
    def __init__(self, encoder, decoder, optimizer, scheduler, loss_fn, metric_fn, num_epochs=100, resume=False, resume_epoch=0, steps_per_epoch=None):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.num_epochs = num_epochs
        self.resume = resume

        self.init_epoch = 1
        self.total_steps = 0
        if self.resume:
            if opt.resume_from_last:
                last_step = int(tf.io.gfile.glob(
                    f'{opt.save_dir}/encoder_step*')[-1].strip('').split('step')[1][:6])
                self.init_epoch = last_step // opt.steps_per_epoch + 1
                self.total_steps = last_step
                encoder_path = f'{opt.save_dir}/encoder_step{self.total_steps:06}.ckpt'
                decoder_path = f'{opt.save_dir}/decoder_step{self.total_steps:06}.ckpt'
            elif opt.resume_steps:
                self.init_epoch = opt.resume_steps // opt.steps_per_epoch
                self.total_steps = opt.resume_steps + 1
                encoder_path = f'{opt.save_dir}/encoder_step{self.total_steps:06}.ckpt'
                decoder_path = f'{opt.save_dir}/decoder_step{self.total_steps:06}.ckpt'
            else:
                self.init_epoch = resume_epoch
                self.total_steps = steps_per_epoch * (resume_epoch - 1)
                encoder_path = f'{opt.save_dir}/best_encoder.ckpt'
                decoder_path = f'{opt.save_dir}/best_decoder.ckpt'
            self.encoder.load_weights(encoder_path)
            self.decoder.load_weights(decoder_path)
            self.scheduler.step(self.total_steps)
            print('load encoder from:', encoder_path)
            print('load decoder from:', decoder_path)

    def train_step(self, images, labels):
        labels_input = labels[:, :-1]
        labels_target = labels[:, 1:]
        seq_len = labels_target.shape[1]
        dec_mask = create_masks_decoder(labels_target)

        with tf.GradientTape() as tape:
            enc_output = self.encoder(images, training=True)
            predictions = self.decoder(
                labels_input, enc_output, training=True, look_ahead_mask=dec_mask)
            loss = self.loss_fn(labels_target, predictions)
            self.metric_fn.update_state(labels_target, predictions)

        # backpropagation using variables, gradients and loss
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # divide total loss by number of predictions
        batch_loss = loss / (opt.seq_len - 1)

        return batch_loss

    @tf.function
    def distributed_train_step(self, images, labels):
        # Performs a distributed train step.
        # required as a TPU v3-8 consists of 8 computation units
        per_replica_losses = strategy.run(
            self.train_step, args=(images, labels))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM,
                               per_replica_losses, axis=None)
        return loss

    def validation_step(self, images, labels):
        total_loss = 0.0

        enc_output = self.encoder(images, training=False)

        batch_size = tf.shape(images)[0]
        output = tf.fill([batch_size, 1], value=START_TOKEN)
        output = tf.cast(output, tf.int32)

        # Teacher forcing - feeding the target as the next input
        for t in tqdm(range(1, opt.seq_len)):
            # passing enc_output to the self.decoder
            dec_mask = create_masks_decoder(output)
            predictions = self.decoder(
                output, enc_output, False, dec_mask)
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            # add loss

            loss = self.loss_fn(labels[:, t], tf.squeeze(predictions))
            total_loss += loss
            self.metric_fn.update_state(labels[:, t], tf.squeeze(predictions))

            # no teacher forcing, predicted char is next LSTMCell input
            dec_input = tf.math.argmax(
                predictions, axis=-1, output_type=tf.int32)
            # dec_input = tf.expand_dims(dec_input, axis=1)
            # add predicted character to predictions sequence
            output = tf.concat([output, dec_input], axis=1)

        return total_loss, output

    @tf.function
    def distributed_val_step(self, images_val, labels_val):
        per_replica_losses, per_replica_predictions_seq = strategy.run(
            self.validation_step, args=(images_val, labels_val))
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM,
                               per_replica_losses, axis=None)
        predictions_seq = strategy.gather(per_replica_predictions_seq, axis=0)

        return loss, predictions_seq

    def evaluate(self, val_dist_dataset):
        total_loss = 0.0
        total_ls_distance = 0.0
        total_acc = 0.0
        results = {}
        results['preds'] = []
        results['labels'] = []
        results['scores'] = []

        for step, (images, labels) in tqdm(enumerate(val_dist_dataset)):
            batch_loss, predictions_seq = self.distributed_val_step(
                images, labels)
            levenshtein_distance, text_labels, text_preds, scores = get_levenshtein_distance(
                predictions_seq, labels)

            results['preds'].extend(text_preds)
            results['labels'].extend(text_labels)
            results['scores'].extend(scores)

            if step == 0:
                check_preds(predictions_seq, labels)

            total_loss += batch_loss / opt.val_steps
            total_ls_distance += levenshtein_distance / opt.val_steps
            total_acc += self.metric_fn.result() / opt.val_steps
            self.metric_fn.reset_states()

            if step + 1 == opt.val_steps:
                return total_loss, total_ls_distance, total_acc, results

    def test_step(self, images):
        enc_output = self.encoder(images, training=False)

        batch_size = tf.shape(images)[0]
        output = tf.fill([batch_size, 1], value=START_TOKEN)
        output = tf.cast(output, tf.int32)

        # Teacher forcing - feeding the target as the next input
        for t in tqdm(range(1, opt.seq_len)):
            # passing enc_output to the self.decoder
            dec_mask = create_masks_decoder(output)
            predictions = self.decoder(
                output, enc_output, False, dec_mask)
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            # no teacher forcing, predicted char is next LSTMCell input
            dec_input = tf.math.argmax(
                predictions, axis=-1, output_type=tf.int32)

            # add predicted character to predictions sequence
            output = tf.concat([output, dec_input], axis=1)

        return output

    @tf.function
    def distributed_test_step(self, images):
        per_replica_predictions_seq = strategy.run(
            self.test_step, args=(images,))
        predictions_seq = strategy.gather(per_replica_predictions_seq, axis=0)

        return predictions_seq

    def predict(self, test_dataset, num_test_steps):
        test_dist_dataset = strategy.experimental_distribute_dataset(
            test_dataset)
        # test_dist_dataset = test_dataset
        all_predictions = {}

        for (step, (images, image_ids)) in tqdm(enumerate(test_dist_dataset), total=num_test_steps):
            predictions_seq = self.distributed_test_step(images)
            predictions_text = tokenizer.predict_captions(
                predictions_seq.numpy())
            image_ids = strategy.gather(image_ids, axis=0)

            if step == 0:
                check_test_preds(predictions_seq)

            for text, image_id in zip(predictions_text, image_ids):
                image_id = image_id.numpy().decode()
                all_predictions[image_id] = text

            if step == num_test_steps - 1:
                return all_predictions

    def fit(self, train_dataset, val_dataset):
        best_metric = 10e6
        for epoch in range(self.init_epoch, self.num_epochs + 1):
            print(f'***** EPOCH {epoch} *****')
            t_start = time.time()  # to compute epoch duration
            t_start_batch = time.time()  # to compute batch duration
            total_loss = 0.0

            # create distributed versions of dataset to run on TPU with 8 computation units
            train_dist_dataset = strategy.experimental_distribute_dataset(
                train_dataset)
            val_dist_dataset = strategy.experimental_distribute_dataset(
                val_dataset)

            for (step, (images, labels)) in enumerate(train_dist_dataset):
                self.total_steps += 1
                step += 1
                batch_loss = self.distributed_train_step(images, labels)
                batch_loss = tf.cast(batch_loss, tf.float32)

                wandb.log({'train_loss': batch_loss,
                           'train_accuracy': self.metric_fn.result()})

                # end of epoch validation step
                if step == opt.steps_per_epoch:
                    val_loss, val_ls_distance, val_acc, results = self.evaluate(
                        val_dist_dataset)

                    with tf.io.gfile.GFile(f'{opt.save_dir}/val_results.json', 'w') as f:
                        json.dump(results, f)

                    # log with validation
                    val_loss = tf.cast(val_loss, tf.float32)
                    wandb.log(
                        {'val_loss': val_loss, 'val_ls_distance': val_ls_distance, 'val_acc': val_acc})
                    log(step, batch_loss, t_start_batch,
                        val_loss, val_ls_distance, val_acc, self.scheduler.get_lr())
                    self.metric_fn.reset_states()

                    if val_ls_distance < best_metric:
                        print('best updated to ', val_ls_distance)
                        best_metric = val_ls_distance
                        self.encoder.save_weights(
                            f'{opt.save_dir}/best_encoder.ckpt')
                        self.decoder.save_weights(
                            f'{opt.save_dir}/best_decoder.ckpt')

                # verbose logging step
                elif step % opt.verbose_freq == 0:
                    log(step, batch_loss, t_start_batch,
                        None, None, None, self.scheduler.get_lr())
                    self.metric_fn.reset_states()
                    # reset start time batch
                    t_start_batch = time.time()

                if self.total_steps % opt.save_freq == 0:
                    self.encoder.save_weights(
                        f'{opt.save_dir}/encoder_step{self.total_steps:06}.ckpt')
                    self.decoder.save_weights(
                        f'{opt.save_dir}/decoder_step{self.total_steps:06}.ckpt')

                total_loss += batch_loss

                # stop training when NaN loss is detected
                if step == opt.steps_per_epoch or np.isnan(total_loss):
                    break

                # ste learning rate
                self.scheduler.step(self.total_steps)

            # stop training when NaN loss is detected
            if np.isnan(total_loss):
                break

            print(
                f'Epoch {epoch} Loss {round(total_loss.numpy() / opt.steps_per_epoch, 3)}, time: {int(time.time() - t_start)} sec\n')


#########################################################################################################
# MAIN
#########################################################################################################
# initialize the model, a dummy call to the encoder and deocder is made to allow the summaries to be printed
with strategy.scope():
    # Sparse categorical cross entropy loss is used
    loss_object = FocalLoss(reduction=tf.keras.losses.Reduction.NONE)

    def loss_function(real, pred):
        per_example_loss = loss_object(real, pred)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=opt.batch_size)

    # Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    # Encoder
    encoder = Encoder(
        opt.d_model,
        opt.encoder_drop_rate
    )

    # Decoder
    decoder = Decoder(
        opt.num_layers,
        opt.d_model,
        opt.num_heads,
        opt.dff,
        opt.vocab_size,
        opt.seq_len,
        opt.decoder_drop_rate
    )

    # Adam Optimizer
    optimizer = AdamWeightDecay()

    lr_fn = [lrfn(step, 1e-6, 1e-3, 1e-5, opt.num_epochs)
             for step in range(opt.total_steps)]
    scheduler = LRReduce(optimizer, lr_fn)

    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_function,
        metric_fn=train_accuracy,
        num_epochs=opt.num_epochs,
        resume=opt.resume,
        resume_epoch=opt.resume_epoch,
        steps_per_epoch=opt.steps_per_epoch
    )

if opt.inference:
    test_dataset, test_length = get_test_dataset(opt.test_batch_size)
    num_test_steps = test_length // opt.batch_size + 1
    all_predictions = trainer.predict(test_dataset, num_test_steps)
    with tf.io.gfile.GFile(f'{opt.save_dir}/test_results.json', 'w') as f:
        json.dump(all_predictions, f)

else:
    id_ = opt.exp_id.split('_')[0]
    wandb.init(project='bms-tf-keras-baseline', id=id_, resume="allow")
    train_dataset, train_length = get_dataset('train', opt.batch_size)
    val_dataset, val_length = get_dataset('val', opt.batch_size)
    print('train samples:', train_length)
    print('val samples:', val_length)

    trainer.fit(train_dataset, val_dataset)
