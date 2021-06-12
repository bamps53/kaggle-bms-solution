import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def seed_everything():
    # seed everything
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def get_strategy():
    try:
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
        strategy = tf.distribute.get_strategy()
    return strategy


def set_policy(strategy):
    is_tpu = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
    mixed_precision.set_policy('mixed_bfloat16' if is_tpu else 'float32')
    tf.config.optimizer.set_jit(True)
    print(f'Compute dtype: {mixed_precision.global_policy().compute_dtype}')
    print(f'Variable dtype: {mixed_precision.global_policy().variable_dtype}')
