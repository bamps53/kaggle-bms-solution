import tensorflow as tf
from .datasets import GCS_PATHS


def update_config(CFG, gcs_dir, is_tpu, num_replicas):
    CFG.total_steps = CFG.num_epochs * CFG.steps_per_epoch

    CFG.row_size = (CFG.image_height - 1) // 32 + 1
    CFG.col_size = (CFG.image_width - 1) // 32 + 1

    CFG.train_gcs_dir = GCS_PATHS[CFG.train_gcs_dir]
    CFG.val_gcs_dir = GCS_PATHS[CFG.val_gcs_dir]
    CFG.test_gcs_dir = GCS_PATHS[CFG.test_gcs_dir]

    CFG.pad_token = tf.constant(CFG.pad_token, dtype=tf.int64)
    CFG.start_token = tf.constant(CFG.start_token, dtype=tf.int64)
    CFG.end_token = tf.constant(CFG.end_token, dtype=tf.int64)

    CFG.dtype = tf.bfloat16 if is_tpu else tf.float32
    CFG.save_dir = f'{gcs_dir}/{CFG.exp_id}'

    CFG.batch_size = CFG.batch_size_base * num_replicas
    CFG.test_batch_size = CFG.batch_size_base * num_replicas

    CFG.init_lr = CFG.init_lr * CFG.batch_size_base / 32
    CFG.final_lr = CFG.final_lr * CFG.batch_size_base / 32
    return CFG
