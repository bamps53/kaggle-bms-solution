import json
import os

import tensorflow as tf
import tensorflow_addons as tfa
import wandb
import click
from omegaconf import OmegaConf


from lib.datasets import get_train_dataset, get_val_dataset, get_test_dataset
from lib.scheduler import get_scheduler
from lib.utils import seed_everything, get_strategy, set_policy
from lib.model import Encoder, Decoder
from lib.loss import FocalLoss
from lib.metric import Evaluator
from lib.trainer import Trainer
from lib.config import update_config

YOUR_GCS_DIR = 'gs://model_storage53'

@click.command()
@click.option('--config_path', '-c', type=str)
@click.option('--mode', '-m', type=click.Choice(['train', 'inference']))
@click.option('--resume', '-r', is_flag=True)
def main(config_path, mode, resume):
    seed_everything()

    strategy, is_tpu = get_strategy()
    dtype_ = tf.bfloat16 if is_tpu else tf.float32
    set_policy(is_tpu)
    num_replicas = strategy.num_replicas_in_sync
    print(f'num_replicas: {num_replicas}')

    CFG = OmegaConf.load(config_path)
    CFG = update_config(CFG, YOUR_GCS_DIR, is_tpu, num_replicas)
    os.makedirs(CFG.save_dir, exist_ok=True)

    
    pad_token = tf.constant(CFG.pad_token, dtype=tf.int32)
    start_token = tf.constant(CFG.start_token, dtype=tf.int32)

    with strategy.scope():
        loss_object = FocalLoss(
            pad_token=pad_token,
            sos_token=start_token,
            num_classes=CFG.vocab_size,
            reduction=tf.keras.losses.Reduction.NONE)

        def loss_function(real, pred):
            per_example_loss = loss_object(real, pred)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=CFG.batch_size)

        # Metrics
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        # Encoder
        if CFG.encoder_transformer:
            pre_norm = True
            encoder = Encoder(CFG.d_model, CFG.encoder_drop_rate, dtype_,
                              CFG.num_layers, CFG.num_heads, CFG.dff, CFG.row_size, CFG.col_size)
        else:
            pre_norm = False
            encoder = Encoder(CFG.d_model, CFG.encoder_drop_rate, dtype_)

        # Decoder
        decoder = Decoder(CFG.num_layers, CFG.d_model, CFG.num_heads,
                          CFG.dff, CFG.vocab_size, CFG.seq_len,
                          CFG.row_size, CFG.col_size, dtype_, CFG.decoder_drop_rate, pre_norm)

        optimizer = tfa.optimizers.AdamW(CFG.init_lr)
        scheduler = get_scheduler(optimizer, CFG.warmup_lr, CFG.init_lr,
                                  CFG.final_lr, CFG.num_total_steps, CFG.warmup_steps)

        evaluator = Evaluator()

        trainer = Trainer(encoder=encoder, decoder=decoder, optimizer=optimizer, scheduler=scheduler,
                          loss_fn=loss_function, metric_fn=train_accuracy, evaluator=evaluator,
                          pad_token=pad_token, start_token=start_token, seq_len=CFG.seq_len, strategy=strategy, 
                          dtype_=dtype_, num_epochs=CFG.num_epochs, resume=resume, steps_per_epoch=CFG.steps_per_epoch)

    if mode == 'inference':
        test_dataset, test_length = get_test_dataset(CFG.test_batch_size)
        num_test_steps = test_length // CFG.batch_size + 1
        all_predictions = trainer.predict(test_dataset, num_test_steps)
        with tf.io.gfile.GFile(f'{CFG.save_dir}/test_results.json', 'w') as f:
            json.dump(all_predictions, f)

    else:
        id_ = CFG.exp_id.split('_')[0]
        wandb.init(project='bms-tf-keras-baseline', id=id_, resume="allow")
        train_dataset, train_length = get_train_dataset(
            CFG.train_gcs_dir, CFG.batch_size, CFG.fold, dtype_,
            CFG.image_height, CFG.image_width, CFG.seq_len, CFG.label_dtype, CFG.gray_scale, CFG.rotate_angle, CFG.zoom_range, CFG.pseudo_gcs_dir)
        val_dataset, val_length = get_val_dataset(
            CFG.val_gcs_dir, CFG.test_batch_size, CFG.fold, dtype_,
            CFG.image_height, CFG.image_width, CFG.seq_len, CFG.label_dtype, CFG.gray_scale)
        trainer.fit(train_dataset, val_dataset)


if __name__ == '__main__':
    main()
