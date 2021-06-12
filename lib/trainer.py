import json
import time

import tensorflow as tf
from tensorflow.python.keras.backend import update
import wandb

from tqdm.notebook import tqdm

from lib.model import create_decoder_mask


def get_last_step(save_dir):
    files = tf.io.gfile.glob(f'{save_dir}/encoder_step*.ckpt.index')
    if len(files) > 0:
        files = [file.split('/')[-1] for file in files]
        files = [file.split('.')[0] for file in files]
        steps = sorted([int(file.split('step')[-1]) for file in files])
        last_step = max(steps)
        return last_step
    else:
        0


class Trainer:
    def __init__(self, encoder, decoder, optimizer, scheduler, loss_fn, metric_fn,
                 evaluator, pad_token, start_token, seq_len, strategy, dtype_, num_epochs=100,
                 eval_freq=1, log_freq=100, save_freq=5000, resume=False, steps_per_epoch=None,
                 resume_from=None, save_dir=None):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.evaluator = evaluator
        self.pad_token = pad_token
        self.start_token = start_token
        self.seq_len = seq_len
        self.strategy = strategy
        self.dtype_ = dtype_
        self.num_epochs = num_epochs
        self.resume = resume
        self.resume_from = resume_from
        self.steps_per_epoch = steps_per_epoch
        self.save_dir = save_dir
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.save_freq = save_freq

        self.init_epoch = 1
        self.total_steps = 0
        if self.resume:
            if self.resume_from:
                save_dir = resume_from
                last_step = get_last_step(save_dir)
                self.init_epoch = last_step // steps_per_epoch + 1
                encoder_path = f'{save_dir}/encoder_step{last_step:06}.ckpt'
                decoder_path = f'{save_dir}/decoder_step{last_step:06}.ckpt'
            else:
                files = tf.io.gfile.glob(f'{self.save_dir}/encoder_step*')
                last_step = 0
                if len(files) > 0:
                    last_step = get_last_step(self.save_dir)
                    self.init_epoch = last_step // steps_per_epoch + 1
                    self.total_steps = last_step
                    encoder_path = f'{self.save_dir}/encoder_step{self.total_steps:06}.ckpt'
                    decoder_path = f'{self.save_dir}/decoder_step{self.total_steps:06}.ckpt'
                else:
                    raise ValueError('There is no checkpoint to resume.')

            self.encoder.load_weights(encoder_path)
            self.decoder.load_weights(decoder_path)
            self.scheduler.step(self.total_steps)
            print('load encoder from:', encoder_path)
            print('load decoder from:', decoder_path)

    def train_step(self, images, labels):
        labels_input = labels[:, :-1]
        labels_target = labels[:, 1:]
        dec_mask = create_decoder_mask(labels_target, self.dtype_)

        with tf.GradientTape() as tape:
            enc_output = self.encoder(images, training=True)
            predictions = self.decoder(
                labels_input, enc_output, training=True, look_ahead_mask=dec_mask)
            loss = self.loss_fn(labels_target, predictions)

            is_pad = tf.not_equal(labels_target, self.pad_token)
            is_sos = tf.not_equal(labels_target, self.start_token)
            mask = tf.logical_or(is_pad, is_sos)
            mask = tf.cast(mask, predictions.dtype)[:, :, None]

            self.metric_fn.update_state(labels_target, predictions, mask)

        # backpropagation using variables, gradients and loss
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    @tf.function
    def distributed_train_step(self, images, labels):
        per_replica_losses = self.strategy.run(
            self.train_step, args=(images, labels))
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                    per_replica_losses, axis=None)
        return loss

    def validation_step(self, images, labels):
        total_loss = 0.0

        enc_output = self.encoder(images, training=False)

        batch_size = tf.shape(images)[0]
        output = tf.fill([batch_size, 1], value=self.start_token)
        output = tf.cast(output, tf.int32)

        # Teacher forcing - feeding the target as the next input
        for t in tqdm(range(1, self.seq_len)):
            dec_mask = create_decoder_mask(output, self.dtype_)
            predictions = self.decoder(
                output, enc_output, False, dec_mask)
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            loss = self.loss_fn(labels[:, t][:, None], predictions)
            total_loss += loss
            self.metric_fn.update_state(labels[:, t], tf.squeeze(predictions))

            dec_input = tf.math.argmax(
                predictions, axis=-1, output_type=tf.int32)
            output = tf.concat([output, dec_input], axis=1)

        return total_loss, output

    @tf.function
    def distributed_val_step(self, images, labels):
        per_replica_losses, per_replica_predictions = self.strategy.run(
            self.validation_step, args=(images, labels))
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                    per_replica_losses, axis=None)
        predictions = self.strategy.gather(per_replica_predictions, axis=0)
        labels = self.strategy.gather(labels, axis=0)

        return loss, predictions, labels

    def evaluate(self, val_dist_dataset, num_val_steps):
        total_loss = 0.0
        total_ls_distance = 0.0
        total_acc = 0.0
        results = {}
        results['preds'] = []
        results['labels'] = []
        results['scores'] = []

        for step, (images, labels) in tqdm(enumerate(val_dist_dataset)):
            batch_loss, predictions, labels = self.distributed_val_step(
                images, labels)
            levenshtein_distance, text_labels, text_preds, scores = self.evaluator.calc_score(
                predictions, labels)

            results['preds'].extend(text_preds)
            results['labels'].extend(text_labels)
            results['scores'].extend(scores)

            if step == 0:
                self.evaluator.check_preds(predictions, labels)

            total_loss += batch_loss / num_val_steps
            total_ls_distance += levenshtein_distance / num_val_steps
            total_acc += self.metric_fn.result() / num_val_steps
            self.metric_fn.reset_states()

            if step + 1 == num_val_steps:
                return total_loss, total_ls_distance, total_acc, results

    def test_step(self, images):
        enc_output = self.encoder(images, training=False)

        batch_size = tf.shape(images)[0]
        output = tf.fill([batch_size, 1], value=self.start_token)
        output = tf.cast(output, tf.int32)

        for _ in tqdm(range(1, self.seq_len)):
            dec_mask = create_decoder_mask(output, self.dtype_)
            predictions = self.decoder(
                output, enc_output, False, dec_mask)
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            dec_input = tf.math.argmax(
                predictions, axis=-1, output_type=tf.int32)
            output = tf.concat([output, dec_input], axis=1)

        return output

    @tf.function
    def distributed_test_step(self, images):
        per_replica_predictions = self.strategy.run(
            self.test_step, args=(images,))
        predictions = self.strategy.gather(per_replica_predictions, axis=0)

        return predictions

    def predict(self, test_dataset, num_test_steps):
        test_dist_dataset = self.strategy.experimental_distribute_dataset(
            test_dataset)

        all_predictions = {}
        for (step, (images, image_ids)) in tqdm(enumerate(test_dist_dataset), total=num_test_steps):
            predictions = self.distributed_test_step(images)
            predictions_text = self.evaluator.tokenizer.predict_captions(
                predictions.numpy())
            image_ids = self.strategy.gather(image_ids, axis=0)

            if step == 0:
                self.evaluator.check_test_preds(predictions)

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
            train_dist_dataset = self.strategy.experimental_distribute_dataset(
                train_dataset)
            val_dist_dataset = self.strategy.experimental_distribute_dataset(
                val_dataset)

            for (step, (images, labels)) in enumerate(train_dist_dataset):
                self.total_steps += 1
                step += 1
                batch_loss = self.distributed_train_step(images, labels)
                batch_loss = tf.cast(batch_loss, tf.float32)

                wandb.log({'train_loss': batch_loss,
                           'train_accuracy': self.metric_fn.result()})

                # end of epoch validation step
                if step == self.steps_per_epoch and epoch % self.eval_freq == 0:
                    val_loss, val_ls_distance, val_acc, results = self.evaluate(
                        val_dist_dataset)

                    with tf.io.gfile.GFile(f'{self.save_dir}/val_results.json', 'w') as f:
                        json.dump(results, f)

                    # log with validation
                    val_loss = tf.cast(val_loss, tf.float32)
                    wandb.log(
                        {'val_loss': val_loss, 'val_ls_distance': val_ls_distance, 'val_acc': val_acc})
                    self.log(step, batch_loss, t_start_batch,
                             val_loss, val_ls_distance, val_acc, self.scheduler.get_lr())
                    self.metric_fn.reset_states()

                    if val_ls_distance < best_metric:
                        print('best updated to ', val_ls_distance)
                        best_metric = val_ls_distance
                        self.encoder.save_weights(
                            f'{self.save_dir}/best_encoder.ckpt')
                        self.decoder.save_weights(
                            f'{self.save_dir}/best_decoder.ckpt')

                # verbose logging step
                elif step % self.log_freq == 0:
                    self.log(step, batch_loss, t_start_batch,
                             None, None, None, self.scheduler.get_lr())
                    self.metric_fn.reset_states()
                    # reset start time batch
                    t_start_batch = time.time()

                if self.total_steps % self.save_freq == 0:
                    self.encoder.save_weights(
                        f'{self.save_dir}/encoder_step{self.total_steps:06}.ckpt')
                    self.decoder.save_weights(
                        f'{self.save_dir}/decoder_step{self.total_steps:06}.ckpt')

                total_loss += batch_loss

                if step == self.steps_per_epoch:
                    break

                # ste learning rate
                self.scheduler.step(self.total_steps)

            print(
                f'Epoch {epoch} Loss {round(total_loss.numpy() / self.steps_per_epoch, 3)}, time: {int(time.time() - t_start)} sec\n')

    def log(self, batch, loss, t_start_batch, val_loss, val_ls_distance, val_acc, lr):
        # training metrics
        print(
            f'Step %s|' % f'{batch}/{self.steps_per_epoch}'.ljust(9, ' '),
            f'loss: %.3f,' % loss,
            f'acc: %.3f, ' % self.metric_fn.result(),
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
