import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import List

import cv2
import efficientnet.tfkeras as efn
import Levenshtein
import numpy as np
import pandas as pd
import swifter
import tensorflow as tf
import torch
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tqdm.auto import tqdm
tqdm.pandas()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


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
    # strategy = tf.distribute.get_strategy()
    strategy = tf.distribute.MirroredStrategy()

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
    exp_id = 'exp0845_finetune_with_strong_noise_lb060_pseudo'
    debug: bool = False
    validate: bool = True
    inference: bool = False

    steps_per_epoch: int = 100000
    batch_size_base: int = 8
    batch_size: int = batch_size_base * REPLICAS
    test_batch_size_base: int = 32
    test_batch_size: int = test_batch_size_base * REPLICAS
    num_epochs: int = 5
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
    train_gcs_dir: str = GCS_PATHS['416x736_no_pad']
    val_gcs_dir: str = GCS_PATHS['416x736_no_pad']
    test_gcs_dir: str = GCS_PATHS['416x736_no_pad_test']
    label_gcs_dir: str = GCS_PATHS['exp084_val']
    val_size: int = 121210
    val_steps: int = val_size // batch_size
    rotate_angle: int = 5

    # configure model
    vocab_size: int = 193
    seq_len: int = 200
    beam_size: int = 5
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

PAD_TOKEN = tf.constant(opt.pad_token, dtype=tf.int32)
START_TOKEN = tf.constant(opt.start_token, dtype=tf.int32)
END_TOKEN = tf.constant(opt.end_token, dtype=tf.int32)

if opt.debug:
    opt.steps_per_epoch = 10
    opt.val_steps = 10
    opt.num_epochs = 5
    opt.steps_per_execution = 1


#########################################################################################################
# tfrecords utils
#########################################################################################################

def split_form(form):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')


def split_form2(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')


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

    def tokenize(self, inchi, max_len=277):
        inchi_1 = split_form(inchi.split('/')[1])
        inchi_others = split_form2('/'.join(inchi.split('/')[2:]))
        inchi = inchi_1 + ' ' + inchi_others
        try:
            x = np.array(tokenizer.text_to_sequence(inchi), np.int32)
            assert len(x) <= max_len, len(x)
            x = np.pad(x, (0, max_len - len(x)),
                       constant_values=tokenizer.stoi["<pad>"])
            return x
            # return x.tolist()
        except:
            return [self.stoi['<pad>']] * max_len
            # return np.ones(max_len, dtype=np.int32) * self.stoi['<pad>']


tokenizer = Tokenizer()

#########################################################################################################
# DATASET
#########################################################################################################
AUTO = tf.data.experimental.AUTOTUNE


def imagenet_normalize(img, labels, dtype_=tf.float32):
    IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = tf.cast(img, dtype_)
    return img, labels


def read_image(file_name, label, height=416, width=736, seq_len=200):
    bits = tf.io.read_file(file_name)
    img = tf.image.decode_png(bits, channels=3)
    img = tf.reshape(img, (height, width, 3))
    label = label[:seq_len]
    label = (label + 1) % 193  # to set pad token as 0
    label = tf.cast(label, tf.int32)
    return img, label


def get_test_dataset(df_path, batch_size=32):
    AUTO = tf.data.experimental.AUTOTUNE

    df = pd.read_csv(df_path)
    df = df.rename(columns={'normed_InChI': 'InChI'})
    df = df[['image_id', 'InChI']]

    file_paths = df['image_id'].map(
        lambda x: f'../input/test_resized416x736_no_pad/{x}.png').tolist()
    labels = df['InChI'].swifter.apply(tokenizer.tokenize).tolist()

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(read_image, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(imagenet_normalize, num_parallel_calls=AUTO)
    ds = ds.prefetch(1)
    ds_len = len(df)

    return ds, ds_len


def get_levenshtein_distance(preds, labels, return_preds=True, do_beam_search=False):
    preds = tf.cast(preds, tf.int32)
    labels = tf.cast(labels, tf.int32)

    if do_beam_search:
        print(preds.shape)
        preds = preds[:, 0, :]
        # preds = flatten_beam_dim(preds)
        # labels = expand_to_beam_size(labels, opt.beam_size)
        # labels = flatten_beam_dim(labels)

    y_trues = tokenizer.predict_captions(labels.numpy())
    y_preds = tokenizer.predict_captions(preds.numpy())

    scores = []
    for true, pred in zip(y_trues, y_preds):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)

    if return_preds:
        return avg_score, y_trues, y_preds, scores
    return avg_score


def check_preds(preds, labels, do_beam_search):
    preds = tf.cast(preds, tf.int32)

    labels = tf.cast(labels, tf.int32)

    if do_beam_search:
        print(preds.shape)
        preds = preds[:, 0, :]

    y_true = tokenizer.predict_captions(labels.numpy())
    y_pred = tokenizer.predict_captions(preds.numpy())
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        print('*'*100)
        print('tokens:', preds[i])
        print('preds :', pred)
        print('labels:', true)
        print('score:', Levenshtein.distance(true, pred))
        if i == 30:
            break


def check_test_preds(preds):
    preds = tf.cast(preds, tf.int32)
    y_pred = tokenizer.predict_captions(preds.numpy())
    for i, pred in enumerate(y_pred):
        print('*'*100)
        print('preds :', pred)
        if i == 30:
            break

#########################################################################################################
# UTILS
#########################################################################################################


def load_weights_from_last(exp_id, encoder, decoder):
    gcs_dir = f'gs://model_storage53/{exp_id}'
    last_step = int(tf.io.gfile.glob(
        f'{gcs_dir}/encoder_step*')[-1].strip('').split('step')[1][:6])
    encoder_path = f'{gcs_dir}/encoder_step{last_step:06}.ckpt'
    decoder_path = f'{gcs_dir}/decoder_step{last_step:06}.ckpt'

    encoder.load_weights(encoder_path)
    decoder.load_weights(decoder_path)
    print('load encoder from:', encoder_path)
    print('load decoder from:', decoder_path)

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

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def proj_pos_split(self, x, pos, proj_func):
        x = proj_func(x)  # (batch_size, seq_len, d_model)
        if pos is not None:
            x = x + pos
        x = self.split_heads(x)
        return x

    def resolve_past(self, x, x_past, x_pos, proj_func):
        # encoder key/value case. use only same past every time.
        if x is None:
            return x_past

        # First seq.
        if x_past is None:
            return self.proj_pos_split(x, x_pos, proj_func)

        # decoder key value case.
        # x should be (bs, seq_len, vocab_size)

        x = x[:, -1:, :]
        if x_pos is not None:
            x_pos = x_pos[:, -1:, :]
        x = self.proj_pos_split(x, x_pos, proj_func)
        x = tf.concat([x_past, x], axis=2)
        return x

    def attend(self, q, k, v, mask):
        batch_size = q.shape[0]
        scaled_attention, _ = scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        scaled_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        return scaled_attention

    def call(self, q, k, v, q_pos=None, k_pos=None, past=None, mask=None):
        if past is None:
            k_past = None
            v_past = None
        else:
            k_past = past['key']
            v_past = past['value']
        q = self.proj_pos_split(q, q_pos, self.wq)
        k = self.resolve_past(k, k_past, k_pos, self.wk)
        v = self.resolve_past(v, v_past, None, self.wv)
        present = {'key': k, 'value': v}

        attention = self.attend(q, k, v, mask)
        output = self.dense(attention)
        return output, present


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

    def call(self, x, enc_output, enc_pos, dec_pos, training, look_ahead_mask=None, padding_mask=None, pasts=None):
        if pasts is None:
            past1 = None
            past2 = None
        else:
            past1, past2 = pasts

        attn1, present1 = self.mha1(
            x, x, x, q_pos=dec_pos, k_pos=dec_pos, past=past1, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        if past2 is None:
            attn2, present2 = self.mha2(
                out1, enc_output, enc_output, q_pos=dec_pos, k_pos=enc_pos, past=past2)
        else:
            attn2, present2 = self.mha2(
                out1, None, None, q_pos=dec_pos, k_pos=enc_pos, past=past2)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, (present1, present2)


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

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None, layer_pasts=None):
        # TODO change to past1 and past2
        if layer_pasts is None:
            layer_pasts = [None] * self.num_layers

        seq_len = tf.shape(x)[1]
        dec_pos = self.pos_encoding_1d[:, :seq_len, :]
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, opt.dtype))

        for i in range(self.num_layers):
            x, layer_pasts[i] = self.dec_layers[i](
                x, enc_output, self.pos_encoding_2d, dec_pos, training, look_ahead_mask, padding_mask, layer_pasts[i])

        predictions = self.final_layer(x)
        return predictions, layer_pasts


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
            mask = tf.cast(tf.not_equal(y_true, PAD_TOKEN),
                           tf.float32)
            num_valid = tf.reduce_sum(mask, axis=1)
            mask = mask[:, :, None]

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
            valid_loss = tf.reduce_sum(weighted_loss * mask, axis=[1, 2])
            valid_loss = valid_loss / num_valid
        return valid_loss

    def get_config(self):
        config = {
            'alpha': self._alpha,
            'gamma': self._gamma,
        }
        base_config = super(FocalLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#########################################################################################################
# LOOP
#########################################################################################################


class Scorer:
    def __init__(self, exp_id, encoder, decoder, focal_loss_fn, ce_loss_fn):
        self.encoder = encoder
        self.decoder = decoder
        self.focal_loss_fn = focal_loss_fn
        self.ce_loss_fn = ce_loss_fn
        load_weights_from_last(exp_id, encoder, decoder)

    def score_step(self, images, labels):
        labels_input = labels[:, :-1]
        labels_target = labels[:, 1:]
        dec_mask = create_masks_decoder(labels_target)

        enc_output = self.encoder(images, training=False)
        predictions, _ = self.decoder(
            labels_input, enc_output, training=False, look_ahead_mask=dec_mask)
        focal_loss = self.focal_loss_fn(labels_target, predictions)
        ce_loss = self.ce_loss_fn(labels_target, predictions)
        return focal_loss, ce_loss

    @tf.function
    def distributed_score_step(self, images, labels):
        per_replica_focal_loss, per_replica_ce_loss = strategy.run(
            self.score_step, args=(images, labels))
        focal_loss = strategy.gather(per_replica_focal_loss, axis=0)
        ce_loss = strategy.gather(per_replica_ce_loss, axis=0)
        return focal_loss, ce_loss

    # def score(self, test_dataset, num_test_steps):
    #     test_dist_dataset = strategy.experimental_distribute_dataset(
    #         test_dataset)
    #     focal_losses = []
    #     ce_losses = []
    #     for step, (images, labels) in tqdm(enumerate(test_dist_dataset), total=num_test_steps):
    #         focal_loss, ce_loss = scorer.distributed_score_step(images, labels)
    #         focal_losses.append(focal_loss.numpy())
    #         ce_losses.append(ce_loss.numpy())
    #     return focal_losses, ce_losses
    def score(self, test_dataset, num_test_steps):
        focal_losses = []
        ce_losses = []
        for step, (images, labels) in tqdm(enumerate(test_dataset), total=num_test_steps):
            focal_loss, ce_loss = scorer.score_step(images, labels)
            focal_losses.append(focal_loss.numpy())
            ce_losses.append(ce_loss.numpy())
        return focal_losses, ce_losses


#########################################################################################################
# MAIN
#########################################################################################################
if __name__ == '__main__':

    with strategy.scope():
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

        focal_loss_object = FocalLoss(reduction=tf.keras.losses.Reduction.NONE)

        def focal_loss_function(y_true, y_pred):
            return focal_loss_object(y_true, y_pred)

        ce_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        def ce_loss_function(y_true, y_pred):
            y_pred = tf.cast(y_pred, dtype=tf.float32)
            mask = tf.cast(tf.not_equal(y_true, PAD_TOKEN),
                           tf.float32)
            num_valid = tf.reduce_sum(mask, axis=1)
            ce_loss = ce_loss_object(y_true, y_pred)
            ce_loss = tf.reduce_sum(ce_loss * mask, axis=1)
            ce_loss = ce_loss / num_valid
            return ce_loss

        scorer = Scorer(
            exp_id=opt.exp_id,
            encoder=encoder,
            decoder=decoder,
            focal_loss_fn=focal_loss_function,
            ce_loss_fn=ce_loss_function,
        )


def evaluate(exp_id, df_name):
    id_ = exp_id.split('_')[0]
    os.makedirs(id_, exist_ok=True)
    save_name = os.path.join(id_, os.path.basename(df_name))

    test_dataset, test_length = get_test_dataset(df_name, CFG.test_batch_size)
    num_test_steps = test_length // CFG.test_batch_size + 1

    focal_losses, ce_losses = scorer.score(test_dataset, num_test_steps)

    df = pd.read_csv(df_name)
    df['focal_score'] = np.concatenate(focal_losses)[:len(df)]
    df['ce_score'] = np.concatenate(ce_losses)[:len(df)]
    df.to_csv(save_name, index=False)


files = """
../input/kf-bms-candidates-v2/test_kf_0523.csv
../input/kf-bms-candidates-v2/test_kf_0525.csv
../input/kf-bms-candidates-v2/test_kf_0527.csv
../input/kf-bms-candidates-v2/test_yokoo_0527.csv
../input/kf-bms-candidates-v2/test_camaro_0525.csv
../input/kf-bms-candidates-v2/test_yokoo_0531.csv
../input/kf-bms-candidates-v2/test_kf_0531_renormed.csv
../input/kf-bms-candidates-v2/test_camaro_old_submissions.csv
../input/kf-bms-candidates-v2/test_kf_0531.csv
../input/kf-bms-candidates-v2/test_camaro_0531.csv
../input/kf-bms-candidates-v2/test_yokoo_0601.csv
""".split()

for file in files:
    evaluate(opt.exp_id, file)
