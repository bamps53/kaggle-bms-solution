from .rdkit_utils import normalize_inchi_batch
import os
import tensorflow as tf
import json
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


YOUR_GCS_DIR = 'gs://model_storage53'
SUB_PATH = '../input/bms-molecular-translation/sample_submission.csv'


def load_test_preds(exp_id):
    src = f'{YOUR_GCS_DIR}/{exp_id}/test_results.json'
    dst = os.path.basename(src)
    tf.io.gfile.copy(src, dst, overwrite=True)

    sub = pd.read_csv(SUB_PATH)
    data = json.load(open('test_results.json'))

    sub['InChI'] = sub['image_id'].map(data)
    sub['InChI'] = 'InChI=1S/' + sub['InChI']

    return sub


def normalize(df, in_column='InChI', out_column='norm_InChI'):
    df[out_column] = normalize_inchi_batch(df[in_column])
    df["is_valid"] = ~df[out_column].isna()
    df[out_column] = df[out_column].where(df.is_valid, df[in_column])
    return df


def fix_strange_inchi(df, in_column='InChI', out_column='norm_InChI'):
    # fix strange normalization
    df['has_question'] = df[out_column].map(lambda x: '?' in x)
    df['has_q'] = df[out_column].map(lambda x: '/q' in x)
    df['has_p'] = df[out_column].map(lambda x: '/p' in x)

    df.loc[df['has_question'], out_column] = df.loc[df['has_question'], in_column]
    df.loc[df['has_q'], out_column] = df.loc[df['has_q'], in_column]
    df.loc[df['has_p'], out_column] = df.loc[df['has_p'], in_column]
    return df


def fix_columns(df):
    df = df[['image_id', 'norm_InChI', 'is_valid']]
    df = df.rename(columns={'norm_InChI': 'InChI'})
    return df


def convert_test_preds(exp_id):
    test_df = load_test_preds(exp_id)
    test_df = normalize(test_df)
    test_df = fix_strange_inchi(test_df)
    test_df = fix_columns(test_df)
    return test_df


if __name__ == '__main__':
    os.makedirs('./output', exist_ok=True)

    exp_ids = [
        'exp072_300_600_transformer_b4_2layers_seqlen200_fix_pos_random_crop_focal',
        'exp084_416_736_transformer_b4_2layers_seqlen200_fix_pos_random_crop_focal',
        'exp090_300_600_en_de_transformer_b4_2layers_seqlen200_fix_pos_random_crop_focal_pre_norm',
        'exp103_416x736_transformer_b4_2layers_seqlen200_salt_pepper',
        'exp0845_finetune_with_strong_noise_lb060_pseudo',
        'exp1031_416x736_transformer_b4_2layers_seqlen200_finetune_with_noise_denoise'
    ]

    for exp_id in exp_ids:
        id_ = exp_id.split('_')[0]
        test_df = convert_test_preds(exp_id)
        test_df.to_csv(f'./output/{id_}_normalized.csv', index=False)
