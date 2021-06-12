from lib.tokenizer import Tokenizer
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
tqdm.pandas()


def main():
    n_folds = 20
    seed = 42

    tokenizer = Tokenizer()

    train = pd.read_csv('../input/bms-molecular-translation/train_labels.csv')
    train['InChI_1'] = train['InChI'].progress_apply(lambda x: x.split('/')[1])
    train['InChI_text'] = train['InChI_1'].progress_apply(tokenizer.split_form) + ' ' + \
        train['InChI'].progress_apply(
            lambda x: '/'.join(x.split('/')[2:])).progress_apply(tokenizer.split_form2).values
    train['InChI_length'] = train['InChI_text'].progress_apply(
        tokenizer.get_length)

    folds = train.copy()
    Fold = StratifiedKFold(n_splits=n_folds,
                           shuffle=True, random_state=seed)
    for n, (_, val_index) in enumerate(Fold.split(folds, folds['InChI_length'])):
        folds.loc[val_index, 'fold'] = int(n)
    print(folds.groupby(['fold']).size())

    folds.to_csv('data/folds.csv', index=False)


if __name__ == '__main__':
    main()
