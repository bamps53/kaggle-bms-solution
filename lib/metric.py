import Levenshtein
import numpy as np
import tensorflow as tf

from lib.tokenizer import Tokenizer


class Evaluator():
    def __init__(self, num_show=30):
        self.tokenizer = Tokenizer()
        self.num_show = num_show

    def _seq2inchi(self, seq):
        seq = tf.cast(seq, tf.int64)
        inchi = self.tokenizer.predict_captions(seq.numpy())
        return inchi

    def calc_score(self, preds, labels, return_preds=True):
        preds = self._seq2inchi(preds)
        labels = self._seq2inchi(labels)

        scores = []
        for pred, label in zip(preds, labels):
            score = Levenshtein.distance(label, pred)
            scores.append(score)
        avg_score = np.mean(scores)

        if return_preds:
            return avg_score, labels, preds, scores
        return avg_score

    def check_preds(self, preds, labels):
        preds = self._seq2inchi(preds)
        labels = self._seq2inchi(labels)

        for i, (pred, label) in enumerate(zip(preds, labels)):
            print('*'*100)
            print('preds :', pred)
            print('labels:', label)
            print('score:', Levenshtein.distance(label, pred))
            if i == self.num_show:
                break

    def check_test_preds(self, preds):
        preds = self._seq2inchi(preds)
        for i, pred in enumerate(preds):
            print('*'*100)
            print('preds :', pred)
            if i == self.num_show:
                break
