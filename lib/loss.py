import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self,
                 pad_token,
                 sos_token,
                 num_classes,
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
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.num_classes = num_classes
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
            is_pad = tf.not_equal(y_true, self.pad_token)
            is_sos = tf.not_equal(y_true, self.sos_token)
            mask = tf.logical_or(is_pad, is_sos)
            mask = tf.cast(mask, tf.float32)[:, :, None]

            y_true = tf.one_hot(y_true, self.num_classes)
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
        return weighted_loss * mask

    def get_config(self):
        config = {
            'alpha': self._alpha,
            'gamma': self._gamma,
        }
        base_config = super(FocalLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FocalLossForRescore(tf.keras.losses.Loss):
    def __init__(self,
                 pad_token,
                 sos_token,
                 num_classes,
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
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.num_classes = num_classes
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
            mask = tf.cast(tf.not_equal(y_true, self.pad_token),
                           tf.float32)
            num_valid = tf.reduce_sum(mask, axis=1)
            mask = mask[:, :, None]

            y_true = tf.one_hot(y_true, self.num_classes)
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