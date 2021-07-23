from __future__ import division

import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction


class SSDLoss(Loss):
    def __init__(self, alpha=1.0, ratio=3, min_neg=3, name='SSDLoss', reduction=Reduction.NONE, batch=None):
        """
        Loss function of the Single Shot MultiBox Detector Architecture. It combines a smooth-L1 Loss for localization
        and cross-entropy for classification.
        Default reduction is 'SUM_OVER_BATCH_SIZE', if the loss is given as a function to compile.
        See: keras/engine/training_utils.py Line: 707
        :param alpha: a factor that weights the localization loss against the confidence loss
        :param ratio: the ratio between positives and negatives. For every positive there is:
        positives * ratio = negatives
        :param min_neg: the minimum of negative boxes to use for the confidence calculation
        """
        super().__init__(name=name, reduction=reduction)
        self.alpha = tf.constant(alpha)
        self.ratio = tf.constant(ratio)
        self.min_neg = tf.constant(min_neg)

    @staticmethod
    def smooth_L1(y_true, y_pred):
        """
        Implementation of the Smooth L1 Lost for bounding boxes, calculated as:
            x = y-pred - y_true
            -- 0.5 * x ** 2   if |x| < 1
            -- |x| - 0.5      otherwise
        :param y_true: Ground truth bounding boxes with shape (batch, #boxes, 4)
        :param y_pred: Predicted bounding boxes with shape (batch, #boxes, 4)
        :return: The Smooth L1 Loss. Shape: (batch, #boxes)

        Reference: https://arxiv.org/pdf/1504.08083.pdf
        """
        diff = y_pred - y_true
        abs_diff = tf.abs(diff)
        element_loss = tf.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
        return tf.reduce_sum(element_loss, axis=-1)

    @staticmethod
    def cross_entropy(y_true, y_pred):
        """
        Implementation of the cross entropy loss: matched * -log(Softmax(y_pred))
        :param y_true: Ground truth class labels: Shape (batch, #boxes, 2)
        :param y_pred: Predicted class after Softmax layer. Shape (batch, #boxes, 2)
        :return: The cross entropy loss. Shape: (batch, #boxes)
        """
        # The offset 1e-15 is necessary because for the unlikely case that y_pred is 0, the log will result in NaN
        y_pred = tf.maximum(tf.minimum(y_pred, 1.0 - 10 ** -15), 10 ** -15)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    def localisation_loss(self, y_true, y_pred):
        """
        Computes the localisation loss. This is calculated from the smooth L1 loss of the positives boxes.
        :param y_true: Ground truth labels. Shape (batch, #boxes, 6) with last axis:
        [class_bg, class_obj, offset_xmin, offset_ymin, offset_xmax, offset_ymax]
        :param y_pred: Predicted labels. Shape (batch, #boxes, 6) with last axis:
        [class_bg, class_obj, predicted_xmin, predicted_ymin, predicted_xmax, predicted_ymax]
        :return:
        """
        mask_positives = y_true[..., 1]
        # Calculate the smooth L1 loss between every ground truth box and every default box
        loc_loss_all = tf.cast(self.smooth_L1(y_true[..., 2:6], y_pred[..., 2:6]), dtype=tf.float32)
        # Pick only the positive predicted boxes with the positive mask and sum over all coordinates of a box
        loss_positives = tf.reduce_sum(loc_loss_all * mask_positives, axis=-1)
        return loss_positives

    def confidence_loss(self, y_true, y_pred):
        """
        Computes the confidence loss. The calculation results from the cross entropy loss of all positive boxes and a
        part of the negative ones. This part consists of the highest losses of the negative boxes, so the ratio of
        positives and negatives is equal to 'self.ratio'.

        The number of negatives involved in the loss can be calculated sample- or batch-wise
        :param y_true: Ground truth labels. Shape (batch, #boxes, 6) with last axis:
        [class_bg, class_obj, offset_xmin, offset_ymin, offset_xmax, offset_ymax]
        :param y_pred: Predicted labels. Shape (batch, #boxes, 6) with last axis:
        [class_bg, class_obj, predicted_xmin, predicted_ymin, predicted_xmax, predicted_ymax]
        :return:
        """
        mask_negatives = y_true[..., 0]
        mask_positives = y_true[..., 1]
        # Calculate the cross entropy loss for every default box
        conf_loss_all = tf.cast(self.cross_entropy(y_true[..., :2], y_pred[..., :2]), dtype=tf.float32)
        # Summing up the positive class losses per batch. Shape (batch,)
        conf_loss_positives = tf.reduce_sum(conf_loss_all * mask_positives, axis=-1)
        # The confidence loss for the negatives not summed up: (batch, #boxes)
        conf_loss_negatives = conf_loss_all * mask_negatives

        # Hard negative mining: Most boxes are negative, so there is an imbalance between the training data.
        # Instead of using all negative boxes for the calculation of the loss, only the ones with the highest
        # confidence loss are used. The ratio between positives and negatives is 1:'self.ratio'.

        # the number of objects per batch: shape (batch,)
        nr_positives = tf.cast(tf.reduce_sum(mask_positives, axis=-1), dtype=tf.int32)

        # the number of negatives boxes per batch: (batch,)
        nr_negatives = tf.cast(tf.reduce_sum(mask_negatives, axis=-1), dtype=tf.int32)

        # For every positive box, there should be 'self.ratio' negative boxes. If there are not enough
        # negatives, all negative boxes are used. If 'self.min_neg' is set, this is the minimum of negatives
        nr_negatives_keep = tf.minimum(tf.maximum(self.ratio * nr_positives, self.min_neg), nr_negatives)
        # for every sample i we need to find the top number_neg[i] losses
        conf_loss_negatives_filtered = tf.map_fn(fn=lambda i: tf.reduce_sum(tf.math.top_k(input=i[0],
                                                                                          k=i[1],
                                                                                          sorted=False).values),
                                                 elems=(conf_loss_negatives, nr_negatives_keep), dtype=tf.float32)

        # the total confidence loss is the sum of the positive and filtered negative losses
        return conf_loss_positives + conf_loss_negatives_filtered

    def ssd_loss(self, y_true, y_pred):
        # the number of objects per batch: shape (batch,)
        nr_positives = tf.cast(tf.reduce_sum(y_true[..., 1], axis=-1), dtype=tf.float32)
        localisation_loss = self.localisation_loss(y_true, y_pred)
        confidence_loss = self.confidence_loss(y_true, y_pred)

        # the total loss is the confidence loss and the localization loss: Shape (batch,)
        total_loss = self.alpha * localisation_loss + confidence_loss

        # the total loss needs to get divided by the number of positives, if there are no positives,
        # set the total lost to zero
        total_loss_normalized = tf.math.divide_no_nan(total_loss, nr_positives)
        return total_loss_normalized

    def call(self, y_true, y_pred):
        return tf.reduce_sum(self.ssd_loss(y_true, y_pred))


class SSDLossBatch(Loss):
    """
    SSD Loss over the whole batch.
    """
    def __init__(self, alpha=1.0, ratio=3, name='SSDLossBatch', reduction=Reduction.NONE, batch=None, **kwargs):
        super().__init__(name=name, reduction=reduction)
        self.ratio = tf.constant(ratio, dtype=tf.int32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    @staticmethod
    def smooth_L1(y_true, y_pred):
        """
        Implementation of the Smooth L1 Lost for bounding boxes, calculated as:
            x = y-pred - y_true
            -- 0.5 * x ** 2   if |x| < 1
            -- |x| - 0.5      otherwise
        :param y_true: Ground truth bounding boxes with shape (batch, #boxes, 4)
        :param y_pred: Predicted bounding boxes with shape (batch, #boxes, 4)
        :return: The Smooth L1 Loss. Shape: (batch, #boxes)

        Reference: https://arxiv.org/pdf/1504.08083.pdf
        """
        diff = y_pred - y_true
        abs_diff = tf.abs(diff)
        element_loss = tf.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
        return tf.reduce_sum(element_loss, axis=-1)

    @staticmethod
    def cross_entropy(y_true, y_pred):
        """
        Implementation of the cross entropy loss: matched * -log(Softmax(y_pred))
        :param y_true: Ground truth class labels: Shape (batch, #boxes, 2)
        :param y_pred: Predicted class after Softmax layer. Shape (batch, #boxes, 2)
        :return: The cross entropy loss. Shape: (batch, #boxes)
        """
        # The offset 1e-15 is necessary because for the unlikely case that y_pred is 0, the log will result in NaN
        y_pred = tf.maximum(tf.minimum(y_pred, 1.0 - 10 ** -15), 10 ** -15)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    def call(self, y_true, y_pred):
        mask_negatives = y_true[..., 0]
        mask_positives = y_true[..., 1]

        total_positives = tf.cast(tf.reduce_sum(mask_positives), dtype=tf.int32)
        total_negatives = tf.cast(tf.reduce_sum(mask_negatives), dtype=tf.int32)

        ###############################################################################################################
        conf_loss_all = tf.cast(self.cross_entropy(y_true[..., :2], y_pred[..., :2]), dtype=tf.float32)

        conf_loss_positives = tf.reduce_sum(conf_loss_all * mask_positives)
        conf_loss_negatives = conf_loss_all * mask_negatives
        conf_loss_negatives_1d = tf.reshape(conf_loss_negatives, shape=[-1])

        keep_negatives = tf.minimum(total_positives * self.ratio, total_negatives)

        conf_loss_negatives_filtered = tf.nn.top_k(input=conf_loss_negatives_1d, k=keep_negatives, sorted=False).values
        conf_loss_negatives_filtered_sum = tf.reduce_sum(conf_loss_negatives_filtered)

        conf_loss = conf_loss_negatives_filtered_sum + conf_loss_positives
        ###############################################################################################################
        localisation_loss_all = self.smooth_L1(y_true[..., 2:6], y_pred[..., 2:6])
        localisation_loss_positives = localisation_loss_all * mask_positives
        localisation_loss_positives_sum = tf.cast(tf.reduce_sum(localisation_loss_positives), dtype=tf.float32)
        ###############################################################################################################
        total_loss = conf_loss + self.alpha * localisation_loss_positives_sum
        total_loss_norm = tf.math.divide_no_nan(total_loss, tf.cast(total_positives, dtype=tf.float32))

        return total_loss_norm


if __name__ == '__main__':
    gt = [[[0, 1, 0, 0, 120, 100], [0, 1, 50, 50, 250, 250], [1, 0, 30, 50, 120, 145], [0, 1, 50, 50, 250, 250],
           [0, 1, 50, 50, 250, 250]],
          [[0, 1, 50, 50, 120, 100], [1, 0, 100, 100, 250, 250], [1, 0, 80, 85, 142, 250], [0, 1, 50, 50, 250, 250],
           [1, 0, 50, 50, 250, 250]],
          [[1, 0, 50, 50, 120, 100], [1, 0, 100, 100, 250, 250], [1, 0, 80, 85, 142, 250], [1, 0, 50, 50, 250, 250],
           [1, 0, 50, 50, 250, 250]]
          ]

    pred = [[[0.3, 0.7, 0, 0, 100, 80], [0.1, 0.8, 50, 50, 240, 250], [0.8, 0.2, 30, 50, 120, 145],
             [0.3, 0.7, 50, 50, 250, 250], [0.8, 0.2, 50, 50, 250, 250]],
            [[0.5, 0.5, 50, 50, 90, 70], [0.5, 0.5, 100, 100, 250, 250], [0.2, 0.8, 80, 85, 142, 250],
             [0, 1, 50, 50, 250, 250], [0, 1, 50, 50, 250, 250]],
            [[0.5, 0.5, 50, 50, 90, 70], [0.5, 0.5, 80, 180, 250, 250], [0.2, 0.8, 80, 85, 142, 250],
             [0, 1, 50, 50, 250, 250], [0, 1, 50, 50, 200, 200]]
            ]
    loss = SSDLoss(ratio=1, min_neg=0)
    loss_1 = SSDLossBatch()

    true = tf.cast(tf.constant(gt), dtype=tf.float32)
    pred = tf.cast(tf.constant(pred), dtype=tf.float32)
    print(loss(true, pred))
