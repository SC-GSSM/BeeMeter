import unittest
from model.loss_function import SSDLoss, SSDLossBatch
import numpy as np
import tensorflow as tf
from math import log


class TestLoss(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_boxes = [[[0.1, 0.1, 0.33, 0.44], [0.2, 0.9, 0.21, 0.99]],
                           [[0.5, 0.5, 0.5, 0.5], [1.1, 0.3, 0.2, 0.5]]]
        self.pred_boxes = [[[0.15, 0.1, 0.3, 0.4], [0.2, 0.2, 0.2, 0.2]], [[0.6, 0.7, 0.8, 0.9], [0.0, 0.3, 0.2, 0.5]]]
        self.true_class = [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]],
                           [[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
        self.pred_class = [[[0.8, 0.2], [0.2, 0.8]], [[0.5, 0.5], [0.3, 0.7]],
                           [[1.0, 0.0], [0.2, 0.7]], [[0.9, 0.1], [1.0, 0.0]]]

        box_true_reshape = np.reshape(np.tile(self.true_boxes, reps=(2, 1)), (4, 2, 4))
        self.true = np.concatenate([self.true_class, box_true_reshape], axis=-1)
        box_pred_reshape = np.reshape(np.tile(self.pred_boxes, reps=(2, 1)), (4, 2, 4))
        self.pred = np.concatenate([self.pred_class, box_pred_reshape], axis=-1)

    def test_smooth_L1_loss(self):
        true = tf.constant(self.true_boxes)
        pred = tf.constant(self.pred_boxes)
        loss = SSDLoss.smooth_L1(true, pred)
        np.testing.assert_almost_equal(np.array([[0.0025, 0.5571], [0.15, 0.6]]), loss, decimal=6)
        np.testing.assert_almost_equal(np.zeros((2, 2)), SSDLoss.smooth_L1(true, true), decimal=6)

    def test_cross_entropy(self):
        loss = SSDLoss.cross_entropy(tf.constant(self.true_class), tf.constant(self.pred_class))

        res = []
        for batch_index in range(len(self.true_class)):
            batch_res = []
            for box_index in range(len(self.true_class[batch_index])):
                x = self.true_class[batch_index][box_index][0] * -log(
                    self.pred_class[batch_index][box_index][0] + 1e-15)
                y = self.true_class[batch_index][box_index][1] * -log(
                    self.pred_class[batch_index][box_index][1] + 1e-15)
                batch_res.append(x + y)
            res.append(batch_res)

        np.testing.assert_almost_equal(np.array(res), loss, decimal=6)

    def test_localisation_loss(self):
        loss = SSDLoss().localisation_loss(self.true, self.pred)
        x = 0.5 * (0.05 ** 2 + 0.03 ** 2 + 0.04 ** 2 + 0.7 ** 2 + 0.01 ** 2 + 0.79 ** 2)
        y = (1.1 - 0.5)
        res = np.array([0.0, x, 0.0, y])
        np.testing.assert_almost_equal(np.array(res), loss, decimal=6)

    def test_confidence_loss(self):
        loss = SSDLoss(alpha=1.0, ratio=3, min_neg=2).confidence_loss(self.true, self.pred)
        conf_loss_pos = np.array([0.0, -log(0.5) - log(0.7), 0.0, -log(1e-15)])
        conf_loss_neg = np.array([-log(0.8) - log(0.2), 0.0, 0.0, -log(0.9)])  # keep: 2 0 0 1
        np.testing.assert_almost_equal(conf_loss_neg + conf_loss_pos, loss, decimal=6)

    def test_ssd_loss(self):
        loss = SSDLoss(alpha=1.0, ratio=3, min_neg=2).ssd_loss(self.true, self.pred)
        np.testing.assert_almost_equal([0.0, 1.60942212 / 2.0, 0.0, 35.244 / 1.0], loss, decimal=3)
        loss = SSDLoss(alpha=1.0, ratio=3, min_neg=2).ssd_loss(self.true, self.pred)
        batch = self.true.shape[0]
        np.testing.assert_almost_equal([0.0, 1.60942212 / 2.0 * batch, 0.0, 35.244 / 1.0 * batch], loss, decimal=3)

    def test_compare_sample_batch(self):
        loss_batch = SSDLossBatch()
        loss_sample = SSDLoss()

        print(loss_batch(self.true, self.pred))
        print(loss_sample(self.true, self.pred))


    if __name__ == '__main__':
        unittest.main()
