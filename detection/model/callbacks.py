import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment
from detection.utils.box_tools import iou
from detection.default_boxes.default_boxes import DefaultBoxHandler
from detection.utils.validation import check_number, check_instance


class EvaluationMetric(tf.keras.callbacks.Callback):
    """
    Calculate Precision/ Recall during training after each epoch.
    """
    def __init__(self, generator, default_box_handler, confidence_threshold=0.5, nms_threshold=0.3, iou_threshold=0.5):
        super(EvaluationMetric, self).__init__()
        self.generator = generator
        self.default_box_handler = check_instance(default_box_handler, DefaultBoxHandler)

        self.confidence_threshold = check_number(confidence_threshold, value_min=0, value_max=1)
        self.nms_threshold = check_number(nms_threshold, value_min=0, value_max=1)
        self.iou_threshold = check_number(iou_threshold, value_min=0, value_max=1)

    def on_epoch_end(self, epoch, logs=None):
        prediction_result = []
        labels = []
        for batch_x, batch_y in self.generator:
            prediction = self.model.predict(batch_x)
            decoded = self.default_box_handler.faster_decode_default_boxes(prediction,
                                                                           confidence_threshold=self.confidence_threshold,
                                                                           nms_threshold=self.nms_threshold,
                                                                           clip=True)
            decoded_labels = self.default_box_handler.faster_decode_default_boxes(batch_y.astype("float32"),
                                                                                  no_score=True)
            prediction_result += decoded
            labels += decoded_labels

        assigned_predictions = self.predictions2labels(prediction_result, labels, iou_threshold=self.iou_threshold)

        mean_precision, mean_recall, f1 = self.flat_precision_recall(assigned_predictions)

        logs['precision'] = mean_precision
        logs['recall'] = mean_recall
        logs['f1'] = f1
        print("Precision: {0:.3f}, Recall: {1:.3f}, F1: {2:.3f}".format(mean_precision, mean_recall, f1))

    @staticmethod
    def predictions2labels(x, y, iou_threshold=0.5):
        """
        Assigns a ground truth box to each predicted box if they overlap at least 'iou_threshold'.
        :param x: list of np.array of shape [#predictions, 5] with (score, x1, y1, x2, y2) on last axis
        :param y: list of np.array if shape [#ground_truth, 4] with (x1, y1, x2, y2) on last axis
        :param iou_threshold:
        :return: List of len(#images) of np.arrays of shape (#predictions, 2) with last axis
        [confidence score, evaluation]. Evaluation is 0 for a false positive prediction and 1 for a true positive
        prediction or -1 else.
        """
        # make a new prediction only if there is no stored
        result = []
        for i, prediction in enumerate(x):
            # if there are no prediction skip because only true/false positive are interesting
            if prediction.size == 0:
                result.append(np.array([[-1, -1]]))
                continue
            # get labels for the current image from the dataset
            labels = y[i]
            # if there are no labels on the image set all predictions to false positive
            if labels.size == 0:
                result.append(np.vstack([prediction[..., 0], np.zeros((prediction.shape[0],))]).T)
                continue
            elif labels.shape[1] == 5:
                print("Warning: Labels has last axis of size 5. Probably there is an score inside. "
                      "First value is removed!")
                labels = labels[:, 1:]
            # calculate the IoU between every predicted and every ground truth box
            overlap = iou(prediction[..., 1:], labels, combinations=True)

            # set the overlap to zero, which is smaller then the threshold
            overlap_filtered = overlap * (overlap > iou_threshold)
            # match every predicted box to exact one label
            match_pred, match_gt = linear_sum_assignment(overlap, maximize=True)
            # set all boxes to fp
            tp_fp = np.zeros((prediction.shape[0],))
            # every box, which has been matched to a value higher than zero, is a true positive prediction. Otherwise
            # its a false positive, because there is no label
            satisfy_thresh = np.where(overlap_filtered[match_pred, match_gt] > 0.0, True, False)
            # create a mask over all predictions whether they are tp
            mask = match_pred[satisfy_thresh]
            # set the tp to one. The masking step is necessary because the there can be more predicted boxes than labels
            tp_fp[mask] = 1
            # extract the scores from the predictions
            scores = prediction[..., 0]
            # merge the score with the evaluation
            result.append(np.vstack([scores, tp_fp]).T)
        return result

    def flat_precision_recall(self, assigned_predictions):
        """
        Calculate the precision, recall and the f1 score of the given evaluation dataset:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)  (harmonic mean of precision and recall)
        with tp: true positive, fp: false positive, fn: false negative, tn: true negative
                                Object exists       Object fail
        Object predicted            tp                  fp
        No object predicted         fn                  tn
        :return:
        """
        merged_predictions = np.concatenate(assigned_predictions, axis=0)
        tp = np.sum(merged_predictions[..., 1] == 1)
        fp = merged_predictions.shape[0] - tp
        precision = tp / (tp + fp)
        recall = tp / self.generator.dataset.nr_labels  # number of ground truth is tp + fn
        f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
        return precision, recall, f1
