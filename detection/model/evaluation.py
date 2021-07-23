import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import tensorflow as tf

from detection.model.models.ssd import SSD
from detection.utils.box_tools import iou, draw_box
from detection import BASE_DATA_PATH, BASE_TRAINING_PATH


class Evaluator:
    def __init__(self, ssd_model, data_generator, rt_model=False, name='Evaluation'):
        self.box_handler = ssd_model.box_handler
        self.model = ssd_model.model
        self.generator = data_generator
        self.name = name
        self.rt_model = rt_model

        if self.rt_model:
            signature_keys = list(self.model.signatures.keys())
            print("Signatures:", signature_keys)
            self.infer = self.model.signatures['serving_default']

        self.current_predictions = None
        self.current_labels = self.get_labels()

    def predict(self):
        self.current_predictions = []
        for batch_x, batch_y in self.generator:
            if self.rt_model:
                prediction = self.infer(tf.constant(batch_x.astype("float32")))
                self.current_predictions.append(prediction['box_and_class'].numpy())
            else:
                prediction = self.model.predict(batch_x)
                self.current_predictions.append(prediction)

    def decode(self, confidence_threshold=0.5, nms_threshold=0.45, methode='normal', sigma=0.3):
        decoded_prediction = []
        for i, (_, batch_y) in enumerate(self.generator):
            decoded = self.box_handler.faster_decode_default_boxes(self.current_predictions[i],
                                                                   confidence_threshold=confidence_threshold,
                                                                   nms_threshold=nms_threshold, nms_methode=methode,
                                                                   clip=True, sigma=sigma)
            decoded_prediction += decoded
        return decoded_prediction

    def get_labels(self):
        labels = []
        for _, batch_y in self.generator:
            decoded_labels = self.box_handler.faster_decode_default_boxes(batch_y.astype("float32"),
                                                                          no_score=True)
            labels += decoded_labels
        return labels

    def predict_and_decode(self, confidence_threshold=0.5, nms_threshold=0.45, methode='normal', sigma=0.3):
        prediction_result = []
        for batch_x, _ in self.generator:
            prediction = self.model.predict(batch_x)
            decoded = self.box_handler.faster_decode_default_boxes(prediction,
                                                                   confidence_threshold=confidence_threshold,
                                                                   nms_threshold=nms_threshold, nms_methode=methode,
                                                                   clip=True, sigma=sigma)
            prediction_result += decoded
        return prediction_result

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

    @staticmethod
    def precision_at_recall(assigned_prediction, nr_labels_total):
        """
        :return: two array with the total length according to the predictions over all data. The first array contains
         the precisions, the second the recalls
        """
        # merge the matched predictions of all images to one array
        merged_predictions = np.concatenate(assigned_prediction, axis=0)
        # sort predictions by highest confidence first
        ordered_prediction = merged_predictions[merged_predictions[..., 0].argsort()][::-1]
        # remove scores from predictions
        c = ordered_prediction[..., 1]
        cum_tp = np.cumsum(c)
        cum_fp = np.cumsum(c == 0)
        precision = np.where(cum_tp + cum_fp > 0, cum_tp / (cum_tp + cum_fp), 0)
        recall = cum_tp / nr_labels_total
        return precision, recall

    @staticmethod
    def calc_average_precision(precisions, recalls, recall_points=None):
        average_precision = 0.0
        if recall_points is None:
            return np.trapz(precisions, recalls)
        for i in np.linspace(0, 1, num=recall_points, endpoint=True):
            x = precisions[recalls >= i]
            if x.size == 0:
                calc_pred = 0.0
            else:
                calc_pred = np.amax(x)
            average_precision += calc_pred
        return average_precision / recall_points

    def evaluate(self, iou_threshold=0.5, save=False, show=True, path='', nms_threshold=0.3, methode='normal',
                 sigma=0.3, recall_points=None):
        average_precision_stor = {}
        if self.current_predictions is None:
            self.predict()

        if isinstance(iou_threshold, float):
            iou_threshold = [iou_threshold]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for thresh in iou_threshold:
            prediction = self.decode(confidence_threshold=0.01, nms_threshold=nms_threshold, methode=methode,
                                     sigma=sigma)
            assigned_prediction = self.predictions2labels(prediction, self.current_labels, iou_threshold=thresh)
            precision, recall = self.precision_at_recall(assigned_prediction, nr_labels_total=self.generator.dataset.nr_labels)
            average_precision = self.calc_average_precision(precision, recall, recall_points=recall_points)
            ax.plot(recall, precision, label="th={0} aP={1:.4f}".format(thresh, average_precision))
            average_precision_stor[thresh] = average_precision

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        # ax.set_xlim([0, 1.0])
        # ax.set_ylim([0, 1.0])
        plt.tight_layout()
        if save:
            fig.savefig(os.path.join(path, 'evaluation'))
        if show:
            plt.show()
        return average_precision_stor

    def evaluate_flat(self, confidence_threshold=0.5, nms_threshold=0.45, methode='normal', sigma=0.3,
                      iou_threshold=0.5, verbose=True):
        if self.current_predictions is None:
            self.predict()
        prediction = self.decode(confidence_threshold=confidence_threshold, nms_threshold=nms_threshold,
                                 methode=methode, sigma=sigma)
        assigned_prediction = self.predictions2labels(prediction, self.current_labels, iou_threshold=iou_threshold)
        precision, recall, f1 = self.flat_precision_recall(assigned_prediction)

        if verbose:
            print("Precision: {0:.3f}, Recall: {1:.3f}, F1: {2:.3f}".format(precision, recall, f1))

        return precision, recall, f1

    def compare_nms_methods(self, confidence_threshold=0.5, iou_thresh=0.5, nms_thresh=[0.25, 0.35, 0.45],
                            sigma=[0.3, 0.4, 0.5], show=True):
        max_f1 = 0
        max_conf = None
        result = {'normal': [], 'linear': [], 'gaussian': []}
        for m in ['normal', 'linear']:
            for t in nms_thresh:
                precision, recall, f1 = self.evaluate_flat(confidence_threshold=confidence_threshold,
                                                           iou_threshold=iou_thresh, nms_threshold=t, methode=m,
                                                           verbose=False)
                print("{0:<8} {1:.2f}: {2:.4f}".format(m, t, f1))
                if f1 > max_f1:
                    max_f1 = f1
                    max_conf = (m, t, f1)
                result[m].append([t, f1])
        for s in sigma:
            precision, recall, f1 = self.evaluate_flat(confidence_threshold=confidence_threshold,
                                                       iou_threshold=iou_thresh, sigma=s, methode='gaussian',
                                                       verbose=False)
            print("{0:<8} {1:.2f}: {2:.4f}".format("gaussian", s, f1))
            if f1 > max_f1:
                max_f1 = f1
                max_conf = ('gaussian', s, f1)
            result['gaussian'].append([s, f1])
        print("Best result: \n {0:<8} {1:.2f}: {2:.4f}".format(*max_conf))
        if show:
            for key, value in result.items():
                if len(value) != 0:
                    x, y = list(zip(*value))
                    plt.plot(x, y, label=key)
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('mAP')
        plt.tight_layout()
        plt.gcf().savefig("compare_nms.png", dpi=500)
        plt.show()

        return result


if __name__ == '__main__':
    from detection.model.registry import load_model
    from detection.dataset.dataset_generator import DataGenerator, Dataset

    img_dir = ['/home/t9s9/PycharmProjects/BeeMeter/data/training/base_training_img',
               '/home/t9s9/PycharmProjects/BeeMeter/data/training/gopro_training_img']
    label_dir = ['/home/t9s9/PycharmProjects/BeeMeter/data/training/base_labels.db',
                 '/home/t9s9/PycharmProjects/BeeMeter/data/training/gopro_labels.db']

    weights = "/media/t/Bachelor/bonus_experiments/MobileNetV2_B_10_expand_noTop/checkpoints/MobileNetV2_B_10_expand_noTopweights.h5"
    config = "/media/t/Bachelor/bonus_experiments/MobileNetV2_B_10_expand_noTop/model_config.conf"
    model = load_model(config_path=config, weights_path=weights, new=False)

    datasets = []
    for img_p, lab_p in zip(img_dir, label_dir):
        datasets.append(Dataset.from_sqlite(img_p, lab_p, verbose=True, limit=None))

    training_dataset = sum(datasets)
    validation_dataset = training_dataset.validation_split(ratio=0.15, seed=0)

    validation_dataset.load_images_in_memory(verbose=True)

    validation_generator = DataGenerator(dataset=validation_dataset,
                                         encoder=model.box_handler,
                                         batch_size=2,
                                         shuffle=False,
                                         resize=(model.width, model.height))


    eval = Evaluator(ssd_model=model, data_generator=validation_generator, rt_model=False)

    print(eval.evaluate_flat(confidence_threshold=0.5, methode='linear', nms_threshold=0.3))
    #
    # # eval.compare_nms_methods(nms_thresh=list(np.arange(0.1, 0.6, 0.01, dtype=np.float16)),
    # #                          sigma=list(np.arange(0.1, 0.6, 0.01, dtype=np.float16)))
    print(eval.evaluate(iou_threshold=[0.5], save=False, nms_threshold=0.3, methode='normal'))
