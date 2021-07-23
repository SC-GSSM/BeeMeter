import time
import numpy as np
from pathlib import Path
import cv2

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from detection.default_boxes.default_boxes import DefaultBoxHandler
from detection.dataset.dataset_generator import Dataset
from detection.model.evaluation import Evaluator

tf.keras.backend.clear_session()
tf.keras.backend.set_learning_phase(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class TenorRTSSD:
    def __init__(self, model=None, box_handler=None):
        tf.keras.backend.clear_session()
        self.model = model
        self.box_handler = box_handler
        self.width, self.height = 400, 200

    @classmethod
    def from_config(cls, model_path, config_path):
        saved_model = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
        print("Model loaded.")
        model = saved_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        print("Model signature.")

        box_handler = DefaultBoxHandler.from_config(config_path)
        print("Boxes created.")
        return cls(model=model, box_handler=box_handler)

    def predict(self, x, conf_threshold=0.5, nms_threshold=0.3, sigma=0.5, nms_methode='normal'):
        x = tf.constant(x, dtype=tf.float32)
        x = tf.expand_dims(x, axis=0)
        prediction = self.model(x)
        prediction = prediction['box_and_class'].numpy()
        prediction = self.box_handler.faster_decode_default_boxes(prediction, confidence_threshold=conf_threshold,
                                                                  nms_threshold=nms_threshold, sigma=sigma,
                                                                  nms_methode=nms_methode)
        return prediction

    def benchmark(self, warmup=200, runs=800):
        """
        Runs dummy data through the network to get a benchmark of the prediction.
        :param warmup:
        :param runs:
        :return:
        """
        elapsed_time = []
        img = tf.constant(np.random.normal(size=(1, 200, 400, 3)).astype(np.float32))
        print("Data created.")

        for i in range(warmup):
            output = self.model(img)
        print("Finished warmup.")

        for i in range(1, runs + 1):
            img = tf.constant(np.random.normal(size=(1, 200, 400, 3)).astype(np.float32))
            start_time = time.time()
            output = self.model(img)
            end_time = time.time()
            elapsed_time = np.append(elapsed_time, end_time - start_time)
            if i % 50 == 0:
                print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))
        print('Throughput: {0:.2f} ms/image   {1:.0f} FPS'.format(elapsed_time.mean() * 1000,
                                                                  runs * 1 / elapsed_time.sum()))

    def evaluate(self, image_path, labels_path, iou_threshold=0.5):
        result = []

        ds = Dataset.from_sqlite(image_path, labels_path)
        for path in ds.filenames:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(img_rgb, (self.width, self.height))
            prediction = self.predict(img_resize, conf_threshold=0.01)[0]

            prediction[..., [1, 3]] = np.round(prediction[..., [1, 3]] * (img.shape[1] / self.width), decimals=0)
            prediction[..., [2, 4]] = np.round(prediction[..., [2, 4]] * (img.shape[0] / self.height), decimals=0)
            result.append(prediction)

        assigned_prediction = Evaluator.predictions2labels(result, ds.labels, iou_threshold=iou_threshold)
        precision, recall = Evaluator.precision_at_recall(assigned_prediction, ds.nr_labels)
        mAP = Evaluator.calc_average_precision(precision, recall)
        print("mAP: {0:.3f}".format(mAP))
        return mAP


if __name__ == '__main__':
    m = TenorRTSSD.from_config(model_path='../resources/demo/tensorRT_FP16',
                               config_path="../resources/demo/model_config.conf")
    m.benchmark()
