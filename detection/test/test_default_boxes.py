from default_boxes.default_boxes import DefaultBoxHandler
from model.ssd_config import SSDConfig
import unittest
import numpy as np


class TestDefaultBoxes(unittest.TestCase):
    def test_from_config(self):
        config = SSDConfig(path='default_config.json')
        test_feature_map_sizes = [(30, 30), (20, 20), (7, 7)]
        handler = DefaultBoxHandler(feature_map_sizes=test_feature_map_sizes, **config.get_default_box_config())

        handler.to_config(path='default_boxes')

        new_handler = DefaultBoxHandler.from_config(path='default_boxes.config')

        np.testing.assert_equal(handler.default_boxes, new_handler.default_boxes)

    def test_encode_to_decode(self):
        config = SSDConfig(path='default_config.json')
        test_feature_map_sizes = [(30, 30), (20, 20), (7, 7)]
        handler = DefaultBoxHandler(feature_map_sizes=test_feature_map_sizes, **config.get_default_box_config())

        gt = [np.array([[100, 100, 200, 300], [100, 150, 120, 180], [220, 300, 250, 380]]),
              np.array([[100, 100, 200, 300], [20, 80, 35, 95], [180, 190, 220, 254]])]
        encoded = handler.encode_default_boxes(gt, cut_default_boxes=True)
        # print(encoded[encoded[..., 1] == 1])
        decoded = handler.decode_default_boxes(encoded, soft=False)
        # print(decoded[..., 1:])

    def test_fit(self):
        gt = [np.array([[150, 150, 300, 300], [650, 650, 800, 800]]),
              np.array([[150, 150, 300, 300], [600, 600, 800, 800]])]
        handler = DefaultBoxHandler(image_size=(1000, 1000), feature_map_sizes=[(2, 2)], aspect_ratios_global=[1])
        print(handler.default_boxes)
        print(handler.encode_default_boxes(gt))
        handler.fit(gt)

    def test_std(self):
        from dataset.dataset_generator import Dataset
        base_dir = "/home/t9s9/PiCamPictures/training/"
        dataset = Dataset.from_sqlite(images_dirs=base_dir + "base_training_img", path=base_dir + "base_labels.db",
                                      verbose=False, limit=2)
        handler = DefaultBoxHandler(image_size=(400, 200), feature_map_sizes=[(2, 2)], aspect_ratios_global=[1])
        l = dataset.labels[0]
        l[..., [0, 2]] = np.round(l[..., [0, 2]] * (400 / dataset.img_width), decimals=0)
        l[..., [1, 3]] = np.round(l[..., [1, 3]] * (200 / dataset.img_height), decimals=0)
        encoded = handler.encode_default_boxes([l])
