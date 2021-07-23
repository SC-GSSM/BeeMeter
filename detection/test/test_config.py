import unittest
from model.ssd_config import SSDConfig


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = SSDConfig(path='default_config.json')
        self.assertEqual(c.batch, 8)
        self.assertEqual(c.epochs, 10)
        self.assertEqual(c.input_size, (500, 400))
        self.assertEqual(c.name, 'Test Training')
        self.assertEqual(c.training_path, "/home/t9s9/PiCamPictures/training_img")
        self.assertEqual(c.validation_path, "/home/t9s9/PiCamPictures/validation_img")
        self.assertEqual(c.label_path, "/home/t9s9/PycharmProjects/BeeMeter/data/PiCamLabels.db")
        self.assertEqual(c.boxes_per_cell_global, 3)
        self.assertEqual(c.boxes_per_cell_local, [2, 3, 2])
        self.assertEqual(c.get_default_box_config(),
                         {'image_size': (500, 400), 'fixed_scale': None, 'scale_min': 0.3, 'scale_max': 0.9,
                          'aspect_ratios_global': [0.5, 1.0, 1.5], 'aspect_ratios': [[0.5, 1], [1, 1.5, 2], [1, 2]],
                          'use_bonus_square_box': False, 'standardizing': [1.2, 1.2, 1.2, 1.2]})
