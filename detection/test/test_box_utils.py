from detection.utils.box_tools import convert_boxes, relative2absolute, absolute2relative, iou, intersection, clip_boxes, \
    validate_boxes
import unittest
import numpy as np


class TestBoxUtils(unittest.TestCase):
    def test_conversion(self):
        boxes_corner = np.array([[50, 100, 100, 150], [83, 157, 850, 648], [3, 999, 7, 1041]])
        boxes_minmax = np.array([[80, 150, 0, 200], [83, 850, 157, 648], [0, 100, 0, 200]])
        boxes_center = np.array([[133, 154, 200, 200], [154, 789, 145, 124], [120, 147, 1527, 1452]])

        np.testing.assert_allclose(
            convert_boxes(convert_boxes(convert_boxes(boxes_corner, 'corner2minmax'), 'minmax2center'),
                          'center2corner'), boxes_corner, rtol=1.0)
        np.testing.assert_allclose(
            convert_boxes(convert_boxes(convert_boxes(boxes_corner, 'corner2center'), 'center2minmax'),
                          'minmax2corner'), boxes_corner, rtol=1.0)
        np.testing.assert_allclose(
            convert_boxes(convert_boxes(convert_boxes(boxes_minmax, 'minmax2center'), 'center2corner'),
                          'corner2minmax'), boxes_minmax, rtol=1.0)
        np.testing.assert_allclose(
            convert_boxes(convert_boxes(convert_boxes(boxes_minmax, 'minmax2corner'), 'corner2center'),
                          'center2minmax'), boxes_minmax, rtol=1.0)
        np.testing.assert_allclose(
            convert_boxes(convert_boxes(convert_boxes(boxes_center, 'center2corner'), 'corner2minmax'),
                          'minmax2center'), boxes_center, rtol=1.0)
        np.testing.assert_allclose(
            convert_boxes(convert_boxes(convert_boxes(boxes_center, 'center2minmax'), 'minmax2corner'),
                          'corner2center'), boxes_center, rtol=1.0)

    def test_relative_absolute(self):
        size = 1000, 800
        boxes = np.array([[0, 0, 100, 100], [233, 245, 348, 457], [475, 369, 487, 578]])
        boxes3d = np.tile(boxes, reps=(4, 1)).reshape((4, 3, 4))
        np.testing.assert_allclose(relative2absolute(absolute2relative(boxes, size, 'corner'), size, 'corner'), boxes)
        np.testing.assert_allclose(relative2absolute(absolute2relative(boxes3d, size, 'corner'), size, 'corner'),
                                   boxes3d)

    def test_iou(self):
        size = 1000, 1200
        boxesA = np.array([[0, 0, 100, 100], [100, 100, 200, 200]])
        boxesB = np.array([[500, 500, 800, 800], [150, 150, 700, 700]])
        boxesC = np.array([[[100, 100, 200, 200]]])
        result = np.array([0.0, 1.0 / 125.0])

        print(iou(boxesC, boxesB))
        np.testing.assert_array_almost_equal(iou(boxesA, boxesB), result, decimal=3)
        np.testing.assert_equal(iou(boxesA, boxesB), iou(boxesB, boxesA))
        np.testing.assert_equal(iou(boxesA, boxesA), np.tile([1.0], boxesA.shape[0]))
        np.testing.assert_array_almost_equal(
            iou(absolute2relative(boxesA, size, 'corner'), absolute2relative(boxesB, size, 'corner')), result,
            decimal=3)

    def test_intersection_combination(self):
        def intersection_comb(a, b):
            res = []
            for box in b:
                res.append(intersection(a, np.array([box])))
            return np.array(res).T

        for i in range(10):
            size_x = np.random.randint(100, 1000)
            size_y = np.random.randint(100, 1000)
            x = np.concatenate(
                (np.random.randint(0, 200, size=(size_x, 2)), np.random.randint(201, 500, size=(size_x, 2))),
                axis=1)
            y = np.concatenate(
                (np.random.randint(0, 200, size=(size_y, 2)), np.random.randint(201, 500, size=(size_y, 2))),
                axis=1)

            np.testing.assert_equal(intersection(x, y, combinations=True), intersection_comb(x, y))

    def test_clipping(self):
        image_size = (500, 500)
        boxes = np.array([[-100, 100, 325, 145], [0, 0, 10, 10], [-123, -478, 100000000, 142]])
        boxes_norm = absolute2relative(boxes, image_size, box_format='corner')

        boxes_clipped = clip_boxes(boxes, image_size, normalized=False)
        boxes_norm_clipped = clip_boxes(boxes_norm, image_size, normalized=True)

        self.assertTrue(np.all(validate_boxes(boxes_clipped, image_size=image_size, box_format='corner')))
        self.assertTrue(np.all(validate_boxes(boxes_norm_clipped, image_size=image_size, box_format='corner')))


if __name__ == '__main__':
    unittest.main()
