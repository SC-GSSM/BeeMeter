import cv2
import numpy as np
from detection.utils.box_tools import validate_boxes, clip_boxes


class Transformer:
    def __init__(self, operations):
        """
        Wrapper that applies every operation in 'operations' on all given images and labels.
        :param operations: Array of operation classes or instance of class 'TransformationChain'
        """
        if not (isinstance(operations, list) or isinstance(operations, TransformationChain)):
            raise ValueError('The operations passed to the Transformer need to be either a list or an instance of a '
                             'TransformationChain')
        self.operations = operations

    def __call__(self, images, labels=None):
        processed_images = []
        if labels is not None:
            processed_labels = []
            for image, label in zip(images, labels):
                for transformation in self.operations:
                    image, label = transformation(image, label)
                processed_images.append(image)
                processed_labels.append(label)
            return np.array(processed_images), np.array(processed_labels)
        else:
            for image in images:
                for transformation in self.operations:
                    image = transformation(image, None)
                processed_images.append(image)
            return np.array(processed_images)


class TransformationChain:
    def __init__(self, number):
        self.number = number
        self.chain = [
            [DataTypeConverter(target='float'),
             Brightness(min_value=-50, max_value=50, apply=0.5),
             Contrast(min_value=0.5, max_value=1.5, apply=0.5),
             DataTypeConverter(target='int'),
             VerticalFlip(apply=0.5),
             HorizontalFlip(apply=0.5)],

            [DataTypeConverter(target='float'),
             Brightness(min_value=-40, max_value=40, apply=0.5),
             Contrast(min_value=0.6, max_value=1.4, apply=0.5),
             DataTypeConverter(target='int'),
             ColorSpaceConverter(actual='RGB', target='HSV'),
             DataTypeConverter(target='float'),
             Hue(min_value=-20, max_value=20, apply=0.5),
             Saturation(min_value=-20, max_value=20, apply=0.5),
             DataTypeConverter(target='int'),
             ColorSpaceConverter(actual='HSV', target='RGB'),
             HorizontalFlip(apply=0.5),
             VerticalFlip(apply=0.5)
             ]
        ]

        if self.number > len(self.chain) - 1:
            raise IndexError('The chain with the number {0} does not exist. '
                             'Please choose one of: {1}'.format(self.number, str(list(range(len(self.chain))))))

    def __len__(self):
        return len(self.chain[self.number])

    def __iter__(self):
        for item in range(len(self)):
            yield self.chain[self.number][item]


class Resize:
    def __init__(self, output_width, output_height, interpolation=cv2.INTER_LINEAR):
        """
        Class to resizing images and labels.
        :param output_width:
        :param output_height:
        :param interpolation: the algorithm that calculates the resizing
        """
        algorithms = {'linear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC, 'nearest': cv2.INTER_NEAREST}
        self.output_width = output_width
        self.output_height = output_height
        if isinstance(interpolation, str):
            try:
                self.interpolation = algorithms[interpolation]
            except KeyError:
                raise KeyError(
                    'Unknown algorithm {0}. Please choose one of: {1}'.format(interpolation, algorithms.keys()))
        else:
            self.interpolation = interpolation

    def __call__(self, image, labels=None):
        height, width, channel = image.shape
        ratio_w = self.output_width / width
        ratio_h = self.output_height / height
        image = cv2.resize(image, dsize=(self.output_width, self.output_height),
                           interpolation=self.interpolation)
        if labels is not None:
            resized_labels = np.copy(labels)
            resized_labels[:, [0, 2]] = np.round(resized_labels[:, [0, 2]] * ratio_w, decimals=0)
            resized_labels[:, [1, 3]] = np.round(resized_labels[:, [1, 3]] * ratio_h, decimals=0)
            return image, resized_labels
        return image


class Transformation:
    def __init__(self, apply=1.0):
        """
        Basic class to implement a transformation.
        :param apply: the probability that the transformation is applied
        """
        self.skip = apply

    def apply_transformation(self, image, labels):
        return image, labels

    def __call__(self, image, labels):
        if np.random.uniform(0, 1) < self.skip:
            return self.apply_transformation(image, labels)
        else:
            return image, labels


class Brightness(Transformation):
    def __init__(self, min_value=-50, max_value=50, apply=1.0):
        super().__init__(apply=apply)
        self.min = min_value
        self.max = max_value

    def apply_transformation(self, image, labels):
        if image.dtype == np.float32:
            x = np.random.randint(self.min, self.max)
            img = np.clip(image + x, a_min=0, a_max=255)
            return img, labels
        else:
            raise ValueError("Cannot apply brightness conversion to INT array. Please convert to float.")


class Contrast(Transformation):
    def __init__(self, min_value=0.5, max_value=1.5, apply=1.0):
        super().__init__(apply=apply)
        self.min = min_value
        self.max = max_value

    def apply_transformation(self, image, labels):
        x = np.random.uniform(self.min, self.max)
        image = np.clip(127.5 + x * (image - 127.5), 0, 255)
        return image, labels


class Hue(Transformation):
    def __init__(self, min_value=-20, max_value=20, apply=1.0):
        super().__init__(apply=apply)
        self.min = min_value
        self.max = max_value

    def apply_transformation(self, image, labels):
        # apply random hue
        random_hue = np.random.randint(self.min, self.max)
        image[:, :, 0] = (image[:, :, 0] + random_hue) % 180
        return image, labels


class Saturation(Transformation):
    def __init__(self, min_value=-20, max_value=20, apply=1.0):
        super().__init__(apply=apply)
        self.min = min_value
        self.max = max_value

    def apply_transformation(self, image, labels):
        # apply random saturation
        random_saturation = np.random.randint(self.min, self.max)
        image[:, :, 0] = (image[:, :, 1] + random_saturation) % 255
        return image, labels


class Value(Transformation):
    def __init__(self, min_value=-20, max_value=20, apply=1.0):
        super().__init__(apply=apply)
        self.min = min_value
        self.max = max_value

    def apply_transformation(self, image, labels):
        # apply random value
        random_value = np.random.randint(self.min, self.max)
        image[:, :, 0] = (image[:, :, 2] + random_value) % 255
        return image, labels


class HorizontalFlip(Transformation):
    def __init__(self, apply=1.0):
        super().__init__(apply=apply)

    def apply_transformation(self, image, labels):
        new_labels = np.copy(labels)
        new_labels[:, [0, 2]] = image.shape[1] - new_labels[:, [2, 0]]
        return image[:, ::-1], new_labels


class VerticalFlip(Transformation):
    def __init__(self, apply=1.0):
        super().__init__(apply=apply)

    def apply_transformation(self, image, labels):
        new_labels = np.copy(labels)
        new_labels[:, [1, 3]] = image.shape[0] - new_labels[:, [3, 1]]
        return image[::-1], new_labels


class DataTypeConverter:
    def __init__(self, target='float'):
        self.target = target

    def __call__(self, image, labels):
        if self.target.lower() == 'float':
            return image.astype(np.float32), labels
        else:
            return np.round(image, decimals=0).astype(np.uint8), labels


class ColorSpaceConverter:
    def __init__(self, actual, target):
        transformations = {('BGR', 'RGB'): cv2.COLOR_BGR2RGB, ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
                           ('RGB', 'HSV'): cv2.COLOR_RGB2HSV, ('HSV', 'RGB'): cv2.COLOR_HSV2RGB}
        try:
            self.transformation = transformations[(actual.upper(), target.upper())]
        except KeyError:
            raise KeyError('The transformation from {0} to {1} does not exist. Please try one of those: {2}'.format(
                actual, target, transformations.keys()
            ))

    def __call__(self, image, labels=None):
        conv_img = cv2.cvtColor(image, code=self.transformation)
        return (conv_img, labels) if labels is not None else conv_img


class Zoom(Transformation):
    """
    Zooms into the image and applies a random translation in x and y direction. The zoomed image is only shifted so far
    that it remains within the borders of the original image. The output size is always equal to the input size.
    """

    def __init__(self, scale=1.5, min_scale=None, max_scale=None, min_area=500, min_objects=1, apply=1.0):
        """
        :param scale: a fixed float value that specifies the scaling. Range 1.0 <= scale <= 3.0
        :param min_scale: If the value is set the scaling is random and the fixed value is overwritten. This value
        is the lower bound.
        :param max_scale: If the value is set the scaling is random and the fixed value is overwritten. This value
        is the upper bound.
        :param min_area: filter all boxes that have a smaller area than this value
        :param min_objects: the creates sample needs to have at least this values of objects to be valid otherwise
        this transformation is withdrawn
        :param apply: probability that this transformation is applied
        """
        super().__init__(apply=apply)
        if not isinstance(scale, float) and 1.0 <= scale <= 3.0:
            raise ValueError('The scale factor needs to be a float between 1.0 and 3.0.')
        self.scale = scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_area = min_area
        self.min_objects = min_objects

    def apply_transformation(self, image, labels):
        new_image = np.copy(image)
        if self.min_scale is not None and self.max_scale is not None:
            self.scale = np.random.uniform(self.min_scale, self.max_scale)

        height, width, _ = new_image.shape
        new_height, new_width = height * self.scale, width * self.scale

        trans_x = np.random.randint(width - new_width, 0)
        trans_y = np.random.randint(height - new_height, 0)

        transformation_matrix = np.array([[self.scale, 0, trans_x],
                                          [0, self.scale, trans_y]])

        new_image = cv2.warpAffine(new_image, transformation_matrix, (width, height))

        new_labels = np.copy(labels)
        mins = np.hstack([new_labels[:, :2], np.ones((new_labels.shape[0], 1))]).T
        maxs = np.hstack([new_labels[:, 2:], np.ones((new_labels.shape[0], 1))]).T
        new_mins = (np.dot(transformation_matrix, mins)).T
        new_maxs = (np.dot(transformation_matrix, maxs)).T
        new_labels[:, :2] = np.round(new_mins, decimals=0).astype(np.int)
        new_labels[:, 2:] = np.round(new_maxs, decimals=0).astype(np.int)

        new_labels = clip_boxes(new_labels, image_size=(width, height), normalized=False)
        valid_labels_mask = validate_boxes(new_labels, min_area=500, image_size=(width, height), normalized=False)
        new_labels = new_labels[valid_labels_mask]

        # if there are no more objects after transforming, revert this transformation
        if new_labels.shape[0] < self.min_objects:
            return image, labels

        return new_image, new_labels
