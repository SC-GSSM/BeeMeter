import pickle
from os.path import splitext

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from detection.utils.box_tools import draw_box, iou, absolute2relative, relative2absolute, convert_boxes, clip_boxes
from detection.utils.validation import InconsistentNumberDefaultBoxError, check_array
from detection.default_boxes.non_maximum_suppression import nms, nms_np


class DefaultBoxHandler:
    def __init__(self, image_size, feature_map_sizes, fixed_scale=None, scale_min=0.2, scale_max=0.9,
                 aspect_ratios_global=None, aspect_ratios=None, use_bonus_square_box=False, standardizing=None,
                 **kwargs):
        """
        This class takes over the creation, encoding and decoding of the default boxes according to the given
        configurations. For the purpose of training the default boxes must be matched with the ground truth boxes and
        then converted to an offset (encoding). For the inference the predicted boxes have to be transformed into
        absolute box coordinates (decoding).
        :param image_size: Size of the models input image size. Format: (width, height)
        :param feature_map_sizes: A list of tuples for each feature map (height, width)
        :param fixed_scale: If no min and max is given, one can use fixed scalings.
        :param scale_min: Minimum scale for the boxes. The scaling is selected at the equal distance between
        minimum and maximum.
        :param scale_max: Maximum scale for the boxes.
        :param aspect_ratios_global: The Rations of the default boxes. Global means that the ratios are the same for
        every feature layer.
        :param aspect_ratios: The ratios can be specified individually for each layer if no global value is given.
        :param use_bonus_square_box: In the original paper the authors recommend an extra bounding box with larger size
         and ratio 1. If False it is not used.
        :param standardizing: None or a list of four values to standardize the box offset. See fit function to
        calculate these.

        Reference: https://arxiv.org/pdf/1512.02325.pdf
        """
        self.input_width, self.input_height = image_size
        self.use_bonus_square_box = use_bonus_square_box
        self.feature_map_sizes = feature_map_sizes
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.fixed_scale = fixed_scale

        if aspect_ratios_global is not None:
            self.aspect_ratios = [aspect_ratios_global] * len(feature_map_sizes)
        elif aspect_ratios is not None:
            if len(aspect_ratios) == len(feature_map_sizes):
                self.aspect_ratios = aspect_ratios
            else:
                raise ValueError('If the aspect ratios are specified per feature map, they must match.')
        else:
            raise ValueError('Some kind of aspect ratio is necessary. Either a global specification or one for'
                             ' each feature map.')

        if fixed_scale is None:
            self.scales = np.linspace(scale_min, scale_max, len(self.feature_map_sizes) + 1)
        else:
            self.scales = fixed_scale
            if not len(fixed_scale) == len(self.feature_map_sizes) + 1:
                raise ValueError("The number of scaling must be one more than the number of feature maps!")

        # create all default boxes. This only depends on the initial arguments
        self.default_boxes = self.create_default_boxes()

        # One hot coding for the objects
        self.background_id = np.array([1.0, 0.0])
        self.bee_id = np.array([0.0, 1.0])

        # standard deviation for box standardizing
        self.box_std = [1.0, 1.0, 1.0, 1.0]
        if isinstance(standardizing, list):
            if len(standardizing) == 4:
                self.box_std = standardizing
            else:
                raise ValueError(
                    "Exact four values in a list needed: The standard deviation for [xmin, ymin, xmax, ymax].")

    @classmethod
    def from_config(cls, obj):
        """
        Loads the settings for the default boxes from a configuration file.
        :param obj: Path to config file or opened config file as dictionary
        :return: the DefaultBoxHandler class with the setting from the config file
        """
        if type(obj) == dict:
            args = obj
        elif type(obj) == str:
            if splitext(obj)[1] == '.conf':
                with open(obj, 'rb') as file:
                    args = pickle.load(file)

            else:
                raise ValueError("Invalid configuration file. Should be '.conf'.")
        args['image_size'] = args['input_shape'][:2]
        args['standardizing'] = args['standardizing_boxes']
        return cls(**args)

    def to_config(self, path):
        """
        Stores the current configurations in a binary file.
        :param path: the path to store the configuration. Extension is not needed.
        :return: Nothing.
        """
        d = {'image_size': (self.input_width, self.input_height), 'feature_map_sizes': self.feature_map_sizes,
             'fixed_scale': self.fixed_scale, 'scale_min': self.scale_min, 'scale_max': self.scale_max,
             'aspect_ratios': self.aspect_ratios, 'use_bonus_square_box': self.use_bonus_square_box,
             'standardizing': self.box_std}
        with open(path + '.config', 'wb') as file:
            pickle.dump(d, file, protocol=4)  # since python 3.4

    @property
    def num_boxes(self):
        """
        Return the total number of default boxes generated by this class.
        :return: integer: number default boxes
        """
        return self.default_boxes.shape[0]

    def fit(self, ground_truth):
        """
        Fits the standard deviation to the given data, to standardized the training data during training. In the best
        case the data transferred here corresponds to the entire training data set
        :param ground_truth: list with length batch_size of arrays with shape (#boxes, 4).
        The boxes are in corner format.
        """
        # reset standard deviation
        self.box_std = [1., 1., 1., 1.]
        # assign the ground truth boxes to the default boxes and encode them
        encoded = self.encode_default_boxes(ground_truth, iou_threshold=0.5, bg_threshold=0.3, verbose=False,
                                            cut_default_boxes=True)
        # calculate the standard deviation over all boxes and set the class variable
        mask = np.empty(encoded[..., 2:6].shape)
        mask[:, :] = (encoded[:, :, 0] == 1)[:, :, np.newaxis]
        mask_zeros = np.ma.masked_array(encoded[..., 2:6], mask=mask)
        self.box_std = np.std(mask_zeros, axis=(0, 1)).compressed().tolist()
        box_mean = np.mean(mask_zeros, axis=(0, 1)).compressed().tolist()
        print("MEAN", box_mean)
        print("FITTED:", self.box_std)

    def create_default_boxes(self, verbose=True):
        """
        Creates the default boxes for all feature layers and combines them in one array.
        Normalizes the default boxes to be in [0, 1], changes the format to 'corner' and do clipping.
        :param verbose: whether to show a progressbar
        :return: Array with all default boxes of shape (#boxes, 4)
        """
        boxes = []
        if verbose:
            it = tqdm(range(len(self.feature_map_sizes)), desc="Creating default boxes for every feature layer")
        else:
            it = range(len(self.feature_map_sizes))

        for i in it:
            # For each feature map the appropriate default boxes are created
            boxes.append(self.create_default_boxes_for_ith_layer(i))
        boxes = np.concatenate(boxes)
        # convert boxes from center to corner format
        converted_boxes = convert_boxes(boxes, 'center2corner')
        # clip the boxes so that it fits in the image
        clipped_boxes = clip_boxes(converted_boxes, image_size=(self.input_width, self.input_height),
                                   box_format='corner', normalized=False)
        # normalize the boxes in range [0, 1]
        normalized_boxes = absolute2relative(clipped_boxes, box_format='corner',
                                             image_size=(self.input_width, self.input_height))
        return normalized_boxes

    def create_default_boxes_for_ith_layer(self, i):
        """
        According to the original SSD paper the default boxes are created for each feature map. For a given scale s,
        the side length for each aspect ration a is calculated as follows: w = s * sqrt(a) and h = s / sqrt(a).
        The placement of the boxes is done using a grid of the size of the feature map. For a feature map of size n x m
        the centers are calculated as ((i + 0.5) / n, (j + 0.5) / m) for i in [0,n) and j in [0, m).
        :param i: The index of the feature map is the position of the i-th feature map.
        :return: an array of all default boxes of layer i. Shape (#boxes, 4) with #boxes = n * m * a
        """
        # Check if the feature map exists
        if not 0 <= i < len(self.feature_map_sizes):
            raise ValueError("There is no feature map with the number {0}. It must be between 0 and {1}".format(i, len(
                self.feature_map_sizes) - 1))
        # list to store all the side lengths
        box_sizes = []
        size = min(self.input_height, self.input_width)
        for ratio in self.aspect_ratios[i]:
            # in the original paper, the proposal is made to choose an additional, special side length for aspect
            # ratio 1.
            if ratio == 1 and self.use_bonus_square_box:
                bonus_width = size * np.sqrt(self.scales[i] * self.scales[i + 1]) * np.sqrt(ratio)
                bonus_height = size * np.sqrt(self.scales[i] * self.scales[i + 1]) / np.sqrt(ratio)
                box_sizes.append((bonus_width, bonus_height))
            width = size * self.scales[i] * np.sqrt(ratio)
            height = size * self.scales[i] / np.sqrt(ratio)
            box_sizes.append([width, height])

        offset = 0.5
        feature_h, feature_w = self.feature_map_sizes[i]
        feature_h, feature_w = int(feature_h), int(feature_w)
        # calculate the center points coordinates in x and y direction
        # Shape (feature_map_width,)
        cx = np.linspace((self.input_width * offset) / feature_w,
                         (self.input_width * (offset + feature_w - 1)) / feature_w, feature_w)
        # Shape (feature_map_height,)
        cy = np.linspace((self.input_height * offset) / feature_h,
                         (self.input_height * (offset + feature_h - 1)) / feature_h, feature_h)
        # create a 2d array from the two 1D arrays: combine each cx and cy together. Flow direction is row.
        # Shape (feature_map_size, 2) with feature_map_size = feature_map_width * feature_map_height
        grid_x, grid_y = np.meshgrid(cx, cy)
        centers = np.array([grid_x.T, grid_y.T]).T.reshape((-1, 2))
        # repeat each center, since one is needed for each aspect ratio: Shape (feature_map_size * ratios, 2)
        centers_for_all_boxes = np.repeat(centers, repeats=len(self.aspect_ratios[i]), axis=0)
        # repeat all side lengths so that each center has a box for each aspect ratio:
        # Shape (feature_map_size * ratios, 2)
        box_sizes_all = np.tile(box_sizes, reps=(centers.shape[0], 1))
        # merge the center points and the side length: Shape: (feature_map_size * ratios, 4)
        boxes = np.concatenate((centers_for_all_boxes, box_sizes_all), axis=1)
        return boxes

    def encode_default_boxes(self, gt_boxes, iou_threshold=0.5, bg_threshold=0.3, cut_default_boxes=False,
                             verbose=True):
        """
        Assigns the ground truth boxes of a batch of images to the corresponding default boxes.
        First, each ground truth box is assigned to the default box with the highest IoU value. Then each default box is
        matched to a ground truth box with a threshold value above 'iou_threshold'.
        :param gt_boxes: list with length batch_size of arrays with shape (#boxes, 4). The boxes are in corner format.
        :param iou_threshold: Threshold needed in the second step of matching
        :param bg_threshold: if a background box overlaps more than the threshold with an object box, the background is
         set to neutral and has no influence on the training
        :param cut_default_boxes: if True the default boxes are not in the output shape. They are not necessary for the
        loss calculation, so they can be deleted
        :param verbose: whether to show a progressbar
        :return: (batch_size, #default_boxes, #classes + 4 + 4) where the last axis is (one-hot-class-vector, ground
        truth offset, default box)
        If cut_default_boxes shape is: (batch_size, #default_boxes, #classes + 4) where the last axis is
        (one-hot-class-vector, ground truth offset)
        """
        if self.default_boxes is None:
            self.default_boxes = self.create_default_boxes()
        # the number of images in one batch. Will be set in the dataset generator
        batch_size = len(gt_boxes)
        # create a template of shape (#default_boxes, 12)
        result_template_single = np.concatenate((np.zeros(shape=(self.num_boxes, 6)), self.default_boxes), axis=1)
        # set each class on background by default
        result_template_single[:, :2] = self.background_id
        # repeat the template for every image in the batch: (batch_size, #default_boxes, 12)
        result_template = np.repeat(np.expand_dims(result_template_single, axis=0), repeats=batch_size, axis=0)
        # for every image in the batch ...
        it = tqdm(range(batch_size), desc="Matching GT to DFLT boxes") if verbose else range(batch_size)
        for batch_index in it:
            # if there are no gt boxes in the image
            if gt_boxes[batch_index].size == 0:
                continue
            # list that stores the indexes of all matched default boxes. This is needed to encode them later.
            matched_defaults_index = []
            # convert gt boxes from absolute to relative coordinates
            image_gt = absolute2relative(gt_boxes[batch_index], box_format='corner',
                                         image_size=(self.input_width, self.input_height))
            # calculate the iou between every Ground truth box and every default box: Shape (#gt_boxes in image, #boxes)
            iou_matrix = iou(image_gt, self.default_boxes, combinations=True, box_format='corner')

            # First match each ground truth box to the default box with best IoU value. The problem is equal to the
            # bipartite matching problem. Returns gt_box which corresponds to the rows of the cost matrix and thus the
            # ground true boxes. The attribute matched_default corresponds to the columns and gives the index of the
            # default boxes.
            gt_box, matched_default = linear_sum_assignment(iou_matrix, maximize=True)

            # Set the class ID and the gt_box of the matching result into the template
            result_template[batch_index, matched_default, :-4] = np.hstack(
                (np.array(np.tile(self.bee_id, reps=(image_gt.shape[0], 1))), image_gt))
            # Set the cost for the matched boxes (whole column) to 0 so that they are not overwritten in the second step
            iou_matrix[:, matched_default] = 0
            # add the indexes of the matched default boxes to the list
            matched_defaults_index += list(matched_default)

            # Second, every default box gets a gt_box if its IoU threshold is higher than self.iou_threshold. So in the
            # cost matrix for each column, the corresponding row is chosen according to the highest possible value as
            # long as it is above the threshold.
            # For every default box get the highest gt_box: shape (#default_boxes, )
            max_index = np.argmax(iou_matrix, axis=0)

            # Get the according maximum values to the calculated max_index
            max_values = iou_matrix[max_index, range(iou_matrix.shape[1])]

            # Compare these values with the threshold and return a list of indexes that meet the criterion
            matched_default = (max_values > iou_threshold).nonzero()[0]

            # take only the max_index that exceeds the threshold
            max_index_filtered = max_index[matched_default]

            # Set the class ID and the gt_box of the matching result into the template
            result_template[batch_index, matched_default, :-4] = np.hstack(
                (np.array(np.tile(self.bee_id, reps=(max_index_filtered.shape[0], 1))), image_gt[max_index_filtered]))
            # add the indexes of the matched default boxes to the list
            matched_defaults_index += list(matched_default)
            # set the cost matched boxes to zero, so that they dont get removes in the next step
            iou_matrix[:, matched_default] = 0

            # Set all boxes that are to close to the matched boxes to neutral => class 00
            max_iou_background = np.amax(iou_matrix, axis=0)
            matched_default = (max_iou_background > bg_threshold).nonzero()[0]
            result_template[batch_index, matched_default, 0] = 0

            # Bounding box encoding for regression: Convert ground truth boxes into an offset to the default boxes. The
            # offset calculation is made as follows:
            #       * xmin = xmin_gt - xmin_df / w_df / std_xmin
            #       * ymin = ymin_gt - ymin_df / h_df / std_ymin
            #       * xmax = xmax_gt - xmax_df / w_df / std_xmax
            #       * ymax = ymax_gt - ymax_df / h_df / std_ymax
            md_ind = np.expand_dims(matched_defaults_index, axis=-1)  # Expand dimension for broadcasting
            result_template[batch_index, md_ind, 2:6] -= result_template[batch_index, md_ind, 6:10]
            result_template[batch_index, md_ind, [2, 4]] /= (
                    result_template[batch_index, md_ind, 8] - result_template[batch_index, md_ind, 6])
            result_template[batch_index, md_ind, [3, 5]] /= (
                    result_template[batch_index, md_ind, 9] - result_template[batch_index, md_ind, 7])

        # For the standard normalization process, the ground truth boxes are divided by their standard deviation or a
        # given flat value

        result_template[..., 2:6] /= self.box_std
        # sometimes (for example during training) the default boxes are not needed, so they can be cut out
        if cut_default_boxes:
            result_template = result_template[:, :, :6]
        return result_template

    def decode_default_boxes(self, encoded_boxes, confidence_threshold=0.5, nms_threshold=0.5, sigma=0.5,
                             nms_methode='normal', clip=False):
        """
        Decodes the predicted bounding boxes. For this purpose they must be converted from a distance to the default
        boxes into correct coordinates. Then follows the filtering, where first all boxes with a confidence higher
        than the 'confidence_threshold' are taken. After that, non-maximum-suppression algorithm is applied.
        :param encoded_boxes: Input shape is (#batch_size, #default_boxes, 2 + 4)
        :param confidence_threshold:
        :param nms_threshold:
        :param sigma: TODO
        :param nms_methode: TODO
        :param clip: the boxes can be clipped into the image size
        :return:
        # TODO finish commentary
        """
        check_array(encoded_boxes, expected_shape=(None, None, 6))

        # check if the number of predicted boxes is correct
        if not encoded_boxes.shape[1] == self.num_boxes:
            raise InconsistentNumberDefaultBoxError(self.default_boxes.shape[0], encoded_boxes.shape[1])

        # Filter exactly the boxes whose detection confidence is higher than the threshold. This is done here to reduce
        # the array size for later calculations
        # Creating a boolean array as mask for the masked array which keeps values higher than the threshold
        mask = np.tile(np.expand_dims(encoded_boxes[..., 1] < confidence_threshold, axis=-1), reps=6)
        decoded_boxes = np.ma.masked_array(encoded_boxes, mask=mask)

        # Decoding bounding boxes from offset to true boxes
        #       * xmin = xmin_pred  * w_df * std_xmin + xmin_df
        #       * ymin = ymin_pred  * h_df * std_ymin + ymin_df
        #       * xmax = xmax_pred  * w_df * std_xmax + xmax_df
        #       * ymax = ymax_pred  * h_df * std_ymax + ymax_df
        decoded_boxes[:, :, 2:6] *= self.box_std
        decoded_boxes[:, :, [2, 4]] *= np.expand_dims(self.default_boxes[:, 2] - self.default_boxes[:, 0], axis=-1)
        decoded_boxes[:, :, [3, 5]] *= np.expand_dims(self.default_boxes[:, 3] - self.default_boxes[:, 1], axis=-1)
        decoded_boxes[:, :, 2:6] += self.default_boxes
        # Convert the box coordinates from relative to absolute
        decoded_boxes[:, :, 2:6] = relative2absolute(decoded_boxes[:, :, 2:6], (self.input_width, self.input_height),
                                                     'corner')
        # apply non-maximum-suppression
        suppressed_result = nms_np(decoded_boxes[..., 1:6], confidence_threshold=confidence_threshold,
                                   nms_threshold=nms_threshold, sigma=sigma, methode=nms_methode)
        if clip:
            suppressed_result = clip_boxes(suppressed_result, image_size=(self.input_width, self.input_height),
                                           normalized=False)
        return suppressed_result

    def faster_decode_default_boxes(self, encoded_boxes, confidence_threshold=0.5, nms_threshold=0.5, sigma=0.5,
                                    nms_methode='normal', clip=False, no_score=False):
        """
        Decodes the predicted bounding boxes. For this purpose they must be converted from a distance to the default
        boxes into correct coordinates. Then follows the filtering, where first all boxes with a confidence higher
        than the 'confidence_threshold' are taken. After that, non-maximum-suppression algorithm is applied.
        :param encoded_boxes: Input shape is (#batch_size, #default_boxes, 2 + 4)
        :param confidence_threshold:
        :param nms_threshold:
        :param sigma: TODO
        :param nms_methode: TODO
        :param clip: the boxes can be clipped into the image size
        :param no_score:
        :return:
        # TODO finish commentary
        """
        check_array(encoded_boxes, expected_shape=(None, None, 6))

        # check if the number of predicted boxes is correct
        if not encoded_boxes.shape[1] == self.num_boxes:
            raise InconsistentNumberDefaultBoxError(self.default_boxes.shape[0], encoded_boxes.shape[1])

        result = []

        for sample in encoded_boxes:
            mask = sample[:, 1] >= confidence_threshold
            decoded_boxes = sample[mask]
            filtered_defaults = self.default_boxes[mask]
            # Decoding bounding boxes from offset to true boxes
            #       * xmin = xmin_pred  * w_df * std_xmin + xmin_df
            #       * ymin = ymin_pred  * h_df * std_ymin + ymin_df
            #       * xmax = xmax_pred  * w_df * std_xmax + xmax_df
            #       * ymax = ymax_pred  * h_df * std_ymax + ymax_df
            decoded_boxes[:, 2:6] *= self.box_std
            decoded_boxes[:, [2, 4]] *= np.expand_dims(filtered_defaults[:, 2] - filtered_defaults[:, 0], axis=-1)
            decoded_boxes[:, [3, 5]] *= np.expand_dims(filtered_defaults[:, 3] - filtered_defaults[:, 1], axis=-1)
            decoded_boxes[:, 2:6] += filtered_defaults
            # Convert the box coordinates from relative to absolute
            decoded_boxes[:, 2:6] = relative2absolute(decoded_boxes[:, 2:6], (self.input_width, self.input_height),
                                                      'corner')
            # apply non-maximum-suppression
            suppressed_result = nms(decoded_boxes=decoded_boxes[:, 1:6], confidence_threshold=confidence_threshold,
                                    nms_threshold=nms_threshold, sigma=sigma, methode=nms_methode)
            if clip:
                suppressed_result[:, 1:] = clip_boxes(suppressed_result[:, 1:],
                                                      image_size=(self.input_width, self.input_height),
                                                      normalized=False)
            if no_score:
                suppressed_result = suppressed_result[:, 1:]
            result.append(suppressed_result)
        return result

    def visualize_default_boxes(self, layer, mode='boxes', bg_image=None):
        """
        Visualizes the default boxes with OpenCV for a specific layer on a background or blank image.
        :param layer: Number of feature map, starting with 0
        :param mode: draw mode: can be 'boxes' to draw all boxes of one aspect ratio in the middle of the image or 'all'
        to draw all default boxes or 'center' to see the center points of the cells
        :param bg_image: default None to use a white image or path to image source
        :return: image with default boxes drawn on it
        """
        if bg_image:
            image = cv2.imread(bg_image)
            image = cv2.resize(image, (self.input_height, self.input_width))
        else:
            image = np.zeros((self.input_height, self.input_width, 3))
            image[:] = (255, 255, 255)

        boxes = self.create_default_boxes_for_ith_layer(layer)
        if mode == 'boxes':
            boxes = boxes[:len(self.aspect_ratios)]
            boxes[:, [0, 1]] = [self.input_width / 2, self.input_height / 2]
            image = draw_box(image, boxes, box_format='center')
        elif mode == 'all':
            image = draw_box(image, boxes, box_format='center')
        elif mode == 'centers':
            for box in boxes:
                cv2.circle(image, (int(box[0]), int(box[1])), 2, color=(0, 0, 0), thickness=-1)

        cv2.imshow('Default boxes for feature map {0}'.format(layer), image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return image

    def visualize_encoded_boxes(self, gt_boxes, i=0, iou_threshold=0.5, image=None):
        if image is None:
            image = np.zeros((self.input_height, self.input_width, 3))
            image[:] = (255, 255, 255)
        r = self.encode_default_boxes(gt_boxes, iou_threshold=iou_threshold)
        r = r[i]
        r = r[r[:, 1] == 1, 6:]
        r = relative2absolute(r, image_size=(self.input_width, self.input_height), box_format='corner')
        image_gt = draw_box(image, gt_boxes[i], box_format='corner', color=(255, 0, 0))
        image_gt = draw_box(image_gt, r, box_format='corner', color=(0, 0, 255))

        cv2.imshow('Encoded boxes', image_gt)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    DefaultBoxHandler.from_config("/home/t9s9/PycharmProjects/BeeMeter/detection/training/Test1/model_config.conf")
