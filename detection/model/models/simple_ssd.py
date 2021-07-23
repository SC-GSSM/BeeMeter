import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Input, Reshape, Concatenate, Softmax, DepthwiseConv2D, \
    BatchNormalization, ReLU, Activation
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.utils.layer_utils import count_params
from tqdm import tqdm

from detection.model.layers import Rescaling, L2Normalization
from detection.default_boxes.default_boxes import DefaultBoxHandler
from detection.model.registry import register_model, load_base_net
from detection.model.loss_function import SSDLoss
from detection.model.models.ssd import SSD
from detection.model.base_network.mobile_net import MobileNetV2


class SimpleSSD(SSD):
    def __init__(self, model=None, box_handler=None):
        super().__init__(model=model, box_handler=box_handler)

    def create_model(self, input_shape=(300, 300, 3), n_feature_maps=6,
                     aspect_ratios_global=[1, 2, 3, 1 / 2, 1 / 3], l2_pen=0.0005, use_pretrained=True, scale_min=0.2,
                     scale_max=0.9, fixed_scale=None, use_bonus_square_box=False, standardizing_boxes=None,
                     input_mean=None, input_variance=None, base_net_kwargs={}, **kwargs):

        boxes_per_cell = [aspect_ratios_global] * n_feature_maps

        input_layer = Input(shape=(height, width, channel), dtype=tf.uint8)
        rescaling = Rescaling(scale=1. / 255, name='rescaling')(input_layer)

        if isinstance(base_net, str):
            if base_net == 'VGG19':
                base_layers = VGG19(include_top=False, weights=None)
            elif base_net == 'VGG16':
                base_layers = VGG16(include_top=False, weights=None)
        else:
            base_layers = BaseNetwork(l2_pen=l2_pen).from_list(base_net)(x=rescaling)
            base_layers_featured = itemgetter(*feature_maps)(base_layers)  # base_layers[-n_feature_maps:]

        output_layer = self.create_feature_layers(filter_sizes=boxes_per_cell, appendix=base_layers_featured,
                                                  l2_pen=l2_pen)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        return self
