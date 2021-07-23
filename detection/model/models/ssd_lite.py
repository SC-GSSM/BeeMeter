import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Input, Reshape, Concatenate, Softmax, DepthwiseConv2D, \
    BatchNormalization, Activation, Conv2DTranspose, Add
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


@register_model
class SSDLite(SSD):
    def __init__(self, model=None, box_handler=None):
        super().__init__(model=model, box_handler=box_handler)

    def create_model(self, input_shape=(300, 300, 3), base_net='MobileNetV2', aspect_ratios_per_layer=None,
                     aspect_ratios_global=[1, 2, 3, 1 / 2, 1 / 3], l2_pen=0.0005, scale_min=0.2, activation=tf.nn.relu6,
                     scale_max=0.9, fixed_scale=None, use_bonus_square_box=False, standardizing_boxes=None,
                     kernel_initializer='he_normal', keep_top=4, base_net_kwargs={}, **kwargs):
        # save attributes as config
        self.config = locals()
        if len(input_shape) == 2:
            model_input_shape = input_shape[1], input_shape[0]
        elif len(input_shape) == 3:
            model_input_shape = input_shape[1], input_shape[0], input_shape[2]
        else:
            raise ValueError("Wrong input shape.")
        input_layer = Input(shape=model_input_shape)

        rescaling = Rescaling(scale=1. / 255, name='rescaling')(input_layer)

        base_model = load_base_net(base_net)(l2_pen=l2_pen, activation=activation, **base_net_kwargs)
        last_layer, predictors = base_model(rescaling)

        top1 = self._top_dw_block(512, activation, kernel_initializer, block_count=1)(last_layer)
        top2 = self._top_dw_block(256, activation, kernel_initializer, block_count=2)(top1)
        top3 = self._top_dw_block(256, activation, kernel_initializer, block_count=3)(top2)
        top4 = self._top_dw_block(128, activation, kernel_initializer, block_count=4)(top3)

        predictor_docking = predictors + [last_layer] + [top1, top2, top3, top4][:keep_top]

        if aspect_ratios_global is not None:
            boxes_per_cell = [len(aspect_ratios_global)] * len(predictor_docking)
        elif aspect_ratios_per_layer is not None:
            if len(aspect_ratios_per_layer) == len(predictor_docking):
                boxes_per_cell = [len(i) for i in aspect_ratios_per_layer]
            else:
                raise ValueError('The number of different boxes per cell must be equal to the number of feature maps.')
        else:
            raise ValueError(
                'A value for the boxes per cell is required. Either the global value or the local per feature map.')

        output = self.create_prediction_layers(filter_sizes=boxes_per_cell, prev_layer=predictor_docking,
                                               activation=activation, l2_pen=l2_pen,
                                               kernel_initializer=kernel_initializer,
                                               depthwise_predictors=True)

        self.model = Model(inputs=input_layer, outputs=output)

        self.box_handler = DefaultBoxHandler((self.width, self.height), self.feature_map_sizes,
                                             fixed_scale=fixed_scale,
                                             scale_min=scale_min,
                                             scale_max=scale_max,
                                             aspect_ratios_global=aspect_ratios_global,
                                             aspect_ratios=aspect_ratios_per_layer,
                                             use_bonus_square_box=use_bonus_square_box,
                                             standardizing=standardizing_boxes)

        return self

    def create_prediction_layers(self, filter_sizes, prev_layer, activation, l2_pen, kernel_initializer,
                                 depthwise_predictors=True):
        self.feature_map_sizes = []
        box_layers = []
        class_layers = []
        for nr, filters in enumerate(filter_sizes):
            # Classes
            # Shape (batch, height, width, #boxes * 2)
            class_conv = self._prediction_block(filters * 2, activation, name='class_{0}'.format(nr),
                                                depthwise=depthwise_predictors,
                                                l2_pen=l2_pen, kernel_initializer=kernel_initializer)(prev_layer[nr])
            #
            self.feature_map_sizes.append(class_conv.get_shape()[1:3])
            # Shape (batch, height * width * #boxes, 2)
            conv_class_reshaped = Reshape((-1, 2), name='class_{0}_reshaped'.format(nr))(class_conv)
            #
            class_layers.append(conv_class_reshaped)

            # Boxes
            # Shape (batch, height, width, #boxes * 4)
            box_conv = self._prediction_block(filters * 4, activation, name='box_{0}'.format(nr),
                                              depthwise=depthwise_predictors,
                                              l2_pen=l2_pen, kernel_initializer=kernel_initializer)(prev_layer[nr])

            # Shape (batch, height * width * #boxes, 4)
            conv_box__reshaped = Reshape((-1, 4), name='box_{0}_reshaped'.format(nr))(box_conv)
            #
            box_layers.append(conv_box__reshaped)

        # Shape (batch, #all_boxes, 2)
        class_concat = Concatenate(axis=1, name='class_concat')(class_layers)
        # Activation: Shape (batch, #all_boxes, 2)
        class_softmax = Softmax(name='class_softmax')(class_concat)

        # Shape (batch, #all_boxes, 4)
        box_concat = Concatenate(axis=1, name='box_concat')(box_layers)

        # Shape (batch, #all_boxes, 6)
        box_class = Concatenate(axis=2, name='box_and_class')([class_softmax, box_concat])

        return box_class

    @staticmethod
    def _top_dw_block(filters, activation, kernel_initializer, block_count):
        def wrapper(x):
            name = "ssd_ext_"
            x = Conv2D(filters // 2, kernel_size=1, padding='same', use_bias=False, activation=None,
                       kernel_initializer=kernel_initializer, name='{0}conv1_{1}'.format(name, block_count))(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='{0}conv1_bn_{1}'.format(name, block_count))(x)
            x = Activation(activation, name='{0}conv1_act_{1}'.format(name, block_count))(x)

            x = ZeroPadding2D(padding=MobileNetV2.correct_pad(x, 3), name='{0}pad_{1}'.format(name, block_count))(x)

            x = DepthwiseConv2D(kernel_size=(3, 3), padding='valid', strides=(2, 2), use_bias=False,
                                kernel_initializer=kernel_initializer, activation=None,
                                name='{0}dw_{1}'.format(name, block_count))(x)

            x = BatchNormalization(axis=-1, name='{0}dw_bn_{1}'.format(name, block_count))(x)
            x = Activation(activation, name='{0}dw_act_{1}'.format(name, block_count))(x)

            x = Conv2D(filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), activation=None,
                       kernel_initializer=kernel_initializer, name='{0}conv2_{1}'.format(name, block_count))(x)
            x = BatchNormalization(axis=-1, name='{0}conv2_bn_{1}'.format(name, block_count))(x)
            x = Activation(activation, name='{0}conv2_act_{1}'.format(name, block_count))(x)
            return x

        return wrapper

    @staticmethod
    def _prediction_block(filters, activation, name, l2_pen=0.0005, kernel_initializer='he_normal', depthwise=True):
        def wrapper(x):
            if depthwise:
                dw = DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=1, strides=(1, 1), use_bias=False,
                                     padding='same', name='{0}_dw'.format(name), activation=None,
                                     kernel_initializer='he_normal')(x)
                bn1 = BatchNormalization(axis=-1, name='{0}_bn1'.format(name))(dw)
                act = Activation(activation, name='{0}_act'.format(name))(bn1)
                conv = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                              name='{0}_conv'.format(name), kernel_initializer='he_normal',
                              activation=None)(act)
                bn2 = BatchNormalization(axis=-1, name='{0}_bn2'.format(name))(conv)
                return bn2
            else:
                return Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              name='{0}_conv'.format(name), kernel_initializer=kernel_initializer,
                              kernel_regularizer=l2(l2_pen), activation=None)(x)

        return wrapper
