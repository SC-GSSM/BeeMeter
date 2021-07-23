import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Input, Reshape, Concatenate, Softmax, DepthwiseConv2D, \
    BatchNormalization, Activation, Conv2DTranspose, Add

from detection.model.models.ssd_lite import SSDLite
from detection.model.registry import register_model, load_base_net
from detection.model.layers import Rescaling
from detection.default_boxes.default_boxes import DefaultBoxHandler


@register_model
class FFSSD(SSDLite):
    def __init__(self, model=None, box_handler=None):
        super().__init__(model=model, box_handler=box_handler)

    def create_model(self, input_shape=(300, 300, 3), base_net='MobileNetV2', aspect_ratios_per_layer=None,
                     aspect_ratios_global=[1, 2, 3, 1 / 2, 1 / 3], l2_pen=0.0005, scale_min=0.2, activation=tf.nn.relu6,
                     scale_max=0.9, fixed_scale=None, use_bonus_square_box=False, standardizing_boxes=None, fuse=None,
                     kernel_initializer='he_normal', keep_top=4, base_net_kwargs={}, **kwargs):

        # save attributes as config
        self.config = locals()
        input_width, input_height, input_channel = input_shape
        input_layer = Input(shape=(input_height, input_width, input_channel))

        rescaling = Rescaling(scale=1. / 255, name='rescaling')(input_layer)

        base_model = load_base_net(base_net)(l2_pen=l2_pen, activation=activation, **base_net_kwargs)
        last_layer, predictors = base_model(rescaling)

        top1 = self._top_dw_block(512, activation, kernel_initializer, block_count=1)(last_layer)
        top2 = self._top_dw_block(256, activation, kernel_initializer, block_count=2)(top1)
        top3 = self._top_dw_block(256, activation, kernel_initializer, block_count=3)(top2)
        top4 = self._top_dw_block(128, activation, kernel_initializer, block_count=4)(top3)

        predictor_docking = predictors + [last_layer] + [top1, top2, top3, top4][:keep_top]
        print(len(predictor_docking))
        to_delete = []
        if fuse is not None:
            for i, j in fuse:
                sum_mod = self.sum_module(name='Sum_module', filters=256, depthwise=True)(lower=predictor_docking[i],
                                                                                          upper=predictor_docking[j])
                predictor_docking.pop(i)
                predictor_docking.pop(j)
                predictor_docking.insert(i, sum_mod)

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

    @staticmethod
    def sum_module(name, filters=512, activation=tf.nn.relu, depthwise=True):
        def wrapper(lower, upper):
            print(lower.shape, upper.shape)
            if depthwise:
                lower_dw = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same',
                                           name=name + '_lower_dw')(lower)
                lower_conv = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), name=name + '_lower_pw')(
                    lower_dw)
            else:
                lower_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                    name=name + '_lower_conv')(lower)
            lower_bn = BatchNormalization(axis=-1, name=name + '_lower_bn')(lower_conv)

            if depthwise:
                upper_dedw = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same',
                                             name=name + '_upper_dedw')(upper)
                upper_deconv = Conv2DTranspose(filters=filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                               activation=None, name=name + '_upper_deconv')(upper_dedw)
                upper_deconv = ZeroPadding2D(((0, 1), (1, 1)), name=name + "_deconv_pad")(upper_deconv)
                upper_dw = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                           name=name + '_upper_dw')(upper_deconv)
                upper_conv = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), name=name + '_upper_pw')(
                    upper_dw)
            else:
                upper_deconv = Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(2, 2),
                                               activation=None, name=name + '_upper_deconv', padding='same')(upper)
                upper_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                    name=name + '_upper_conv')(upper_deconv)
            upper_bn = BatchNormalization(axis=-1, name=name + '_upper_bn')(upper_conv)

            print(lower_bn.shape, upper_bn.shape)
            add = Add(name=name + '_add')([lower_bn, upper_bn])
            return Activation(activation, name=name + '_act')(add)

        return wrapper
