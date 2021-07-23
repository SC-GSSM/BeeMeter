import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Input, Reshape, Concatenate, Softmax
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.utils.layer_utils import count_params

from detection.default_boxes.default_boxes import DefaultBoxHandler
from detection.model.layers import Rescaling, L2Normalization
from detection.model.loss_function import SSDLoss
from detection.model.registry import register_model, load_base_net
from detection.utils.image_processing import Resize, Transformer

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


@register_model
class SSD:
    def __init__(self, model=None, box_handler=None):
        tf.keras.backend.clear_session()
        self.model = model
        self.box_handler = box_handler
        self.feature_map_sizes = box_handler.feature_map_sizes if box_handler else None
        self.normalization_layer = None
        self.config = None

    @property
    def width(self):
        if self.model is not None:
            return self.model.input_shape[2]
        else:
            raise ValueError('Please create or load a model first.')

    @property
    def height(self):
        if self.model is not None:
            return self.model.input_shape[1]
        else:
            raise ValueError('Please create or load a model first.')

    @property
    def channel(self):
        if self.model is not None:
            return self.model.input_shape[3]
        else:
            raise ValueError('Please create or load a model first.')

    def create_model(self, input_shape=(300, 300, 3), n_feature_maps=6, base_net='VGG16', aspect_ratios_per_layer=None,
                     aspect_ratios_global=[1, 2, 3, 1 / 2, 1 / 3], l2_pen=0.0005, use_pretrained=True, scale_min=0.2,
                     scale_max=0.9, fixed_scale=None, use_bonus_square_box=False, standardizing_boxes=None,
                     input_mean=None, input_variance=None, base_net_kwargs={}, **kwargs):
        # save attributes as config
        self.config = locals()

        input_width, input_height, input_channel = input_shape

        base = load_base_net(base_net)(l2_pen=l2_pen, **base_net_kwargs)

        if aspect_ratios_global is not None:
            boxes_per_cell = [len(aspect_ratios_global)] * n_feature_maps
        elif aspect_ratios_per_layer is not None:
            if len(aspect_ratios_per_layer) == n_feature_maps:
                boxes_per_cell = [len(i) for i in aspect_ratios_per_layer]
            else:
                raise ValueError('The number of different boxes per cell must be equal to the number of feature maps.')
        else:
            raise ValueError(
                'A value for the boxes per cell is required. Either the global value or the local per feature map.')

        input_layer = Input(shape=(input_width, input_height, input_channel))

        if input_mean is not None and input_variance is not None:
            self.normalization_layer = Normalization(axis=-1, name='normalization')
            self.normalization_layer.mean = input_mean
            self.normalization_layer.variance = input_variance
            rescaling = self.normalization_layer(input_layer)
        else:
            rescaling = Rescaling(scale=1. / 255, name='rescaling')(input_layer)

        last_layer, norm_layer = base(rescaling)

        conv6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(l2_pen), name='conv6')(last_layer)

        conv7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=l2(l2_pen), name='conv7')(conv6)

        conv8_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_pen), name='conv8_1')(conv7)
        conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv8_padding')(conv8_1)
        conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_pen), name='conv8_2')(conv8_1)

        conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_pen), name='conv9_1')(conv8_2)
        conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv9_padding')(conv9_1)
        conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_pen), name='conv9_2')(conv9_1)

        conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_pen), name='conv10_1')(conv9_2)

        conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                          kernel_initializer='he_normal', kernel_regularizer=l2(l2_pen), name='conv10_2')(conv10_1)

        conv11_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_pen), name='conv11_1')(conv10_2)

        conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                          kernel_initializer='he_normal', kernel_regularizer=l2(l2_pen), name='conv11_2')(conv11_1)

        feature_docking = [norm_layer, conv7, conv8_2, conv9_2, conv10_2, conv11_2]

        output = self.create_feature_layers(filter_sizes=boxes_per_cell, prev_layer=feature_docking, l2_pen=l2_pen)

        self.model = Model(inputs=input_layer, outputs=output)

        self.box_handler = DefaultBoxHandler((self.width, self.height), self.feature_map_sizes,
                                             fixed_scale=fixed_scale,
                                             scale_min=scale_min,
                                             scale_max=scale_max,
                                             aspect_ratios_global=aspect_ratios_global,
                                             aspect_ratios=aspect_ratios_per_layer,
                                             use_bonus_square_box=use_bonus_square_box,
                                             standardizing=standardizing_boxes)

        if use_pretrained:
            self.model = base.load_weights(model=self.model)

        return self

    @classmethod
    def from_config_new(cls, config, weights_path=None):
        """
        Load a SSD model from a config file and optionally load trained weights into it.
        The model will be recreated with the use of the create model function. This can be useful if there are version
        conflicts between the training version of tensorflow and this version.
        """
        instance = cls().create_model(**config)
        instance.config = config
        instance.model.compile()
        if weights_path is not None:
            instance.load_weights(weights_path)
        return instance

    def to_config(self, path):
        self.config['class_name'] = type(self).__name__
        del self.config['self']
        # update box std if necessary
        self.config['standardizing_boxes'] = self.box_handler.box_std
        self.config['feature_map_sizes'] = np.array(self.feature_map_sizes)
        print(self.config)
        with open(path / 'model_config.conf', 'wb') as file:
            pickle.dump(self.config, file, protocol=4)  # since python 3.4

    @classmethod
    def from_config(cls, config, weights_path, training=False):
        """
        Load a SSD model from config file and optionally load trained weights into it. This function uses the Keras
         function load_model instead of recreating it.
        """
        if training:
            ssd_loss = SSDLoss()
            model = load_model(weights_path, compile=True, custom_objects={'Rescaling': Rescaling,
                                                                           'SSDLoss': ssd_loss,
                                                                           'L2Normalization': L2Normalization,
                                                                           'relu6': tf.nn.relu6})
        else:
            model = load_model(weights_path, compile=False, custom_objects={'Rescaling': Rescaling,
                                                                            'L2Normalization': L2Normalization,
                                                                            'relu6': tf.nn.relu6})
        box_handler = DefaultBoxHandler.from_config(config)
        return cls(model=model, box_handler=box_handler)

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_model(self, name=None):
        if name is None:
            name = "ported_model.h5"
        self.model.save(name)

    def plot(self, name=None):
        if name is None:
            name = "model.png"
        tf.keras.utils.plot_model(self.model, show_shapes=True, to_file=name)

    def summary(self):
        self.model.summary()

    def estimate_gpu(self, gpu_size=16000):
        """
        Estimate the max. batch size on given GPU memory.
        """
        gpu_byte = gpu_size * 1e6
        trainable_param = self.count_parameter()[0]
        total_tensor_size = 0
        for layer in self.model.layers:
            if len(layer.output_shape) == 4:
                total_tensor_size += np.prod(layer.output_shape[1:])
        per_sample = (total_tensor_size + trainable_param) * 4  # cuda 4 bytes per float
        max_batch_size = gpu_byte / per_sample
        print("Total tensor size: {0:,}\nSize per sample: {1:,}\n\nEstimated batch size: {2:.2f}".format(
            total_tensor_size,
            per_sample,
            max_batch_size))

    def computational_cost(self, verbose=True):
        """ Calculates the number of multiplications-adds on all types of convolutions in this network. """
        total_cost = 0
        total_param = 0
        for l in self.model.layers:
            if type(l) == Conv2D or type(l) == tf.keras.layers.DepthwiseConv2D:
                _, input_h, input_w, input_c = l.input_shape
                _, output_h, output_w, output_c = l.output_shape
                kernel_h, kernel_w = l.kernel_size
                output_c = l.filters if type(l) == Conv2D else 1
                if l.bias is not None:
                    bias_shape = l.bias.shape[0]
                else:
                    bias_shape = 0

                total_cost += output_h * output_w * kernel_h * kernel_w * input_c * output_c  # add bias?
                total_param += kernel_w * kernel_h * input_c * output_c + bias_shape
        if verbose:
            print("Total cost of convolutions: {0:,}".format(total_cost))
            print("Total parameters of convolutions: {0:,}".format(total_param))
        return total_cost

    def count_parameter(self, verbose=True):
        trainable_count = count_params(self.model.trainable_weights)
        non_trainable_count = count_params(self.model.non_trainable_weights)
        if verbose:
            print("Feature Maps: {0} Boxes: {1}".format([list(i) for i in self.feature_map_sizes],
                                                        self.box_handler.num_boxes))
            print('Total params: {0:,}'.format(trainable_count + non_trainable_count))
            print('Trainable params: {0:,}'.format(trainable_count))
            print('Non-trainable params: {0:,}'.format(non_trainable_count))
        return trainable_count, non_trainable_count

    def create_feature_layers(self, filter_sizes, prev_layer, l2_pen):
        self.feature_map_sizes = []
        box_layers = []
        class_layers = []
        for nr, filters in enumerate(filter_sizes):
            # Shape (batch, height, width, #boxes * 2)
            conv_class = Conv2D(filters * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                name='class_{0}'.format(nr), kernel_initializer='he_normal',
                                kernel_regularizer=l2(l2_pen))(prev_layer[nr])
            #
            self.feature_map_sizes.append(conv_class.get_shape()[1:3])
            # Shape (batch, height * width * #boxes, 2)
            conv_class_reshaped = Reshape((-1, 2), name='class_{0}_reshaped'.format(nr))(conv_class)
            #
            class_layers.append(conv_class_reshaped)

            # Shape (batch, height, width, #boxes * 4)
            conv_box = Conv2D(filters * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              name='box_{0}'.format(nr), kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_pen))(prev_layer[nr])
            # Shape (batch, height * width * #boxes, 4)
            conv_box__reshaped = Reshape((-1, 4), name='box_{0}_reshaped'.format(nr))(conv_box)
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

    def _predict(self, x, method=0):
        if method == 0:
            prediction = self.model.predict(x)
        elif method == 1:
            prediction = self.model(np.copy(x), training=False).numpy()
        elif method == 2:
            prediction = self.model.predict_on_batch(x)
        else:
            raise ValueError("Unknown prediction methode.")
        return prediction

    def predict(self, x, batch_size=1, conf_threshold=0.5, nms_threshold=0.3, sigma=0.5, nms_methode='normal',
                forward_methode=0, faster_decode=True):
        image_height, image_width, _ = x[0].shape
        result = []
        # preprocess input images
        x = Transformer(operations=[Resize(output_width=self.width, output_height=self.height)])(x, labels=None)

        for i in range(0, len(x), batch_size):
            data = np.array(x[i:i + batch_size])
            prediction = self._predict(data, method=forward_methode)

            decode_func = self.box_handler.faster_decode_default_boxes if faster_decode else self.box_handler.decode_default_boxes
            decoded = decode_func(prediction, confidence_threshold=conf_threshold,
                                  nms_threshold=nms_threshold, sigma=sigma,
                                  nms_methode=nms_methode)
            result += decoded

        for i in result:
            # scale the predicted boxes to the size of the input images
            i[..., [1, 3]] = np.round(i[..., [1, 3]] * (image_width / self.width), decimals=0)
            i[..., [2, 4]] = np.round(i[..., [2, 4]] * (image_height / self.height), decimals=0)
        return result


if __name__ == '__main__':
    x = SSD().create_model()
    x.model.summary()
