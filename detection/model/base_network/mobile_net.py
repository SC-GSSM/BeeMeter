import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, DepthwiseConv2D, Add, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_file

from detection.model.registry import register_base_net


@register_base_net
class MobileNet:
    def __init__(self, alpha=1.0, rho=1, l2_pen=0.0005, activation=tf.nn.relu6, kernel_initializer='he_normal',
                 predictors=['conv11'], **kwargs):
        self.block_count = 0
        self.alpha = alpha
        self.rho = rho
        self.act = activation
        self.predictors = predictors
        self.conv_kwargs = {'kernel_initializer': kernel_initializer, 'kernel_regularizer': l2(l2_pen)}

    def __call__(self, input_layer):
        conv0 = self._conv_block(32, strides=(2, 2))(input_layer)
        conv1 = self._dw_block(64)(conv0)

        conv2 = self._dw_block(128, strides=(2, 2))(conv1)
        conv3 = self._dw_block(128)(conv2)

        conv4 = self._dw_block(256, strides=(2, 2))(conv3)
        conv5 = self._dw_block(256)(conv4)

        conv6 = self._dw_block(512, strides=(2, 2))(conv5)
        conv7 = self._dw_block(512)(conv6)
        conv8 = self._dw_block(512)(conv7)
        conv9 = self._dw_block(512)(conv8)
        conv10 = self._dw_block(512)(conv9)
        conv11 = self._dw_block(512)(conv10)

        conv12 = self._dw_block(1024, strides=(2, 2))(conv11)
        conv13 = self._dw_block(1024)(conv12)

        predictor_layers = []
        for layer in self.predictors:
            predictor_layers.append(locals()[layer])

        return conv13, predictor_layers

    def _conv_block(self, filters, kernel=(3, 3), strides=(1, 1)):
        def wrapper(x):
            corrected_filters = int(filters * self.alpha)
            x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(x)
            x = Conv2D(corrected_filters, kernel, padding='valid', use_bias=False, strides=strides,
                       name='conv1', **self.conv_kwargs)(x)
            x = BatchNormalization(axis=-1, name='conv1_bn')(x)
            return Activation(self.act, name='conv1_relu')(x)

        return wrapper

    def _dw_block(self, filters, strides=(1, 1)):
        def wrapper(x):
            corrected_filters = int(self.alpha * filters)

            if strides != (1, 1):
                # ((0, 1), (0, 1)),
                x = ZeroPadding2D(padding=MobileNetV2.correct_pad(x, 3), name='conv_pad_%d' % self.block_count)(x)

            x = DepthwiseConv2D((3, 3), padding='same' if strides == (1, 1) else 'valid',
                                depth_multiplier=self.rho,
                                strides=strides,
                                use_bias=False,
                                name='conv_dw_%d' % self.block_count,
                                **self.conv_kwargs)(x)

            x = BatchNormalization(axis=-1, name='conv_dw_%d_bn' % self.block_count)(x)
            x = Activation(self.act, name='conv_dw_%d_relu' % self.block_count)(x)

            x = Conv2D(corrected_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
                       name='conv_pw_%d' % self.block_count, **self.conv_kwargs)(x)
            x = BatchNormalization(axis=-1, name='conv_pw_%d_bn' % self.block_count)(x)
            return Activation(self.act, name='conv_pw_%d_relu' % self.block_count)(x)

        self.block_count += 1
        return wrapper

    def load_weights(self, model):
        if self.rho != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if self.alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        # if rows != cols or rows not in [128, 160, 192, 224]:
        #     rows = 224
        #     print('Warning: `input_shape` is undefined or non-square, '
        #           'or `rows` is not in [128, 160, 192, 224]. '
        #           'Weights for input shape (224, 224) will be'
        #           ' loaded as the default.')

        if self.alpha == 1.0:
            alpha_text = '1_0'
        elif self.alpha == 0.75:
            alpha_text = '7_5'
        elif self.alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/')
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
        weights = get_file(model_name, BASE_WEIGHT_PATH, cache_subdir='models')
        model.load_weights(weights, by_name=True)
        return model


@register_base_net
class MobileNetV2:
    def __init__(self, alpha=1.0, l2_pen=0.0005, activation=tf.nn.relu6, kernel_initializer='he_normal',
                 predictors=['expand13'], **kwargs):
        self.alpha = alpha
        self.predictors = predictors
        self.act = activation
        self.conv_kwargs = {'kernel_initializer': kernel_initializer, 'kernel_regularizer': l2(l2_pen)}

    def __call__(self, input_layer):
        first_block_filters = self._make_divisible(32 * self.alpha, 8)
        x = ZeroPadding2D(padding=self.correct_pad(input_layer, 3), name='Conv1_pad')(input_layer)
        x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='valid', use_bias=False, name='Conv1',
                   **self.conv_kwargs)(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
        x = Activation(self.act, name='Conv1_relu')(x)

        expand0, _, x = self._inverted_res_block(x, filters=16, alpha=self.alpha, stride=1, expansion=1, block_id=0)

        expand1, _, x = self._inverted_res_block(x, filters=24, alpha=self.alpha, stride=2, expansion=6, block_id=1)
        expand2, _, x = self._inverted_res_block(x, filters=24, alpha=self.alpha, stride=1, expansion=6, block_id=2)

        expand3, _, x = self._inverted_res_block(x, filters=32, alpha=self.alpha, stride=2, expansion=6, block_id=3)
        expand4, _, x = self._inverted_res_block(x, filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=4)
        expand5, _, x = self._inverted_res_block(x, filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=5)

        expand6, _, x = self._inverted_res_block(x, filters=64, alpha=self.alpha, stride=2, expansion=6, block_id=6)
        expand7, _, x = self._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=7)
        expand8, _, x = self._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=8)
        expand9, _, x = self._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=9)

        expand10, _, x = self._inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=10)
        expand11, _, x = self._inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=11)
        expand12, _, x = self._inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=12)

        expand13, _, x = self._inverted_res_block(x, filters=160, alpha=self.alpha, stride=2, expansion=6, block_id=13)
        expand14, _, x = self._inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=14)
        expand15, _, x = self._inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=15)

        expand16, _, x = self._inverted_res_block(x, filters=320, alpha=self.alpha, stride=1, expansion=6, block_id=16)

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if self.alpha > 1.0:
            last_block_filters = self._make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280

        x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1', **self.conv_kwargs)(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = Activation(self.act, name='out_relu')(x)

        predictor_layers = []
        for layer in self.predictors:
            predictor_layers.append(locals()[layer])

        return x, predictor_layers

    def _inverted_res_block(self, inputs, expansion, stride, alpha, filters, block_id):
        """Inverted ResNet block."""
        in_channels = tf.keras.backend.int_shape(inputs)[-1]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = self._make_divisible(pointwise_conv_filters, 8)
        prefix = 'block_{}_'.format(block_id)

        if block_id:
            # Expand
            expand = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None,
                            name=prefix + 'expand', **self.conv_kwargs)(inputs)
            expand = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(expand)
            expand = Activation(self.act, name=prefix + 'expand_relu')(expand)
        else:
            prefix = 'expanded_conv_'
            expand = inputs

        # Depthwise
        if stride == 2:
            expand_out = ZeroPadding2D(padding=self.correct_pad(expand, 3), name=prefix + 'pad')(expand)
        else:
            expand_out = expand
        depthwise = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False,
                                    padding='same' if stride == 1 else 'valid', name=prefix + 'depthwise',
                                    **self.conv_kwargs)(expand_out)
        depthwise = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(depthwise)
        depthwise = Activation(self.act, name=prefix + 'depthwise_relu')(depthwise)

        # Project
        project = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None,
                         name=prefix + 'project', **self.conv_kwargs)(depthwise)
        project = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(project)

        if in_channels == pointwise_filters and stride == 1:
            return expand, depthwise, Add(name=prefix + 'add')([inputs, project])
        return expand, depthwise, project

    @staticmethod
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    @staticmethod
    def correct_pad(inputs, kernel_size):
        """Returns a tuple for zero-padding for 2D convolution with downsampling.

        Arguments:
          inputs: Input tensor.
          kernel_size: An integer or tuple/list of 2 integers.

        Returns:
          A tuple.
        """
        img_dim = 2 if tf.keras.backend.image_data_format() == 'channels_first' else 1
        input_size = tf.keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if input_size[0] is None:
            adjust = (1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
        correct = (kernel_size[0] // 2, kernel_size[1] // 2)
        return ((correct[0] - adjust[0], correct[0]),
                (correct[1] - adjust[1], correct[1]))
