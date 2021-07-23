from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_file

from detection.model.layers import L2Normalization
from detection.model.registry import register_base_net


@register_base_net
class VGG19:
    def __init__(self, l2_pen=0.0005):
        self.conv_kwargs = {'activation': 'relu', 'padding': 'same', 'use_bias': True,
                            'kernel_initializer': 'he_normal', 'kernel_regularizer': l2(l2_pen)}

    def __call__(self, input_layer):
        block1_conv1 = Conv2D(64, (3, 3), name='block1_conv1', **self.conv_kwargs)(input_layer)
        block1_conv2 = Conv2D(64, (3, 3), name='block1_conv2', **self.conv_kwargs)(block1_conv1)
        block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', padding='same')(block1_conv2)

        block2_conv1 = Conv2D(128, (3, 3), name='block2_conv1', **self.conv_kwargs)(block1_pool)
        block2_conv2 = Conv2D(128, (3, 3), name='block2_conv2', **self.conv_kwargs)(block2_conv1)
        block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', padding='same')(block2_conv2)

        block3_conv1 = Conv2D(256, (3, 3), name='block3_conv1', **self.conv_kwargs)(block2_pool)
        block3_conv2 = Conv2D(256, (3, 3), name='block3_conv2', **self.conv_kwargs)(block3_conv1)
        block3_conv3 = Conv2D(256, (3, 3), name='block3_conv3', **self.conv_kwargs)(block3_conv2)
        block3_conv4 = Conv2D(256, (3, 3), name='block3_conv4', **self.conv_kwargs)(block3_conv3)
        block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(block3_conv4)

        block4_conv1 = Conv2D(512, (3, 3), name='block4_conv1', **self.conv_kwargs)(block3_pool)
        block4_conv2 = Conv2D(512, (3, 3), name='block4_conv2', **self.conv_kwargs)(block4_conv1)
        block4_conv3 = Conv2D(512, (3, 3), name='block4_conv3', **self.conv_kwargs)(block4_conv2)
        block4_conv4 = Conv2D(512, (3, 3), name='block4_conv4', **self.conv_kwargs)(block4_conv3)
        block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same')(block4_conv4)

        block5_conv1 = Conv2D(512, (3, 3), name='block5_conv1', **self.conv_kwargs)(block4_pool)
        block5_conv2 = Conv2D(512, (3, 3), name='block5_conv2', **self.conv_kwargs)(block5_conv1)
        block5_conv3 = Conv2D(512, (3, 3), name='block5_conv3', **self.conv_kwargs)(block5_conv2)
        block5_conv4 = Conv2D(512, (3, 3), name='block5_conv4', **self.conv_kwargs)(block5_conv3)
        block5_pool = MaxPooling2D((2, 2), strides=(1, 1), name='block5_pool', padding='same')(block5_conv4)

        norm_block4_conv4 = L2Normalization(name='block4_conv4_norm')(block4_conv4)

        return block5_pool, norm_block4_conv4

    @staticmethod
    def load_weights(model):
        VGG19_IMAGENET = ('https://github.com/fchollet/deep-learning-models/releases/'
                          'download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        weights = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG19_IMAGENET,
                           cache_subdir='models', file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights, by_name=True)
        return model


@register_base_net
class VGG16:
    def __init__(self, l2_pen=0.0005):
        self.l2_pen = l2_pen
        self.conv_kwargs = {'activation': 'relu', 'padding': 'same', 'use_bias': True,
                            'kernel_initializer': 'he_normal', 'kernel_regularizer': l2(l2_pen)}

    def __call__(self, input_layer):
        block1_conv1 = Conv2D(64, (3, 3), name='block1_conv1', **self.conv_kwargs)(input_layer)
        block1_conv2 = Conv2D(64, (3, 3), name='block1_conv2', **self.conv_kwargs)(block1_conv1)
        block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', padding='same')(block1_conv2)

        block2_conv1 = Conv2D(128, (3, 3), name='block2_conv1', **self.conv_kwargs)(block1_pool)
        block2_conv2 = Conv2D(128, (3, 3), name='block2_conv2', **self.conv_kwargs)(block2_conv1)
        block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', padding='same')(block2_conv2)

        block3_conv1 = Conv2D(256, (3, 3), name='block3_conv1', **self.conv_kwargs)(block2_pool)
        block3_conv2 = Conv2D(256, (3, 3), name='block3_conv2', **self.conv_kwargs)(block3_conv1)
        block3_conv3 = Conv2D(256, (3, 3), name='block3_conv3', **self.conv_kwargs)(block3_conv2)
        block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(block3_conv3)

        block4_conv1 = Conv2D(512, (3, 3), name='block4_conv1', **self.conv_kwargs)(block3_pool)
        block4_conv2 = Conv2D(512, (3, 3), name='block4_conv2', **self.conv_kwargs)(block4_conv1)
        block4_conv3 = Conv2D(512, (3, 3), name='block4_conv3', **self.conv_kwargs)(block4_conv2)
        block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same')(block4_conv3)

        block5_conv1 = Conv2D(512, (3, 3), name='block5_conv1', **self.conv_kwargs)(block4_pool)
        block5_conv2 = Conv2D(512, (3, 3), name='block5_conv2', **self.conv_kwargs)(block5_conv1)
        block5_conv3 = Conv2D(512, (3, 3), name='block5_conv3', **self.conv_kwargs)(block5_conv2)
        block5_pool = MaxPooling2D((2, 2), strides=(1, 1), name='block5_pool', padding='same')(block5_conv3)

        norm_block4_conv3 = L2Normalization(name='block4_conv3_norm')(block4_conv3)

        return block5_pool, norm_block4_conv3

    @staticmethod
    def load_weights(model):
        VGG16_IMAGENET = ('https://github.com/fchollet/deep-learning-models/releases/'
                          'download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG16_IMAGENET,
                           cache_subdir='models', file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights, by_name=True)
        return model
