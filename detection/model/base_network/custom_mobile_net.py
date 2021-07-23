import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, Activation
from detection.model.registry import register_base_net
from detection.model.base_network.mobile_net import MobileNet, MobileNetV2


@register_base_net
class MyMobileNet(MobileNet):
    def __init__(self, alpha=1.0, rho=1, l2_pen=0.0005, kernel_initializer='he_normal', predictors=['conv11'],
                 **kwargs):
        super(MyMobileNet, self).__init__(alpha=alpha, rho=rho, l2_pen=l2_pen, predictors=predictors,
                                          kernel_initializer=kernel_initializer, **kwargs)

    def __call__(self, input_layer):
        conv0 = self._conv_block(32, strides=(2, 2))(input_layer)
        conv1 = self._dw_block(64)(conv0)
        conv2 = self._dw_block(128)(conv1)
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


@register_base_net
class MyMobileNetV2(MobileNetV2):
    def __init__(self, alpha=1.0, l2_pen=0.0005, activation=tf.nn.relu6, kernel_initializer='he_normal',
                 predictors=['expand13'], **kwargs):
        super(MyMobileNetV2, self).__init__(alpha=alpha, l2_pen=l2_pen, activation=activation, predictors=predictors,
                                            kernel_initializer=kernel_initializer, **kwargs)

    def __call__(self, input_layer):
        first_block_filters = self._make_divisible(32 * self.alpha, 8)
        x = ZeroPadding2D(padding=self.correct_pad(input_layer, 3), name='Conv1_pad')(input_layer)
        # 1: 400x200
        x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='valid', use_bias=False, name='Conv1',
                   **self.conv_kwargs)(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
        x = Activation(self.act, name='Conv1_relu')(x)
        # 2: 200x100
        expand0, layer0, x = self._inverted_res_block(x, filters=16, alpha=self.alpha, stride=1, expansion=1,
                                                      block_id=0)
        # 3-4:200x100
        expand1, _, layer1 = self._inverted_res_block(layer0, filters=24, alpha=self.alpha, stride=2, expansion=6,
                                                      block_id=1)
        expand2, _, layer2 = self._inverted_res_block(layer1, filters=24, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=2)
        # 5-7: 100x50
        expand3, _, layer3 = self._inverted_res_block(layer2, filters=32, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=3)
        expand4, _, layer4 = self._inverted_res_block(layer3, filters=32, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=4)
        expand5, _, layer5 = self._inverted_res_block(layer4, filters=32, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=5)
        # 8-11: 100x50
        expand6, _, layer6 = self._inverted_res_block(layer5, filters=64, alpha=self.alpha, stride=2, expansion=6,
                                                      block_id=6)
        expand7, _, layer7 = self._inverted_res_block(layer6, filters=64, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=7)
        expand8, _, layer8 = self._inverted_res_block(layer7, filters=64, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=8)
        expand9, _, layer9 = self._inverted_res_block(layer8, filters=64, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=9)
        # 12-14: 50x25
        expand10, _, layer10 = self._inverted_res_block(layer9, filters=96, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=10)
        expand11, _, layer11 = self._inverted_res_block(layer10, filters=96, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=11)
        expand12, _, layer12 = self._inverted_res_block(layer11, filters=96, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=12)
        # 15-17: 50x25
        expand13, _, layer13 = self._inverted_res_block(layer12, filters=160, alpha=self.alpha, stride=2, expansion=6,
                                                        block_id=13)
        expand14, _, layer14 = self._inverted_res_block(layer13, filters=160, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=14)
        expand15, _, layer15 = self._inverted_res_block(layer14, filters=160, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=15)
        # 18: 25x13
        expand16, _, layer16 = self._inverted_res_block(layer15, filters=320, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=16)

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if self.alpha > 1.0:
            last_block_filters = self._make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280
        # 19: 15x13
        x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1', **self.conv_kwargs)(layer16)
        x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = Activation(self.act, name='out_relu')(x)

        predictor_layers = []
        for layer in self.predictors:
            predictor_layers.append(locals()[layer])

        return x, predictor_layers


@register_base_net
class MyMobileNetVT(MobileNetV2):
    def __init__(self, alpha=1.0, l2_pen=0.0005, activation=tf.nn.relu6, kernel_initializer='he_normal',
                 predictors=['expand13'], **kwargs):
        super(MyMobileNetVT, self).__init__(alpha=alpha, l2_pen=l2_pen, activation=activation, predictors=predictors,
                                            kernel_initializer=kernel_initializer, **kwargs)

    def __call__(self, input_layer):
        first_block_filters = self._make_divisible(32 * self.alpha, 8)
        x = ZeroPadding2D(padding=self.correct_pad(input_layer, 3), name='Conv1_pad')(input_layer)
        x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='valid', use_bias=False, name='Conv1',
                   **self.conv_kwargs)(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
        x = Activation(self.act, name='Conv1_relu')(x)

        expand0, layer0, x = self._inverted_res_block(x, filters=16, alpha=self.alpha, stride=1, expansion=1,
                                                      block_id=0)

        expand1, _, layer1 = self._inverted_res_block(layer0, filters=24, alpha=self.alpha, stride=2, expansion=6,
                                                      block_id=1)
        expand2, _, layer2 = self._inverted_res_block(layer1, filters=24, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=2)
        expand3, _, layer3 = self._inverted_res_block(layer2, filters=32, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=3)
        expand4, _, layer4 = self._inverted_res_block(layer3, filters=32, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=4)
        expand5, _, layer5 = self._inverted_res_block(layer4, filters=32, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=5)

        expand6, _, layer6 = self._inverted_res_block(layer5, filters=64, alpha=self.alpha, stride=2, expansion=6,
                                                      block_id=6)
        expand7, _, layer7 = self._inverted_res_block(layer6, filters=64, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=7)
        expand8, _, layer8 = self._inverted_res_block(layer7, filters=64, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=8)
        expand9, _, layer9 = self._inverted_res_block(layer8, filters=64, alpha=self.alpha, stride=1, expansion=6,
                                                      block_id=9)
        expand10, _, layer10 = self._inverted_res_block(layer9, filters=96, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=10)
        expand11, _, layer11 = self._inverted_res_block(layer10, filters=96, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=11)
        expand12, _, layer12 = self._inverted_res_block(layer11, filters=96, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=12)

        expand13, _, layer13 = self._inverted_res_block(layer12, filters=96, alpha=self.alpha, stride=2, expansion=6,
                                                        block_id=13)
        expand14, _, layer14 = self._inverted_res_block(layer13, filters=96, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=14)
        expand15, _, layer15 = self._inverted_res_block(layer14, filters=160, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=15)
        expand16, _, layer16 = self._inverted_res_block(layer15, filters=160, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=16)
        expand17, _, layer17 = self._inverted_res_block(layer16, filters=160, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=17)
        expand18, _, layer18 = self._inverted_res_block(layer17, filters=220, alpha=self.alpha, stride=1, expansion=6,
                                                        block_id=18)

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if self.alpha > 1.0:
            last_block_filters = self._make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280

        x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1', **self.conv_kwargs)(layer18)
        x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = Activation(self.act, name='out_relu')(x)

        predictor_layers = []
        for layer in self.predictors:
            predictor_layers.append(locals()[layer])

        return x, predictor_layers
