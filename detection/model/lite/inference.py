import tensorflow as tf
import time
import cv2
import numpy as np


class LiteModel:
    def __init__(self, model_path=None, model_content=None):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, model_content=model_content)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("Init lite model.")

    def predict(self, x):
        """

        :param x: Input of the network of shape [1, height, width, channel] or [height, width, channel]
        :return: output of the network of shape [1, #boxes, #classes + 4]
        """
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], x)

        self.interpreter.invoke()

        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

        return prediction

    @property
    def width(self):
        return self.input_details[0]['shape'][2]

    @property
    def height(self):
        return self.input_details[0]['shape'][1]

    @property
    def channel(self):
        return self.input_details[0]['shape'][3]

    @property
    def input_type(self):
        return self.input_details[0]['dtype']


if __name__ == '__main__':
    from pathlib import Path
    from detection.model.registry import load_model
    from detection.utils.box_tools import draw_box

    d = load_model(config_path="/home/t9s9/PycharmProjects/BeeMeter/detection/training/VerySmallMobileNetV2/model_config.conf", weights_path="/home/t9s9/PycharmProjects/BeeMeter/detection/training/VerySmallMobileNetV2/checkpoints/VerySmallMobileNetV2-64_loss-2.6521_val_loss-3.0227.h5")

    model = LiteModel(model_path="trained_int_model.tflite")
    images = []
    for path in list(Path("/home/t9s9/PycharmProjects/BeeMeter/data/training/base_training_img").iterdir())[2:3]:
        original_image = cv2.imread(str(path))
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, dsize=(model.width, model.height))
        single_batch_image = np.expand_dims(resized_image, axis=0)
        images.append(single_batch_image.astype("float32"))

    times = []
    for i in images:
        start_time = time.time()
        output_data = model.predict(i)
        np.set_printoptions(threshold=np.inf, linewidth=200)
        print(output_data)
        r = d.box_handler.faster_decode_default_boxes(output_data)
        print(r)
        times.append(time.time() - start_time)
        draw_box(image=np.squeeze(i).astype('uint8'), boxes=r[0][:, 1:], out='show')

    print("Time: {0:.2f}s".format(np.mean(times)))
