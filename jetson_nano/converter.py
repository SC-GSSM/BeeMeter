import time
from pathlib import Path
import cv2

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


def keras2tf(h5_path, pd_save_path):
    """
    Convert a Keras model to a Tensorflow model.
    :param h5_path: Path to Keras model (.h5)
    :param pd_save_path: Path to save the Tensorflow model
    """
    model = tf.keras.models.load_model(h5_path)
    model.save(pd_save_path)


def convert(model_path, precision_mode='FP16'):
    """
    Convert a Tensorflow model to a TensorRT model. This specific for the device to run the model on.
    :param model_path:
    :param precision_mode:
    :return:
    """
    model_path = Path(model_path)
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 16))
    conversion_params = conversion_params._replace(precision_mode=precision_mode)
    #   conversion_params = conversion_params._replace(maximum_cached_engines=100)

    t1 = time.time()
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=str(model_path),
        conversion_params=conversion_params)
    converter.convert()

    print("Model converted in {0:.3f}".format(time.time() - t1))

    def my_input_fn():
        inp1 = tf.zeros(shape=(1, 200, 400, 3), dtype=tf.float32)
        yield [inp1]

    converter.build(input_fn=my_input_fn)
    print("Model build.")

    converter.save(str(model_path.parent / "tensorRT_{0}".format(precision_mode)))
    print("Model saved.")


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    convert("../resources/demo/tf_model", precision_mode='FP16')
