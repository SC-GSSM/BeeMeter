import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

from detection.model.ssd import SSD


def create_representative_data_gen():
    width = ssd.input_shape[2]
    height = ssd.input_shape[1]
    images = []
    for path in tqdm(list(Path("/home/t9s9/PycharmProjects/BeeMeter/data/training/base_training_img").iterdir())[:1000]):
        original_image = cv2.imread(str(path))
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, dsize=(width, height))
        images.append(resized_image.astype("float32") / 255.0)

    def representative_data_gen():
        for image in images:
            yield [np.expand_dims(image, axis=0)]

    return representative_data_gen


def optimize_model(model, save_path=None, integer=False, float16=False):
    """
    Full post-training integer quantization
    :return:
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if float16:
        converter.target_spec.supported_types = [tf.float16]

    if integer:
        # This sets the representative dataset for quantization
        converter.representative_dataset = create_representative_data_gen()

        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for
        # clarity.
        converter.target_spec.supported_types = [tf.int8]

        # Set the input and output tensors to uint8
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8

    # converter.allow_custom_ops = True

    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if save_path is not None:
        if save_path.endswith('.tflite'):
            save_path = Path(save_path)
        else:
            save_path = Path(save_path) / Path("model.tflite")

        save_path.write_bytes(tflite_model)

    print(input_details[0]['dtype'])
    print(output_details[0]['dtype'])
    print(input_details[0]["quantization"])
    return tflite_model


if __name__ == '__main__':
    #m = SSD.from_training("ColabMobileNet1SmallStdBatch")
    #ssd = m.model
    ssd = tf.keras.models.load_model("/home/t9s9/PycharmProjects/BeeMeter/detection/training/VerySmallMobileNetV2/checkpoints/VerySmallMobileNetV2-64_loss-2.6521_val_loss-3.0227.h5", compile=False, custom_objects={'relu6': tf.nn.relu6})
    #tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8

    optimize_model(model=ssd, save_path="", integer=False, float16=True)
