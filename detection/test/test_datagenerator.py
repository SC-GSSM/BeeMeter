from dataset.dataset_generator import Dataset, DataGenerator
from default_boxes.default_boxes import DefaultBoxHandler
from model.ssd import SSD
from model.ssd_config import SSDConfig
from utils.image_processing import Transformer, Resize, HorizontalFlip, VerticalFlip, Brightness, DataTypeConverter, \
    Contrast, TransformationChain
import cv2
from utils.box_tools import relative2absolute, draw_box
import numpy as np


def test_dataset_vis(config_path, batch=1):
    config = SSDConfig(config_path)
    dataset = Dataset(limit=None).from_sqlite(config.training_path, config.label_path)
    ssd = SSD(input_size=(*config.input_size, 3), boxes_per_cell_global=config.boxes_per_cell_global,
              boxes_per_cell=config.boxes_per_cell_local).create_network()

    default_boxes = DefaultBoxHandler(feature_map_sizes=ssd.feature_map_sizes(), **config.get_default_box_config())
    print("Feature map sizes:", ssd.feature_map_sizes(), "\nNumber boxes:", default_boxes.num_boxes)

    training_generator = DataGenerator(input_size=config.input_size, dataset=dataset,
                                       encoder=default_boxes, batch_size=batch,
                                       transformer=None, shuffle=False)
    for b in range(len(training_generator)):
        for image, labels in zip(*training_generator[b]):
            print("Background: {0}, Objects: {1}, Neutrals: {2}".format(labels[labels[:, 0] == 1].shape[0],
                                                                        labels[labels[:, 1] == 1].shape[0], labels[
                                                                            (labels[:, 0] == 0) & (
                                                                                    labels[:, 1] == 0)].shape[0]))
            exp_labels = np.expand_dims(labels, axis=0)
            decoded_labels = default_boxes.decode_default_boxes(exp_labels, confidence_threshold=0.5, debug=True)
            dft = relative2absolute(decoded_labels[:, 6:], box_format='corner', image_size=config.input_size)
            draw_box(image, decoded_labels[:, 2:6], color=(255, 0, 0), box_format='corner')
            draw_box(image, dft, color=(0, 0, 255), box_format='corner')
            cv2.imshow('Image', image)
            cv2.waitKey()
            cv2.destroyAllWindows()


def test_images_transformations(config_path, batch=1):
    config = SSDConfig(config_path)
    dataset = Dataset(limit=5).from_sqlite(config.training_path, config.label_path)
    # ssd = SSD(input_size=(*config.input_size, 3), boxes_per_cell_global=config.boxes_per_cell_global,
    #           boxes_per_cell=config.boxes_per_cell_local).create_network()
    #
    # default_boxes = DefaultBoxHandler(feature_map_sizes=ssd.feature_map_sizes(), **config.get_default_box_config())
    # print("Feature map sizes:", ssd.feature_map_sizes(), "\nNumber boxes:", default_boxes.num_boxes)

    training_generator = DataGenerator(dataset=dataset, encoder=None, batch_size=batch,
                                       local_transformation=TransformationChain(1),
                                       shuffle=False)

    for b in range(len(training_generator)):
        for i in range(batch):
            image, labels = training_generator[b]
            draw_box(image[i], labels[i], color=(0, 0, 255), box_format='corner')
            cv2.imshow('Image', image[i])
            cv2.waitKey()
            cv2.destroyAllWindows()


def test_label_encoding(config_path, batch=2):
    config = SSDConfig(config_path)
    dataset = Dataset(limit=None).from_sqlite(config.training_path, config.label_path)
    ssd = SSD(input_size=(*config.input_size, 3), boxes_per_cell_global=config.boxes_per_cell_global,
              boxes_per_cell=config.boxes_per_cell_local).create_network()

    default_boxes = DefaultBoxHandler(feature_map_sizes=ssd.feature_map_sizes(), **config.get_default_box_config())
    print("Feature map sizes:", ssd.feature_map_sizes(), "\nNumber boxes:", default_boxes.num_boxes)
    #default_boxes.fit(dataset.labels)
    #
    training_generator = DataGenerator(dataset=dataset, encoder=default_boxes, batch_size=batch, shuffle=False,
                                       global_transformation=[Resize(100, 100)],
                                       local_transformation=TransformationChain(0))

    np.set_printoptions(linewidth=300, threshold=np.inf)
    for b in range(len(training_generator)):
        image, labels = training_generator[b]
        decoded = default_boxes.decode_default_boxes(encoded_boxes=labels, confidence_threshold=0.5)
        for i in range(batch):
            draw_box(image[i], decoded[i][:, 1:], color=(0, 0, 255), box_format='corner', show=True)
    for i in training_generator.x:
        cv2.imshow('Image', i)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test_images_transformations('/home/t9s9/PycharmProjects/SSD_renewed/test/default_config.json')
