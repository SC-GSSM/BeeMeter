import os
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
try:
    import albumentations as alb
except ImportError:
    pass

from detection.dataset.database_connector import Database
from detection.default_boxes.default_boxes import DefaultBoxHandler
from detection.utils.validation import check_instance, check_number, check_bool


class DataGenerator(Sequence):
    def __init__(self, dataset, encoder=None, batch_size=8, shuffle=True, transformation=None, std=False, resize=None):
        """
        Class that generates batches of images and boxes for training purpose.
        :param dataset: A class for loading the images and labels.
        :param encoder: A default box handler for encoding the labels for SSD training.
        :param batch_size: Size of the generated batches.
        :param shuffle: The data can be shuffled after one epoch.
        :param transformation: Transformation chain, given by the module albumentations.
        :param std: The bounding box can be normalized by the given training data. See DefaultBoxHandler for details.
        :param resize: Resize all images and boxes before the training to save computation time. Needs to be a tuple of
        (target_width, target_height)
        """
        self.dataset = check_instance(dataset, Dataset)
        self.batch_size = check_number(batch_size, cls=int, value_min=1)
        self.indices = np.arange(len(self.dataset))
        self.encoder = check_instance(encoder, DefaultBoxHandler, allow_none=True)
        self.shuffle = check_bool(shuffle)
        self.transformation = check_instance(transformation, alb.Compose, allow_none=True)
        self.std = check_bool(std)

        if resize is not None:
            self.x, self.y = [], []
            resizer = alb.Compose([alb.Resize(height=resize[1], width=resize[0], always_apply=True)],
                                  bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['box_class']))
            for i in range(len(self.dataset.images)):
                transformed = resizer(image=self.dataset.images[i], bboxes=self.dataset.labels[i],
                                      box_class=[1] * len(self.dataset.labels[i]))
                self.x.append(transformed['image'])
                self.y.append(np.array(transformed['bboxes']))
        else:
            self.x = self.dataset.images
            self.y = self.dataset.labels

        if self.std:
            self.encoder.fit(self.y)

    def __len__(self):
        """ Number of batches in the Sequence """
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """
        Gets the batch at position index.
        :param index: batch position index
        :return: a batch: tuple (images, labels)
        """
        batch_x = []
        batch_y = []
        this_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        for i in this_indices:
            # Transform images here
            if self.transformation is not None:
                transformed = self.transformation(image=self.x[i], bboxes=deepcopy(self.y[i]),
                                                  box_class=[0] * len(self.y[i]))
                batch_x.append(transformed['image'])
                batch_y.append(np.array(transformed['bboxes']))
            else:
                batch_x.append(self.x[i])
                batch_y.append(self.y[i])

        batch_x = np.array(batch_x, copy=True)

        if self.encoder is not None:
            batch_y = self.encoder.encode_default_boxes(batch_y, iou_threshold=0.5, verbose=False,
                                                        cut_default_boxes=True)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            # in-place shuffle the array of indices
            np.random.shuffle(self.indices)


class Dataset:
    """
    Class that represents a Dataset containing images and labels.
    """
    def __init__(self, filenames, image_ids, labels, images=None, nr_labels=None, dataset_size=None, image_size=None):
        # list of the paths to the images
        self.filenames = filenames
        # list of image id
        self.image_ids = image_ids
        # list of labels: at position i are the boxes from the image with the path at position filenames[i]
        self.labels = labels
        # if images are loaded there are located here
        self.images = images

        self.nr_labels = nr_labels
        self.dataset_size = dataset_size

        self.img_width = image_size[0]
        self.img_height = image_size[1]
        self.channel = image_size[2]

    def __len__(self):
        """ Returns the number of instances in the current dataset. """
        return self.dataset_size

    def __add__(self, other):
        """
        This methode allows to add two datasets into one new dataset, combining all images and labels.
        :param other: Second Dataset object to add.
        :return: Dataset object.
        """
        if isinstance(other, Dataset):
            if self.img_width == other.img_width and self.img_height == other.img_height and self.channel == other.channel:
                new_filenames = np.concatenate([self.filenames, other.filenames])
                new_image_ids = np.concatenate([self.image_ids, other.image_ids])
                new_labels = np.concatenate([self.labels, other.labels])
                new_dataset_size = self.dataset_size + other.dataset_size
                new_nr_labels = self.nr_labels + other.nr_labels
                if self.images is not None and other.images is not None:
                    new_images = np.concatenate([self.images, other.images])
                else:
                    new_images = None
            else:
                raise ValueError('The size of the images in the dataset is not equal!')
        else:
            raise ValueError('Only instances of class Dataset can be added to a Dataset.')
        return Dataset(new_filenames, new_image_ids, new_labels, nr_labels=new_nr_labels, dataset_size=new_dataset_size,
                       image_size=(self.img_width, self.img_height, self.channel), images=new_images)

    def __radd__(self, other):
        return self

    def load_images_in_memory(self, verbose=True):
        """
        Load images in memory to speed up training.
        :param verbose: Show progress bar.
        """
        images = []
        if verbose:
            it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
        else:
            it = self.filenames
        for filename in it:
            img = cv2.imread(filename)  # format: BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
        self.images = np.array(images)

    def save(self, path):
        """
        Saves current dataset as new file. Writes all images into ine directory and creates a new database file.
        :param path: output path
        """
        if self.images is None:
            self.load_images_in_memory()

        path = Path(path)
        image_path = (path / 'images')
        path.mkdir(exist_ok=True)
        image_path.mkdir(exist_ok=True)

        db = Database(str(path / "labels.db"))

        for filename, img, labels in zip(self.filenames, self.images, self.labels):
            this_path = str(image_path / Path(filename).name)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(this_path, img_bgr)
            img_id = db.add_image(str(Path(filename).name))
            for x1, y1, x2, y2 in labels:
                db.add_label(img_id[0][0], x1, y1, x2, y2)

    def memory_space(self, unit='mb'):
        """
        Calculates the memory consumption of the images in the dataset.
        :param unit: one of: 'b': Byte, 'kb': Kilobyte, 'mb': Megabyte, 'gb': Gigabyte
        :return: Number of bytes to store all images in the memory.
        """
        if self.images is None:
            return 0
        else:
            convert = {'b': 1.0, 'kb': 10 ** 3, 'mb': 10 ** 6, 'gb': 10 ** 9}
            return (self.images.size * self.images.itemsize) / convert[unit]

    def _update_stats(self):
        """ Recalculate dataset numbers. """
        self.dataset_size = len(self.filenames)
        self.img_height, self.img_width, self.channel = cv2.imread(self.filenames[0]).shape
        self.nr_labels = sum(len(i) for i in self.labels)

    def validation_split(self, ratio=0.1, seed=0):
        """
        Takes a part if the current dataset and creates a new dataset for validation. The current dataset will be
        modified.
        :param ratio: Share of the current date set used for the new one.
        :param seed: One can pass a seed to guarantee reproducibility.
        :return: New validation dataset.
        """
        random_gen = np.random.RandomState(seed=seed)
        nr_valid = int(np.ceil(len(self) * ratio))
        # Selects a number of random indices. The parameter replace ensures that no element is taken twice
        idx = random_gen.choice(np.arange(len(self)), size=nr_valid, replace=False)

        valid_filenames = self.filenames[idx]
        self.filenames = np.delete(self.filenames, idx)
        valid_image_ids = self.image_ids[idx]
        self.image_ids = np.delete(self.image_ids, idx)
        valid_labels = self.labels[idx]
        self.labels = np.delete(self.labels, idx)

        self._update_stats()

        return Dataset(valid_filenames, valid_image_ids, valid_labels, nr_labels=sum(len(i) for i in valid_labels),
                       dataset_size=nr_valid, image_size=(self.img_width, self.img_height, self.channel))

    @classmethod
    def from_sqlite(cls, images_dirs, path, verbose=False, limit=None):
        """
        Create Dataset from SQLite database, containing image names and labels.
        :param images_dirs: Path to image storage location.
        :param path: Path to SQLite database file.
        :param verbose: Show progress bar.
        :param limit: Take the first 'limit' images from the dataset. This is useful for testing and debugging.
        :return: Dataset object.
        """
        if limit:
            if limit < 2:
                raise ValueError('Limit must me 2 or larger!')
        db = Database(path)

        filenames = []
        image_ids = []
        labels = []
        nr_labels = 0

        files = os.listdir(images_dirs)
        if limit:
            files = files[:limit]
        if verbose:
            it = tqdm(files, desc="Processing annotations from database", file=sys.stdout)
        else:
            it = files

        for filename in it:
            valid, img_id = db.image_id_by_filename(filename)
            if valid:
                filenames.append(os.path.join(images_dirs, filename))
                image_ids.append(img_id)
                boxes = []
                for annotation in db.label_coords_by_image_id(img_id):
                    boxes.append([*annotation])
                    nr_labels += 1
                labels.append(np.array(boxes))

        # set the dataset_size to the number of images
        dataset_size = len(filenames)
        # get the image size by open the first image. All images need to be the same size.
        img_height, img_width, channel = cv2.imread(filenames[0]).shape

        return cls(np.array(filenames), np.array(image_ids), np.array(labels), nr_labels=nr_labels,
                   dataset_size=dataset_size,
                   image_size=(img_width, img_height, channel), images=None)
