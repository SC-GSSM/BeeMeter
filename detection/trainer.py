import gc
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from detection import BASE_TRAINING_PATH
from detection.dataset.dataset_generator import DataGenerator, Dataset
from detection.model.callbacks import EvaluationMetric
from detection.model.evaluation import Evaluator
from detection.model.loss_function import SSDLossBatch
from detection.model.registry import get_model
from detection.utils.box_tools import draw_box


class TrainScheduler:
    def __init__(self, names, confs, images, labels, batch_size=16, validation_split=0.1):
        self.this_trainer = None
        for i, to_train in enumerate(names):
            K.clear_session()  # Clear previous models from memory.
            gc.collect()
            if self.this_trainer is not None:
                del self.this_trainer.model

            if "dataset_config" in confs[i].keys():
                dataset_config = deepcopy(confs[i]['dataset_config'])
                del confs[i]['dataset_config']
            else:
                dataset_config = dict()

            self.this_trainer = Trainer(to_train)
            self.this_trainer.prepare_model(conf=confs[i])
            self.this_trainer.prepare_data(images, labels, validation_split=validation_split, batch_size=batch_size,
                                           limit=None, **dataset_config)
            self.this_trainer.start_training()


class Trainer:
    def __init__(self, name, base_dir=BASE_TRAINING_PATH):
        self.base_dir = Path(base_dir)
        self.name = name
        self.training_dir = self.base_dir / Path(name)

        self.model = None
        self.training_generator = None
        self.validation_generator = None

        self.create_workspace()

    def create_workspace(self):
        if not self.training_dir.is_dir():
            self.training_dir.mkdir(exist_ok=True)
            (self.training_dir / Path('checkpoints')).mkdir(exist_ok=True)

    def get_callbacks(self):
        checkpoint_name = self.name + 'weights.h5'
        checkpoint_path = self.training_dir / Path('checkpoints') / Path(checkpoint_name)
        checkpoint = ModelCheckpoint(str(checkpoint_path), monitor='val_loss', save_best_only=True,
                                     verbose=1, mode='auto')
        csv_logger = CSVLogger(filename=str(self.training_dir / Path(self.name + '_log.csv')),
                               separator=',', append=True)

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.0,
                                       patience=10,
                                       verbose=1,
                                       restore_best_weights=True)

        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.2,
                                                 patience=5,
                                                 verbose=1,
                                                 min_delta=0.001,
                                                 cooldown=0,
                                                 min_lr=0.00001)

        eval_metric = EvaluationMetric(generator=self.validation_generator,
                                       default_box_handler=self.model.box_handler)
        # the CSVLogger have to appear after the EvaluationMetric because the metric adding the data to the log.
        return [eval_metric, reduce_learning_rate, checkpoint, csv_logger, early_stopping]

    def plot_history(self, history, show=False):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        ax[0].plot(history.history['loss'], label='loss')
        ax[0].plot(history.history['val_loss'], label='val_loss')
        ax[0].legend()
        ax[0].set_ylabel('SSD Loss')
        ax[0].set_xlabel('Epochs')

        ax[1].plot(history.history['lr'], label="Learning Rate")
        ax[1].set_ylabel('Learning Rate')
        ax[1].set_xlabel('Epochs')
        ax[1].legend()

        ax[2].plot(history.history['precision'], label="Precision")
        ax[2].plot(history.history['recall'], label="Recall")
        ax[2].plot(history.history['f1'], label="F1")
        ax[2].set_ylabel('Value')
        ax[2].set_xlabel('Epochs')
        ax[2].legend()
        fig.savefig(str(self.training_dir / Path('loss_plot')))
        if show:
            plt.show()

    def create_datasets(self, image_paths, labels_paths, validation_split=0.15, limit=None):
        if not len(image_paths) == len(labels_paths):
            raise ValueError('For each directory with images there must be a database with labels. Actually '
                             'there are {0} directory and {1} databases!'.format(len(image_paths), len(labels_paths)))
        datasets = []
        for img_p, lab_p in zip(image_paths, labels_paths):
            datasets.append(Dataset.from_sqlite(img_p, lab_p, verbose=True, limit=limit))

        training_dataset = sum(datasets)
        validation_dataset = training_dataset.validation_split(ratio=validation_split, seed=0)

        training_dataset.load_images_in_memory(verbose=True)
        validation_dataset.load_images_in_memory(verbose=True)

        return training_dataset, validation_dataset

    def prepare_model(self, conf):
        # creating the network
        ssd = get_model(conf['model_name'])()
        self.model = ssd.create_model(**conf)

        # initialise the loss function
        loss = SSDLossBatch(alpha=1.0, ratio=4)
        # initialise the optimizer
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
        # compile model with the loss function and the optimizer
        self.model.model.compile(optimizer=optimizer, loss=loss)

    def prepare_data(self, image_paths, labels_paths, validation_split=0.15, batch_size=8, limit=None, aug=0, std=True):
        print("Width: {0} Height: {1}".format(self.model.width, self.model.height))
        # create the datasets
        training_dataset, validation_dataset = self.create_datasets(image_paths=image_paths,
                                                                    labels_paths=labels_paths,
                                                                    validation_split=validation_split, limit=limit)

        bbox_params = A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.0, label_fields=['box_class'])
        transform_soft = A.Compose([
            A.Sequential([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1, p=0.5)
            ]),
        ], bbox_params)

        transform_hard = A.Compose([
            A.Sequential([
                A.RandomResizedCrop(height=self.model.height, width=self.model.width, scale=(0.5, 0.9), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.2, p=0.5)
            ]),
        ], bbox_params)

        transform_blur = A.Compose([
            A.Sequential([
                A.RandomResizedCrop(height=self.model.height, width=self.model.width, scale=(0.5, 0.9), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2, p=0.5),
                A.GaussianBlur(p=0.5)
            ]),
        ], bbox_params)

        transform_gray = A.Compose([
            A.Sequential([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1, p=0.5),
                A.ToGray(always_apply=True, p=1)
            ]),
        ], bbox_params)

        augmentation = [transform_soft, transform_hard, transform_blur, transform_gray][aug]
        print(augmentation)
        self.training_generator = DataGenerator(dataset=training_dataset,
                                                encoder=self.model.box_handler,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                transformation=augmentation,
                                                std=std,
                                                resize=(self.model.width, self.model.height))
        self.validation_generator = DataGenerator(dataset=validation_dataset,
                                                  encoder=self.model.box_handler,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  transformation=None,
                                                  resize=(self.model.width, self.model.height))

        # initialize the data generators and do global preprocessing
        # self.training_generator = DataGenerator(dataset=training_dataset,
        #                                         encoder=self.model.box_handler,
        #                                         batch_size=batch_size,
        #                                         shuffle=True,
        #                                         std=True,
        #                                         local_transformation=TransformationChain(1),
        #                                         global_transformation=[Resize(output_width=self.model.width,
        #                                                                       output_height=self.model.height)])
        #
        # self.validation_generator = DataGenerator(dataset=validation_dataset,
        #                                           encoder=self.model.box_handler,
        #                                           batch_size=batch_size,
        #                                           shuffle=False,
        #                                           global_transformation=[Resize(output_width=self.model.width,
        #                                                                         output_height=self.model.height)])

        self.model.to_config(self.training_dir)

    def show_data(self):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(22, 22))
        ax = ax.flatten()
        images, labels = self.training_generator[0]
        decoded = self.model.box_handler.faster_decode_default_boxes(labels.astype("float32"))
        for i in range(len(ax)):
            draw_box(image=images[i], boxes=decoded[i][..., 1:], axis=ax[i], engine='plt')
        plt.tight_layout()
        plt.show()

    def start_training(self):
        # start training
        start_time = datetime.now()
        print('Starting training {0} at {1}'.format(self.name, start_time.strftime("%d/%m/%Y %H:%M:%S")))

        history = self.model.model.fit(x=self.training_generator,
                                       epochs=1000,
                                       validation_data=self.validation_generator,
                                       callbacks=self.get_callbacks())
        print(history.history)
        end_time = datetime.now()
        # save the training histories plot
        self.plot_history(history=history, show=True)
        # evaluate
        eval = Evaluator(ssd_model=self.model, data_generator=self.validation_generator)
        eval.evaluate(iou_threshold=[0.3, 0.5, 0.7], save=True, path=self.training_dir)
        duration = end_time - start_time
        print('Ended training {0} at {1}. Training duration {2}'.format(self.name,
                                                                        end_time.strftime("%d/%m/%Y %H:%M:%S"),
                                                                        duration.total_seconds()))


if __name__ == '__main__':
    K.clear_session()  # Clear previous models from memory.

    basic_conf = dict(
        model_name='SSDLite',
        base_net="MyMobileNetV2",
        input_shape=(400, 200, 3),
        aspect_ratios_global=[0.529, 0.625, 1.056, 1.198],
        scale_min=0.05,
        scale_max=0.3,
        base_net_kwargs={"alpha": 1.0, 'predictors': ['expand11', 'expand12', 'expand16']},
        keep_top=4,
        activation=tf.nn.relu6,
        kernel_initializer='he_normal',
        l2_pen=0.0005,
        dataset_config={"std": False}
    )

    images = ["/home/t9s9/PycharmProjects/BeeMeter/data/training/base_training_img",
              "/home/t9s9/PycharmProjects/BeeMeter/data/training/gopro_training_img"]
    labels = ["/home/t9s9/PycharmProjects/BeeMeter/data/training/base_labels.db",
              "/home/t9s9/PycharmProjects/BeeMeter/data/training/gopro_labels.db"]

    names = ["Training_name"]

    TrainScheduler(names=names, confs=[basic_conf], images=images, labels=labels, batch_size=4, validation_split=0.15)
