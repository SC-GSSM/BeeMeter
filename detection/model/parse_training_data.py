import pandas as pd
from pathlib import Path
import pickle
import re
from detection.model.registry import load_model
from detection.model.evaluation import Evaluator
from detection.dataset.dataset_generator import Dataset, DataGenerator


def fill_dict(dictionary, values):
    if len(dictionary) == 0:
        dictionary = {key: [value] for key, value in values.items()}
    else:
        for key, value in dictionary.items():
            if key in values.keys():
                dictionary[key].append(values[key])
            else:
                dictionary[key].append(None)

        for key, value in values.items():
            if key not in dictionary.keys():
                dictionary[key] = [None for _ in range(len(list(dictionary.values())[0]) - 1)] + [values[key]]
    return dictionary


def get_test_data():
    image_paths = ["/home/t9s9/PycharmProjects/BeeMeter/data/training/base_training_img",
                   "/home/t9s9/PycharmProjects/BeeMeter/data/training/gopro_training_img"]
    labels_paths = ["/home/t9s9/PycharmProjects/BeeMeter/data/training/base_labels.db",
                    "/home/t9s9/PycharmProjects/BeeMeter/data/training/gopro_labels.db"]
    datasets = []
    for img_p, lab_p in zip(image_paths, labels_paths):
        datasets.append(Dataset.from_sqlite(img_p, lab_p, verbose=True, limit=None))

    training_dataset = sum(datasets)
    validation_dataset = training_dataset.validation_split(ratio=0.15, seed=0)

    validation_dataset.load_images_in_memory(verbose=True)
    return validation_dataset


def evaluate(model, data):
    validation_generator = DataGenerator(dataset=data,
                                         encoder=model.box_handler,
                                         batch_size=2,
                                         shuffle=False,
                                         transformation=None,
                                         resize=(model.width, model.height))
    eval = Evaluator(ssd_model=model, data_generator=validation_generator)

    return eval.evaluate(iou_threshold=[0.5], save=False, recall_points=None, show=False)[0.5]


training_path = Path("/home/t9s9/PycharmProjects/BeeMeter/detection/training")
# training_path = Path("/media/t/Bachelor/inference")
config_name = "model_config.conf"
log_name = "_log.csv"

overview = dict()
test_data = get_test_data()
trainings = []
for path_obj in training_path.iterdir():
    if str(path_obj.name).startswith('MobileNet_') or str(path_obj.name).startswith('MobileNetV2_') or \
            str(path_obj.name).startswith('MobileNetVT'):
        trainings.append(path_obj.name)

for training_name in trainings:
    print(training_name)

    model = load_model(str(training_path / training_name / config_name), new=False,
                       weights_path=next((training_path / training_name / "checkpoints").iterdir()).absolute())
    #
    conf = pickle.load(open(training_path / training_name / config_name, 'rb'))
    log = pd.read_csv(training_path / training_name / (training_name + log_name))

    conf['trainable param'], conf['non trainable param'] = model.count_parameter(verbose=False)
    conf['total param'] = conf['trainable param'] + conf['non trainable param']
    conf['computational cost'] = model.computational_cost(verbose=False)
    conf['name'] = training_name
    conf['mAP'] = round(evaluate(model, test_data), 4)
    if 'f1' in log.columns:
        max_f1 = log[log['f1'] == max(log['f1'])]
        conf['f1'] = round(max_f1['f1'].iloc[0], 4)
        conf['epoch'] = int(max_f1['epoch'].iloc[0])
        overview = fill_dict(overview, conf)
    else:
        print("a")


df = pd.DataFrame.from_dict(overview)

df = pd.concat([df.drop(['base_net_kwargs'], axis=1), df['base_net_kwargs'].apply(pd.Series)], axis=1)
print(df)
df.to_csv(training_path / 'summary_MobileNets.csv')
