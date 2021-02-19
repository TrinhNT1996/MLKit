import os
import time
import sys
from datetime import datetime

import tensorflow as tf

from sources.utilities.file_utilities import FileUtilities
from sources.utilities.training_utilities import visualize_training_results
from tflite_model_maker import configs
from tflite_model_maker import image_classifier
from tflite_model_maker import ExportFormat
from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker import model_spec

assert tf.__version__.startswith('2')


def remove_unsupported_file(datasets_dir):
    for dir_name in os.listdir(datasets_dir):
        dir_path = os.path.join(datasets_dir, dir_name)
        if os.path.isfile(dir_path):
            os.remove(dir_path)
            print("Remove unsupported file: ", dir_path)
        elif os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                file_extension = file.split(".")[1]
                if not FileUtilities.is_supported_image(file_extension):
                    os.remove(file_path)
                    print("Remove unsupported file: ", file_path)


def get_datasets_info(datasets_dir):
    total_images = 0
    min_label = ''
    max_label = ''
    min_images = sys.maxsize
    max_images = 0
    all_labels = os.listdir(datasets_dir)
    all_labels.sort()
    for label in all_labels:
        label_dir = os.path.join(datasets_dir, label)
        if os.path.isdir(label_dir):
            label_image_count = len(os.listdir(label_dir))
            total_images += label_image_count
            if label_image_count < min_images:
                min_label = label
                min_images = label_image_count
            if label_image_count > max_images:
                max_images = label_image_count
                max_label = label
            print('{:<20}'.format(label), '--', '{:>5}'.format(label_image_count))
    print('---------------------')
    print('{:<15}'.format('Total labels:'), '{:,}'.format(len(all_labels)))
    print('{:<15}'.format('Total images:'), '{:,}'.format(total_images))
    print('{:<15}'.format('Average:'), '{:,}'.format(int(total_images / len(all_labels))), 'images/label')
    print('{:<15}'.format('Min:'), '{:,}'.format(min_images), ' - ', min_label)
    print('{:<15}'.format('Max:'), '{:,}'.format(max_images), ' - ', max_label)


def create_datasets(datasets_dir):
    print("Load datasets: ", datasets_dir)

    data = ImageClassifierDataLoader.from_folder(datasets_dir)
    train_data, rest_data = data.split(0.8)

    validation_data, test_data = rest_data.split(0.5)
    return train_data, validation_data, test_data


def train_model(datasets_dir, export_dir):
    remove_unsupported_file(datasets_dir)
    get_datasets_info(datasets_dir)
    train_data, validation_data, test_data = create_datasets(datasets_dir)

    print("Train the model")
    start_time = time.time()

    epochs = 20
    model = image_classifier.create(train_data,
                                    model_spec=model_spec.efficientnet_lite0_spec,
                                    validation_data=test_data,
                                    batch_size=32,  # None
                                    epochs=epochs,
                                    train_whole_model=None,  # None -
                                    dropout_rate=None,  # None -
                                    learning_rate=None,  # None
                                    momentum=None,  # None
                                    shuffle=True,  # False -
                                    use_augmentation=True,  # False -
                                    use_hub_library=True,  # True
                                    warmup_steps=None,  # None
                                    model_dir=None,  # None
                                    do_train=True)  # True

    print("EXPORT MODEL")

    date = datetime.today().strftime('%Y%m%d-%H%M')
    model_dir = os.path.join(export_dir, date)
    tflite_filename = 'model.tflite'

    config = configs.QuantizationConfig.create_full_integer_quantization(representative_data=test_data,
                                                                         is_integer_only=True)
    model.export(export_dir=model_dir,
                 tflite_filename=tflite_filename,
                 quantization_config=config)

    end_time = time.time()
    print("EXPORT MODEL SUCCESS AFTER: ", (end_time - start_time) / 60, "minutes")

    print('TEST MODEL')
    loss, accuracy = model.evaluate_tflite(os.path.join(model_dir, tflite_filename), test_data)
    print("Test model", "loss: ", loss, "accuracy: ", accuracy)
    visualize_training_results(model.history, epochs, index="1")
