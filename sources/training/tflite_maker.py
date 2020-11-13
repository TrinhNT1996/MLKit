import os
import time
from datetime import datetime

import tensorflow as tf

from sources.utilities.file_utilities import FileUtilities
from sources.utilities.training_utilities import visualize_training_results
from tflite_model_maker import image_classifier
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


def create_datasets(datasets_dir):
    print("Load datasets: ", datasets_dir)

    remove_unsupported_file(datasets_dir)

    data = ImageClassifierDataLoader.from_folder(datasets_dir)
    train_data, rest_data = data.split(0.8)

    validation_data, test_data = rest_data.split(0.5)
    return train_data, validation_data, test_data


def train_model(datasets_dir, export_dir):
    start_time = time.time()

    print("Tensorflow: ", tf.__version__)

    train_data, validation_data, test_data = create_datasets(datasets_dir)

    print("Train the model")

    epochs = 10
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

    print("Visualize training results")

    visualize_training_results(model.history, epochs, index="1")

    print("EXPORT MODEL")

    date = datetime.today().strftime('%Y%m%d-%H%M')
    model.export(export_dir=os.path.join(export_dir, date))

    end_time = time.time()
    print("EXPORT MODEL SUCCESS AFTER: ", (end_time - start_time) / 60, "minutes")

    loss, accuracy = model.evaluate_tflite(os.path.join(export_dir, date, 'model.tflite'), test_data)
    print("Test model", "loss: ", loss, "accuracy: ", accuracy)
