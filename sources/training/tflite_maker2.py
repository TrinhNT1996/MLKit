import tensorflow as tf

from sources.utilities.file_utilities import visualize_training_results

assert tf.__version__.startswith('2')
from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader

print("Tensorflow: ", tf.__version__)
print("-------------------------------------------------------------------------------------------------------------")
print("Load datasets")

image_path = tf.keras.utils.get_file('flower_photos',
                                     'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                     untar=True)
print("Image_path", image_path)
image_path = "datasets/flowers"
data = ImageClassifierDataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

print("-------------------------------------------------------------------------------------------------------------")
print("Train the model")

epochs = 10
model = image_classifier.create(train_data,
                                model_spec='efficientnet_lite0',
                                validation_data=test_data,
                                batch_size=None,  # None
                                epochs=epochs,
                                train_whole_model=True,  # None -
                                dropout_rate=None,  # None -
                                learning_rate=None,  # None
                                momentum=None,  # None
                                shuffle=True,  # False -
                                use_augmentation=True,  # False -
                                use_hub_library=True,  # True
                                warmup_steps=None,  # None
                                model_dir=None,  # None
                                do_train=True)  # True

print("-------------------------------------------------------------------------------------------------------------")
print("Visualize training results")

visualize_training_results(model.history, epochs, index="2")

model.export(export_dir='../..')
