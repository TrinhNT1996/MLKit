import os
import sys
import tensorflow as tf

# Config

dataset_version = 'v6'

cwd = '/Users/trinhnt12/Projects/Sample/Others/MLKitExample/TensorFlow/ImageClassification'
dataset_dir = os.path.join(cwd, 'datasets', 'poc')
videos_dir = os.path.join(dataset_dir, 'videos', dataset_version)
frames_dir = os.path.join(dataset_dir, 'frames', dataset_version)
generated_dir = os.path.join(dataset_dir, 'generated', dataset_version)
models_dir = os.path.join(dataset_dir, 'models', dataset_version)


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


get_datasets_info(frames_dir)

print("Tensorflow: ", tf.__version__)
print("Tensorflow: ", tf.__file__)
print('Tensorflow GPU:', tf.test.gpu_device_name())

