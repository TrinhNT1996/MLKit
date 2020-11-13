import os
from sources.training.tflite_maker import train_model

# Config

dataset_version = 'v5'
isTest = True

cwd = '/Users/trinhnt12/Projects/Sample/Others/MLKitExample/TensorFlow/ImageClassification'
dataset_dir = os.path.join(cwd, 'datasets', 'poc')
videos_dir = os.path.join(dataset_dir, 'videos', dataset_version)
frames_dir = os.path.join(dataset_dir, 'frames', dataset_version)
generated_dir = os.path.join(dataset_dir, 'generated', dataset_version)
models_dir = os.path.join(dataset_dir, 'models', dataset_version)

train_model(frames_dir, models_dir)
