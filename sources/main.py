import os

from sources.convertors.convert_videos_to_images import convert_videos_to_images
from sources.image_generation import ImageGeneration
from sources.resie_image.resize_image import resize_images
from sources.training.tflite_maker import train_model

# Config

dataset_version = 'v6'

dataset_dir = os.path.join(os.getcwd(), 'datasets', 'poc')
videos_dir = os.path.join(dataset_dir, 'videos', dataset_version)
frames_dir = os.path.join(dataset_dir, 'frames', dataset_version)
resized_dir = os.path.join(dataset_dir, 'resized', dataset_version)
generated_dir = os.path.join(dataset_dir, 'generated', dataset_version)
models_dir = os.path.join(dataset_dir, 'models', dataset_version)

if dataset_version == 'test':
    generated_count = 50
else:
    generated_count = 5


def start():
    print("\n\n\n\n")
    print("**************************************   ImageClassification   **************************************")
    print("\n")

    features = [
        "Convert video to image",
        "Resize images",
        "Generate images",
        "Train model without gen ds",
        "Train model with gen ds"
    ]
    for idx, val in enumerate(features):
        print(idx + 1, ". ", val)
    selected_feature = input("Select feature: ")

    if selected_feature == "1":
        convert_videos_to_images(videos_dir, frames_dir)
    if selected_feature == "2":
        resize_images(frames_dir, resized_dir, max_size=2048)
    elif selected_feature == "3":
        generation = ImageGeneration(source_dir=resized_dir,
                                     generated_dir=generated_dir,
                                     count=generated_count)
        generation.generate()
    elif selected_feature == "4":
        train_model(frames_dir, models_dir)
    elif selected_feature == "5":
        train_model(generated_dir, models_dir)

    _continue = input("Do you want continue (y/n): ")
    if _continue == "y":
        start()


start()
