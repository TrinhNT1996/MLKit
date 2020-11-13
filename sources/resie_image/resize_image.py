import os
from sources.utilities.file_utilities import FileUtilities
from PIL import Image


# ---input
# ------sub_dir_1
# ----------video_1
# ----------video_2
# ----------...
# ------sub_dir_2
# ----------video_3
# ----------video_4
# ----------...


def resize_images(input_dir, output_dir, max_size):
    FileUtilities.create_dir_if_needed(output_dir)
    for sub_input_dir_name in os.listdir(input_dir):
        sub_input_dir = os.path.join(input_dir, sub_input_dir_name)
        if os.path.isdir(sub_input_dir):
            sub_output_dir = os.path.join(output_dir, sub_input_dir_name)
            FileUtilities.create_dir_if_needed(sub_output_dir)
            for file in os.listdir(sub_input_dir):
                resize_image(file=os.path.join(sub_input_dir, file),
                             output_dir=sub_output_dir,
                             max_size=max_size)


def resize_image(file, output_dir, max_size):
    file_name = file.split("/")[-1].split(".")[0]
    file_extension = file.split("/")[-1].split(".")[1]
    if not FileUtilities.is_supported_image(file_extension):
        return
    image = Image.open(file)
    width, height = image.size
    ratio = max(width, height) / max_size
    if ratio > 1.0:
        resized_image = image.resize((round(width / ratio), round(height / ratio)))
        quantity = 0.9
    else:
        resized_image = image
        quantity = 1.0
    FileUtilities.create_dir_if_needed(output_dir)
    resized_image.save(os.path.join(output_dir, file_name + "." + file_extension), quantity=quantity)
