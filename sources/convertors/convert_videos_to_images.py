import os
from ffmpy import FFmpeg

from sources.utilities.file_utilities import FileUtilities

# ---input
# ------sub_dir_1
# ----------video_1
# ----------video_2
# ----------...
# ------sub_dir_2
# ----------video_3
# ----------video_4
# ----------...


def convert_videos_to_images(input_dir, output_dir):
    FileUtilities.create_dir_if_needed(output_dir)
    for sub_input_dir_name in os.listdir(input_dir):
        sub_input_dir = os.path.join(input_dir, sub_input_dir_name)
        if os.path.isdir(sub_input_dir):
            sub_output_dir = os.path.join(output_dir, sub_input_dir_name)
            FileUtilities.create_dir_if_needed(sub_output_dir)
            for file in os.listdir(sub_input_dir):
                convert_video_to_images(file=os.path.join(sub_input_dir, file), output_dir=sub_output_dir)


def convert_video_to_images(file, output_dir):
    if not FileUtilities.is_supported_video(file):
        return
    file_name = file.split('/')[-1].split('.')[0]
    outputs = os.path.join(output_dir, file_name + "_img%5d.jpg")
    ff = FFmpeg(inputs={file: None}, outputs={outputs: ['-vf', 'fps=1/0.1']})

    print(ff.cmd)
    ff.run()
