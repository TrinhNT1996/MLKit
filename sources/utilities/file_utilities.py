import os


class FileUtilities:
    @staticmethod
    def create_dir_if_needed(dir_path):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    @staticmethod
    def is_supported_video(file):
        available_types = ["mov", "mp4"]
        file_extension = file.split('/')[-1].split('.')[1]
        return file_extension.lower() in available_types

    @staticmethod
    def is_supported_image(file_extension):
        available_types = ["jpg", "jpeg", "png"]
        return file_extension.lower() in available_types
