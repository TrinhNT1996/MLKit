import os
import imageio
from sources.utilities.augmentations_utilities import AugmentationsUtilities
from sources.utilities.file_utilities import FileUtilities


class ImageUtilities:

    @staticmethod
    def generate_image(file_path, output_dir, count):
        file_name = file_path.split("/")[-1].split(".")[0]
        file_extension = file_path.split("/")[-1].split(".")[1]
        if not FileUtilities.is_supported_image(file_extension):
            return
        FileUtilities.create_dir_if_needed(output_dir)

        image = imageio.imread(file_path)
        augmentations = AugmentationsUtilities.create_simple_augmentations()

        copy_path = "{}/{}.jpeg".format(output_dir, file_name)
        try:
            imageio.imwrite(copy_path, image)
        except:
            print("Error: ", file_path)
            os.remove(copy_path)
            return
        for i in range(count):
            image_aug = augmentations(image=image)
            export_file = "{}/{}gen{}.jpeg".format(output_dir, file_name, str(i))
            imageio.imwrite(export_file, image_aug)

    @staticmethod
    def generate_label(input_dir, output_dir, count):
        if not os.path.isdir(input_dir):
            return
        files = os.listdir(input_dir)
        for file_name in files:
            if os.path.isfile(os.path.join(input_dir, file_name)):
                ImageUtilities.generate_image(file_path=os.path.join(input_dir, file_name),
                                              output_dir=output_dir,
                                              count=count)
