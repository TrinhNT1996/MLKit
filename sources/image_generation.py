import os
import time

from sources.utilities.file_utilities import FileUtilities
from sources.utilities.image_utilities import ImageUtilities
from sources.utilities.utilities import complement_elements


class ImageGeneration:
    source_dir = ""
    generated_dir = ""
    count = 0

    def __init__(self, source_dir, generated_dir, count):
        self.source_dir = source_dir
        self.generated_dir = generated_dir
        self.count = count
        FileUtilities.create_dir_if_needed(generated_dir)

    def gen(self, label):
        ImageUtilities.generate_label(input_dir=os.path.join(self.source_dir, label),
                                      output_dir=os.path.join(self.generated_dir, label),
                                      count=self.count)

    def generate_labels(self):
        input_str = input("Input labels (separate by space): ")
        labels = input_str.split(" ")
        source_labels = os.listdir(self.source_dir)
        for label in labels:
            if label in source_labels:
                self.gen(label=label)

    def generate_new_labels(self):
        source_labels = os.listdir(self.source_dir)
        generated_labels = os.listdir(self.generated_dir)
        gen_labels = complement_elements(source_labels, generated_labels)
        print("Generate images for labels: ", gen_labels)
        for label in gen_labels:
            self.gen(label=label)

    def generate_all_labels(self):
        if not os.path.isdir(self.source_dir):
            return
        label_dirs = os.listdir(self.source_dir)
        for dir_name in label_dirs:
            self.gen(label=dir_name)

    def generate(self):
        start_time = time.time()

        options = ["All", "New labels", "Specifically labels"]
        print("\n\n\n")
        for index, label in enumerate(options):
            print(index + 1, ". ", label)
        option = input("Generate images: ")
        if option == "1":
            self.generate_all_labels()
        elif option == "2":
            self.generate_new_labels()
        elif option == "3":
            self.generate_labels()

        end_time = time.time()
        print("Generate images success after", (end_time - start_time) / 60, "minutes")
