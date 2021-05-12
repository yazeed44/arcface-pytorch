from PIL import Image
from autocrop import Cropper
import os
import settings
input_dataset_path = settings.pre_processing_input_dataset_dir
output_dataset_path = settings.pre_processing_output_dataset_dir


# Pre processing steps
# 1. Collect image paths in input path
# 2. Detect faces and crop (112x96) using autocrop library
# 3. Save cropped picture
# 4. Flip cropped picture horizontally
# 5. Save flipped picture

def get_cacd2000_image_paths_with_identities(directory_name="CACD2000"):
    cacd_image_paths = os.listdir(os.path.join(input_dataset_path, directory_name))
    names = [path.split('_')[1:-1] for path in cacd_image_paths]
    unique_names = []
    for name_array in names:
        if name_array not in unique_names:
            unique_names.append(name_array)
    names_with_paths = []
    for name_array in unique_names:
        images_under_name = []
        for image_path in cacd_image_paths:
            if all([name in image_path for name in name_array]):
                images_under_name.append(os.path.join(input_dataset_path, directory_name, image_path))
        standard_name = ' '.join(name_array)
        names_with_paths.append((standard_name, images_under_name))

    return names_with_paths


def get_casia_image_paths_with_identities(directory_name="CASIA-WebFace", labels_file_name="casia_webface_labels.txt"):
    ids_to_names = []
    with open(os.path.join(input_dataset_path, directory_name, labels_file_name)) as label_file:
        for line in label_file:
            ids_to_names.append(line.split())
    names_with_paths = []
    for id, name in ids_to_names:
        image_paths = os.listdir(os.path.join(input_dataset_path, directory_name, id))
        image_paths = [os.path.join(input_dataset_path, directory_name, id, path) for path in image_paths]
        names_with_paths.append((name.replace('_', ' '), image_paths))

    return names_with_paths


def create_dirs_for_names(names_to_images):
    for name, _ in names_to_images:
        try:
            os.mkdir(os.path.join(output_dataset_path, name))
        except FileExistsError:
            continue


cropper = Cropper(width=112, height=96)
names_to_images = get_casia_image_paths_with_identities() + get_cacd2000_image_paths_with_identities()
create_dirs_for_names(names_to_images)
for identity_name, identity_images in names_to_images:
    for image_path in identity_images:
        cropped_pixels = cropper.crop(image_path)
        # if face is in the image, then result will be an array of pixels, otherwise will be None

        if cropped_pixels is None:
            continue
        # cropped_pixels = (cropped_pixels - 127.5) / 128 # Normalize pixels
        cropped_image = Image.fromarray(cropped_pixels)
        file_name, file_extension = os.path.splitext(image_path)
        output_normal_file_name = os.path.basename(file_name) + "_normal" + file_extension
        output_normal_path = os.path.join(output_dataset_path, identity_name, output_normal_file_name)
        cropped_image.save(output_normal_path)

        cropped_image = cropped_image.transpose(method=Image.FLIP_LEFT_RIGHT)
        output_horizontal_file_name = os.path.basename(file_name) + "_horizontal" + file_extension
        output_horizontal_path = os.path.join(output_dataset_path, identity_name, output_horizontal_file_name)
        cropped_image.save(output_horizontal_path)
