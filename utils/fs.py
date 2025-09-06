import os
import glob

def get_image_path_list(scene_dir):
    image_dir = os.path.join(scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    image_path_list = sorted(image_path_list)
    return image_path_list