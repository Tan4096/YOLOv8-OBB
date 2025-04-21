import os
from pathlib import Path
from tqdm import tqdm

# Specify the directories containing label files (.txt) and images (.jpg)
LABEL_DIR = 'yolov8-obb/DOTA_dataset/labels/val'
IMAGE_DIR = 'yolov8-obb/DOTA_dataset/images/val'


def remove_unlabeled_images(label_root, image_root):
    """
    Removes all .jpg images in 'image_root' that do not have a matching .txt label
    (based on the file stem) in 'label_root'.

    Args:
        label_root (str): Path to the folder containing .txt label files.
        image_root (str): Path to the folder containing .jpg image files.
    """
    # Collect the stem (filename without extension) of each .txt file
    label_stems = set()
    for txt_file in Path(label_root).rglob('*.txt'):
        label_stems.add(txt_file.stem)

    deleted_count = 0
    # Traverse all .jpg files under 'image_root' and remove those not in label_stems
    for img_file in tqdm(list(Path(image_root).rglob('*.jpg')), desc="Checking images"):
        if img_file.stem not in label_stems:
            os.remove(img_file)
            deleted_count += 1

    print(
        f"Operation finished. A total of {deleted_count} images have been deleted.")


if __name__ == "__main__":
    remove_unlabeled_images(LABEL_DIR, IMAGE_DIR)
