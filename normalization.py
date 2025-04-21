import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths to DOTA dataset splits
VAL_LABEL_DIR = 'DOTA_dataset_split/labels/val'
VAL_IMAGE_DIR = 'DOTA_dataset_split/images/val'
TRAIN_LABEL_DIR = 'DOTA_dataset_split/labels/train'
TRAIN_IMAGE_DIR = 'DOTA_dataset_split/images/train'

# Paths where YOLO-format labels will be stored
VAL_OUTPUT_DIR = 'yolo_labels/val'
TRAIN_OUTPUT_DIR = 'yolo_labels/train'

# Create output directories if they do not exist
os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)


def convert_dota_to_yolo(label_file_path, image_file_path):
    """
    Converts a label file in DOTA format to YOLO format, normalizing the four vertex coordinates.

    Args:
        label_file_path (str): Path to the DOTA-format label file.
        image_file_path (str): Path to the corresponding image file (JPG).

    Returns:
        list of str: A list of strings, each representing one object in YOLO format:
                     "class_id x1 y1 x2 y2 x3 y3 x4 y4"
    """
    # Read the image to determine its dimensions
    image_data = cv2.imread(image_file_path)
    img_height, img_width, _ = image_data.shape

    # Read the DOTA label file
    with open(label_file_path, 'r') as f:
        raw_lines = f.readlines()

    yolo_label_lines = []

    for line in raw_lines:
        values = list(map(float, line.strip().split()))
        # The first value is class ID, the next 8 values are the coordinates
        current_class = int(values[0])
        x1, y1, x2, y2, x3, y3, x4, y4 = values[1:]

        # Normalize the coordinates
        x1 /= img_width
        y1 /= img_height
        x2 /= img_width
        y2 /= img_height
        x3 /= img_width
        y3 /= img_height
        x4 /= img_width
        y4 /= img_height

        # Construct the line for YOLO format
        yolo_label = f"{current_class} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"
        yolo_label_lines.append(yolo_label)

    return yolo_label_lines


def process_and_save_labels(label_dir, image_dir, output_dir):
    """
    Scans through all .txt labels in 'label_dir', converts them to YOLO format,
    and writes each converted label file into 'output_dir'.

    Args:
        label_dir (str): Directory containing DOTA .txt label files.
        image_dir (str): Directory containing .jpg images corresponding to the labels.
        output_dir (str): Destination directory for the new YOLO-format .txt files.
    """
    label_files = [file for file in os.listdir(
        label_dir) if file.endswith(".txt")]

    for label_file in tqdm(label_files, desc=f"Converting labels in {label_dir}", unit="file"):
        label_path = os.path.join(label_dir, label_file)
        image_name = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_name)

        # Convert DOTA labels to YOLO format
        yolo_lines = convert_dota_to_yolo(label_path, image_path)

        # Write the YOLO-format labels to a new file
        output_file_path = os.path.join(output_dir, label_file)
        with open(output_file_path, 'w') as out_f:
            out_f.write("\n".join(yolo_lines))


# Convert label files for the validation split
process_and_save_labels(VAL_LABEL_DIR, VAL_IMAGE_DIR, VAL_OUTPUT_DIR)

# Convert label files for the training split
process_and_save_labels(TRAIN_LABEL_DIR, TRAIN_IMAGE_DIR, TRAIN_OUTPUT_DIR)