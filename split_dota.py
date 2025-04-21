import itertools
import os
from glob import glob
from math import ceil
from pathlib import Path
import contextlib

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

# Replace with your actual classes file path
classes_path = ''


def map_img_to_labels(img_list):
    """
    Maps image file paths to corresponding label file paths, following the structure:
    /images/xxx.jpg -> /labels/xxx.txt
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    label_list = []
    for img_path in img_list:
        # Replace "images" with "labels" and change the extension to ".txt"
        new_path = sb.join(img_path.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"
        label_list.append(new_path)
    return label_list


def get_corrected_size(pil_image: Image.Image):
    """
    Gets the width and height of a PIL image, corrected according to JPEG Exif orientation data.
    Some images may carry orientation info (e.g., rotated by 90 or 270 degrees).
    """
    w, h = pil_image.size  # (width, height)
    if pil_image.format == "JPEG":
        with contextlib.suppress(Exception):
            exif_data = pil_image.getexif()
            if exif_data:
                orientation = exif_data.get(274, None)
                if orientation in {6, 8}:
                    w, h = h, w
    return w, h


def compute_bbox_iof(poly_coords, boxes, tiny=1e-6):
    """
    Calculates IOF between polygons and boxes. 
    IOF = (area of intersection) / (area of the polygon).

    Args:
        poly_coords (np.ndarray): shape (N, 8), each polygon has 4 points: (x1, y1, x2, y2, x3, y3, x4, y4).
        boxes (np.ndarray): shape (M, 4), each row is [x1, y1, x2, y2].
        tiny (float): A small epsilon to avoid division by zero.

    Returns:
        np.ndarray: An IOF matrix of shape (N, M).
    """
    # Reshape polygon coordinates to (N, 4, 2)
    polys_reshaped = poly_coords.reshape(-1, 4, 2)
    left_top = np.min(polys_reshaped, axis=-2)
    right_bottom = np.max(polys_reshaped, axis=-2)
    # Form bounding boxes for polygons (N, 4)
    box1 = np.concatenate([left_top, right_bottom], axis=-1)

    # Compute overlap of each pair in terms of outer bounding boxes
    lt = np.maximum(box1[:, None, :2], boxes[..., :2])
    rb = np.minimum(box1[:, None, 2:], boxes[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    area_inters = wh[..., 0] * wh[..., 1]

    # Convert each box to a polygon with 4 points
    x1, y1, x2, y2 = [boxes[..., i] for i in range(4)]
    box_coords = np.stack([x1, y1, x2, y1, x2, y2, x1, y2],
                          axis=-1).reshape(-1, 4, 2)

    # Build shapely Polygons for actual intersection
    poly_list1 = [Polygon(p) for p in polys_reshaped]
    poly_list2 = [Polygon(q) for q in box_coords]

    # Compute the true intersection area (not just the bounding box overlap)
    iof_matrix = np.zeros(area_inters.shape)
    all_indexes = np.nonzero(area_inters)
    for iidx in zip(*all_indexes):
        iof_matrix[iidx] = poly_list1[iidx[0]].intersection(
            poly_list2[iidx[-1]]).area

    # Calculate polygon areas
    area_poly1 = np.array([poly.area for poly in poly_list1],
                          dtype=np.float32).reshape(-1, 1)
    area_poly1 = np.clip(area_poly1, tiny, np.inf)

    result = iof_matrix / area_poly1
    if result.ndim == 1:
        result = result[..., None]
    return result


def load_dota_yolo(dataset_dir, data_split="train"):
    """
    Loads image and label information from the DOTA dataset in YOLO format.

    Assumes a directory structure:
    dataset_dir/
       ├── images
       │     ├── train
       │     └── val
       └── labels
             ├── train
             └── val

    Args:
        dataset_dir (str): The root path to the dataset.
        data_split (str): Either "train" or "val".

    Returns:
        list of dict: Each dict contains keys {ori_size, label, filepath}.
    """
    class_list = []
    with open(classes_path, 'r') as f:
        for line in f:
            cls_name = line.strip()
            class_list.append(cls_name)

    assert data_split in {
        "train", "val"}, "data_split must be 'train' or 'val'."
    img_folder = Path(dataset_dir) / "images" / data_split
    if not img_folder.exists():
        raise FileNotFoundError(f"Path does not exist: {img_folder}")

    img_files = glob(str(img_folder / "*"))
    label_files = map_img_to_labels(img_files)

    data_records = []
    for img_fp, lbl_fp in zip(img_files, label_files):
        # Use PIL to get image dimensions
        w, h = get_corrected_size(Image.open(img_fp))

        with open(lbl_fp, "r") as f:
            lines = [i.strip().split()
                     for i in f.read().strip().splitlines() if i.strip()]
            # Convert class names to class indices according to class_list
            for idx, row in enumerate(lines):
                # row[8] is class name; row[0..7] are coordinates plus an additional "difficult" marker, etc.
                lines[idx] = [class_list.index(row[8])] + row[:-2]
            labels_arr = np.array(lines, dtype=np.float32)

        data_records.append(dict(
            ori_size=(h, w),
            label=labels_arr,
            filepath=img_fp
        ))
    return data_records


def generate_windows(img_size, crop_opts=(1024,), step_opts=(200,), area_ratio=0.6, epsilon=0.01):
    """
    Generates a list of sliding windows for the given image size, using the specified crop sizes and steps.

    area_ratio (float): Minimum ratio of the window's valid area inside the image.
    epsilon (float): Small threshold for numerical stability.
    """
    img_h, img_w = img_size
    all_windows = []

    for c_size, step_size in zip(crop_opts, step_opts):
        if c_size <= step_size:
            raise ValueError(
                f"Invalid crop and gap values: crop={c_size}, gap={step_size}")

        actual_step = c_size - step_size

        # Calculate horizontal sliding
        x_count = 1 if img_w <= c_size else ceil(
            (img_w - c_size) / actual_step + 1)
        x_points = [actual_step * i for i in range(x_count)]
        if len(x_points) > 1 and x_points[-1] + c_size > img_w:
            x_points[-1] = img_w - c_size

        # Calculate vertical sliding
        y_count = 1 if img_h <= c_size else ceil(
            (img_h - c_size) / actual_step + 1)
        y_points = [actual_step * i for i in range(y_count)]
        if len(y_points) > 1 and y_points[-1] + c_size > img_h:
            y_points[-1] = img_h - c_size

        # Build [x1, y1, x2, y2] for each position
        start_points = np.array(
            list(itertools.product(x_points, y_points)), dtype=np.int64)
        end_points = start_points + c_size
        all_windows.append(np.hstack([start_points, end_points]))

    windows_all = np.vstack(all_windows)

    # Calculate how much of each window lies inside the image
    clipped_wins = windows_all.copy()
    clipped_wins[:, 0::2] = np.clip(clipped_wins[:, 0::2], 0, img_w)
    clipped_wins[:, 1::2] = np.clip(clipped_wins[:, 1::2], 0, img_h)

    ideal_areas = (windows_all[:, 2] - windows_all[:, 0]) * \
        (windows_all[:, 3] - windows_all[:, 1])
    real_areas = (clipped_wins[:, 2] - clipped_wins[:, 0]) * \
        (clipped_wins[:, 3] - clipped_wins[:, 1])
    ratio = real_areas / ideal_areas

    # If no window meets the area ratio requirement, keep only the one(s) with maximum overlap
    if not (ratio > area_ratio).any():
        max_ratio = ratio.max()
        ratio[abs(ratio - max_ratio) < epsilon] = 1
    return windows_all[ratio > area_ratio]


def find_objects_in_windows(record, window_boxes, iof_threshold=0.7):
    """
    For each generated window, finds objects whose IOF with that window is above the threshold.

    record: A dict containing image data and labels.
    window_boxes: Sliding window coordinates of shape (N, 4).
    iof_threshold: IOF threshold for filtering objects.
    """
    height, width = record["ori_size"]
    label_data = record["label"]

    if len(label_data) == 0:
        # Return empty arrays for all windows
        return [np.zeros((0, 9), dtype=np.float32)] * len(window_boxes)

    # Calculate IOF
    iof_matrix = compute_bbox_iof(label_data[:, 1:], window_boxes)
    box_to_objs = []
    for i in range(len(window_boxes)):
        valid_mask = iof_matrix[:, i] >= iof_threshold
        box_to_objs.append(label_data[valid_mask])
    return box_to_objs


def slice_and_export(record, window_boxes, window_labels, out_img_dir, out_label_dir):
    """
    Slices the original image according to window boxes and saves the corresponding label files.

    record: Dict with keys {ori_size, label, filepath}.
    window_boxes: Coordinates of shape (N, 4).
    window_labels: A list of length N, each containing labels for the corresponding window.
    out_img_dir, out_label_dir: Output directories for images and labels.
    """
    src_img = cv2.imread(record["filepath"])
    base_name = Path(record["filepath"]).stem

    for idx, w_box in enumerate(window_boxes):
        x1, y1, x2, y2 = w_box
        crop_width = x2 - x1
        crop_height = y2 - y1
        patch = src_img[y1:y2, x1:x2]
        new_name = f"{base_name}__{crop_width}__{x1}___{y1}"

        # Write the patch image
        out_img_path = Path(out_img_dir) / f"{new_name}.jpg"
        cv2.imwrite(str(out_img_path), patch)

        # Write the corresponding label if available
        cur_label = window_labels[idx]
        if len(cur_label) == 0:
            # Skip empty labels
            continue

        # Shift coordinates to local window
        cur_label[:, 1::2] -= x1
        cur_label[:, 2::2] -= y1

        out_label_path = Path(out_label_dir) / f"{new_name}.txt"
        with open(out_label_path, "w") as f:
            for row in cur_label:
                coords_str = " ".join(f"{v:.6g}" for v in row[1:])
                f.write(f"{int(row[0])} {coords_str}\n")


def segment_img_and_label(src_root, dest_root, data_split="train", c_sizes=(1024,), step_sizes=(200,)):
    """
    Performs image slicing for the given data_split ('train' or 'val') and exports corresponding labels.

    Directory structure assumption:
    src_root
    ├── images
    │   └── data_split
    └── labels
        └── data_split

    Output directory structure:
    dest_root
    ├── images
    │   └── data_split
    └── labels
        └── data_split
    """
    dst_img_folder = Path(dest_root) / "images" / data_split
    dst_img_folder.mkdir(parents=True, exist_ok=True)
    dst_lbl_folder = Path(dest_root) / "labels" / data_split
    dst_lbl_folder.mkdir(parents=True, exist_ok=True)

    records = load_dota_yolo(src_root, data_split)
    for rec in tqdm(records, total=len(records), desc=f"Processing {data_split}"):
        windows = generate_windows(rec["ori_size"], c_sizes, step_sizes)
        win_objs = find_objects_in_windows(rec, windows)
        slice_and_export(rec, windows, win_objs,
                         dst_img_folder, dst_lbl_folder)


def segment_train_val(src_root, dest_root, base_size=1024, gap_val=200, scale_rates=(1.0,)):
    """
    Generates sliced data for both 'train' and 'val' sets in the DOTA dataset.

    scale_rates: Allows multi-scale slicing, e.g., (1.0, 0.5).
    """
    c_sizes, steps = [], []
    for rate in scale_rates:
        c_sizes.append(int(base_size / rate))
        steps.append(int(gap_val / rate))

    for ds_split in ["train", "val"]:
        segment_img_and_label(src_root, dest_root, ds_split, c_sizes, steps)


def segment_test_data(src_root, dest_root, base_size=1024, gap_val=200, scale_rates=(1.0,)):
    """
    Handles slicing for the test set in the DOTA dataset, which does not have labels.

    src_root:
       └── images
             └── test

    dest_root:
       └── images
             └── test
    """
    out_test_dir = Path(dest_root) / "images" / "test"
    out_test_dir.mkdir(parents=True, exist_ok=True)

    c_sizes, steps = [], []
    for rate in scale_rates:
        c_sizes.append(int(base_size / rate))
        steps.append(int(gap_val / rate))

    test_img_dir = Path(src_root) / "images" / "test"
    if not test_img_dir.exists():
        raise FileNotFoundError(
            f"Test directory does not exist: {test_img_dir}")

    all_imgs = glob(str(test_img_dir / "*"))
    for img_path in tqdm(all_imgs, total=len(all_imgs), desc="Processing test"):
        w, h = get_corrected_size(Image.open(img_path))
        all_windows = generate_windows((h, w), c_sizes, steps)
        img_cv = cv2.imread(img_path)
        prefix_name = Path(img_path).stem

        for box in all_windows:
            x1, y1, x2, y2 = box
            patch_name = f"{prefix_name}__{x2 - x1}__{x1}___{y1}"
            patch_img = img_cv[y1:y2, x1:x2]
            out_patch_path = out_test_dir / f"{patch_name}.jpg"
            cv2.imwrite(str(out_patch_path), patch_img)


if __name__ == "__main__":
    # To prevent PIL from raising errors on very large images
    Image.MAX_IMAGE_PIXELS = 400000000

    segment_train_val(
        src_root="yolov8-obb/DOTA_dataset",
        dest_root="yolov8-obb/DOTA_dataset_split",
        base_size=640,
        gap_val=200,
        scale_rates=(1.0,)
    )

    segment_test_data(
        src_root="yolov8-obb/DOTA_dataset",
        dest_root="yolov8-obb/DOTA_dataset_split",
        base_size=640,
        gap_val=200,
        scale_rates=(1.0,)
    )
