"""
dataset.py — FridgeAI
Converts Food Recognition 2022 dataset from COCO format to YOLO format.

COCO format:  one annotations.json file with all bounding boxes
YOLO format:  one .txt file per image with: class_id cx cy w h (normalized 0-1)

Usage:
    python src/dataset.py --data_path /root/.cache/kagglehub/datasets/sainikhileshreddy/food-recognition-2022/versions/11/raw_data
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/.cache/kagglehub/datasets/sainikhileshreddy/food-recognition-2022/versions/11/raw_data",
        help="Path to the raw_data folder of Food Recognition 2022",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed",
        help="Where to save the converted YOLO dataset",
    )
    parser.add_argument(
        "--max_classes",
        type=int,
        default=50,
        help="Max number of food classes to keep (keep it small for Colab free tier)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# COCO → YOLO conversion
# ---------------------------------------------------------------------------

def coco_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bounding box to YOLO format.

    COCO:  [x_min, y_min, width, height]  (pixel values)
    YOLO:  [cx, cy, w, h]                 (normalized 0-1, center point)

    Args:
        bbox:       [x_min, y_min, width, height] in pixels
        img_width:  image width in pixels
        img_height: image height in pixels

    Returns:
        [cx, cy, w, h] normalized between 0 and 1
    """
    x_min, y_min, width, height = bbox

    # Calculate center point
    cx = (x_min + width / 2) / img_width
    cy = (y_min + height / 2) / img_height

    # Normalize width and height
    w = width / img_width
    h = height / img_height

    # Clamp values to [0, 1] to avoid out-of-bounds annotations
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w  = max(0, min(1, w))
    h  = max(0, min(1, h))

    return cx, cy, w, h


def convert_split(split_path, output_images, output_labels, class_map, max_classes):
    """
    Convert one split (train/val/test) from COCO to YOLO format.

    Args:
        split_path:    path to the split folder (contains annotations.json + images/)
        output_images: where to copy the images
        output_labels: where to write the .txt label files
        class_map:     dict mapping original category_id → new class index (0-based)
        max_classes:   only keep annotations for classes in class_map
    """
    annotations_path = os.path.join(split_path, 'annotations.json')
    images_folder    = os.path.join(split_path, 'images')

    if not os.path.exists(annotations_path):
        print(f"  No annotations.json found in {split_path}, skipping.")
        return

    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Build lookup: image_id → image info
    image_info = {img['id']: img for img in data['images']}

    # Build lookup: image_id → list of annotations
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Process each image
    converted = 0
    skipped   = 0

    for img_id, img_data in tqdm(image_info.items(), desc=f"  Converting {os.path.basename(split_path)}"):
        file_name  = img_data['file_name']
        img_width  = img_data['width']
        img_height = img_data['height']

        # Get annotations for this image
        anns = annotations_by_image.get(img_id, [])
        if not anns:
            skipped += 1
            continue

        # Build YOLO lines for this image
        yolo_lines = []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in class_map:
                continue  # skip classes not in our filtered set

            class_idx      = class_map[cat_id]
            cx, cy, w, h   = coco_to_yolo(ann['bbox'], img_width, img_height)
            yolo_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not yolo_lines:
            skipped += 1
            continue

        # Copy image
        src_img = os.path.join(images_folder, file_name)
        dst_img = os.path.join(output_images, file_name)
        if os.path.exists(src_img):
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.copy2(src_img, dst_img)

        # Write label file
        label_file = os.path.join(output_labels, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))

        converted += 1

    print(f"  Done: {converted} images converted, {skipped} skipped (no annotations)")


def write_yaml(output_path, class_names):
    """
    Write the dataset.yaml file that YOLO needs for training.
    This file tells YOLO where the data is and how many classes there are.
    """
    yaml_content = f"""# FridgeAI — YOLO dataset config
path: {os.path.abspath(output_path)}
train: train/images
val:   val/images
test:  test/images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(output_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nDataset YAML saved to: {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=== FridgeAI — COCO to YOLO conversion ===\n")

    # Load categories from training annotations
    train_ann_path = os.path.join(
        args.data_path, 'public_training_set_release_2.0', 'annotations.json'
    )
    with open(train_ann_path, 'r') as f:
        train_data = json.load(f)

    # Get all categories and limit to max_classes
    categories  = train_data['categories'][:args.max_classes]
    class_names = [cat['name'] for cat in categories]
    class_map   = {cat['id']: idx for idx, cat in enumerate(categories)}

    print(f"Using {len(class_names)} food classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    print()

    # Define splits
    splits = {
        'train': 'public_training_set_release_2.0',
        'val':   'public_validation_set_2.0',
        'test':  'public_test_release_2.0',
    }

    # Convert each split
    for split_name, split_folder in splits.items():
        print(f"Processing {split_name}...")
        split_path     = os.path.join(args.data_path, split_folder)
        output_images  = os.path.join(args.output_path, split_name, 'images')
        output_labels  = os.path.join(args.output_path, split_name, 'labels')
        os.makedirs(output_images, exist_ok=True)
        os.makedirs(output_labels, exist_ok=True)
        convert_split(split_path, output_images, output_labels, class_map, args.max_classes)

    # Write dataset.yaml
    write_yaml(args.output_path, class_names)

    print("\n=== Conversion complete! ===")
    print(f"YOLO dataset saved to: {args.output_path}")
    print(f"Run training with:")
    print(f"  python src/train.py --data {args.output_path}/dataset.yaml")


if __name__ == "__main__":
    main()