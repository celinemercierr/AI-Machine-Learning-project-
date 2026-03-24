"""
infer.py — FridgeAI
Run inference on a single fridge image and return detected ingredients.

Usage:
    python src/infer.py --source path/to/fridge.jpg --weights outputs/checkpoints/fridgeai_run/weights/best.pt
    python src/infer.py --source path/to/fridge.jpg --weights best.pt --conf 0.4 --save
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run FridgeAI inference on an image")

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image (jpg, png)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="outputs/checkpoints/fridgeai_run/weights/best.pt",
        help="Path to trained model weights (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Minimum confidence threshold to keep a detection",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated image to outputs/predictions/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions",
        help="Directory to save annotated images",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def detect_ingredients(image_path: str, weights: str, conf: float = 0.4) -> list[str]:
    """
    Run YOLOv8 inference on a fridge image.

    Args:
        image_path: Path to the input image.
        weights:    Path to the trained .pt file.
        conf:       Confidence threshold (0–1). Detections below this are discarded.
                    NMS (Non-Maximum Suppression) is applied automatically by YOLO
                    to remove duplicate boxes — as covered in the lecture notes.

    Returns:
        List of unique detected ingredient names, e.g. ["apple", "egg", "tomato"].
    """
    model = YOLO(weights)

    results = model.predict(
        source=image_path,
        conf=conf,
        verbose=False,
    )

    # Extract class names from detections
    detected = []
    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            detected.append({"ingredient": class_name, "confidence": round(confidence, 3)})

    # Remove duplicates (keep highest confidence per ingredient)
    seen = {}
    for det in detected:
        name = det["ingredient"]
        if name not in seen or det["confidence"] > seen[name]:
            seen[name] = det["confidence"]

    unique_ingredients = list(seen.keys())
    return unique_ingredients, detected


def save_annotated_image(image_path: str, weights: str, output_dir: str, conf: float):
    """Save a copy of the image with bounding boxes drawn."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    model.predict(
        source=image_path,
        conf=conf,
        save=True,
        project=output_dir,
        name="result",
        exist_ok=True,
    )
    print(f"Annotated image saved to: {output_dir}/result/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print(f"Running inference on: {args.source}")
    print(f"Model weights: {args.weights}")
    print(f"Confidence threshold: {args.conf}\n")

    ingredients, all_detections = detect_ingredients(args.source, args.weights, args.conf)

    print("Detected ingredients:")
    for det in all_detections:
        print(f"  - {det['ingredient']} (conf: {det['confidence']})")

    print(f"\nUnique ingredients: {ingredients}")
    print(json.dumps({"ingredients": ingredients}, indent=2))

    if args.save:
        save_annotated_image(args.source, args.weights, args.output, args.conf)
