"""
infer.py — FridgeAI
Run inference on a single fridge image and return detected ingredients.

Usage:
    python src/infer.py --source path/to/fridge.jpg --weights outputs/checkpoints/fridgeai_run/weights/best.pt
    python src/infer.py --source path/to/fridge.jpg --weights best.pt --conf 0.20 --save
"""

import argparse
import json
from pathlib import Path
from collections import Counter

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
        default=0.20,                       # ← lowered from 0.4
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug information for each detection",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_area(box) -> float:
    """Return the area (in pixels²) of a detection bounding box."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (x2 - x1) * (y2 - y1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def detect_ingredients(
    image_path: str,
    weights: str,
    conf: float = 0.20,
    debug: bool = False,
) -> dict:
    """
    Run YOLOv8 inference on a fridge image.

    Args:
        image_path: Path to the input image.
        weights:    Path to the trained .pt file.
        conf:       Confidence threshold (0-1). Lowered to 0.20 to capture
                    more detections; YOLO's built-in NMS still removes
                    duplicate/overlapping boxes automatically.
        debug:      If True, print raw detection info for every single box.

    Returns:
        dict with:
            - ingredients : list[str]  — unique ingredient names sorted by
                                         relevance (count × area)
            - detections  : list[dict] — every individual detection
            - summary     : list[dict] — per-ingredient stats: count,
                                         avg confidence, total area
    """
    model = YOLO(weights)

    # ---- DEBUG: what goes IN ------------------------------------------------
    if debug:
        print("=" * 60)
        print(f"[DEBUG] Image path  : {image_path}")
        print(f"[DEBUG] Weights     : {weights}")
        print(f"[DEBUG] Conf thresh : {conf}")
        print("=" * 60)

    results = model.predict(
        source=image_path,
        conf=conf,
        verbose=False,
    )

    # ---- Collect ALL detections ---------------------------------------------
    all_detections = []
    for result in results:

        if debug:
            print(f"[DEBUG] Total raw boxes returned by YOLO: {len(result.boxes)}")

        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            area       = _box_area(box)

            det = {
                "ingredient":  class_name,
                "confidence":  round(confidence, 3),
                "area_px":     round(area, 1),
                "bbox":        [round(c, 1) for c in box.xyxy[0].tolist()],
            }
            all_detections.append(det)

            # ---- DEBUG: what comes OUT per box ------------------------------
            if debug:
                print(
                    f"  [DEBUG] class={class_name:20s}  "
                    f"conf={confidence:.3f}  "
                    f"area={area:>10.1f}px²  "
                    f"bbox={det['bbox']}"
                )

    # ---- Build per-ingredient summary (count, avg conf, total area) ---------
    ingredient_stats: dict[str, dict] = {}
    for det in all_detections:
        name = det["ingredient"]
        if name not in ingredient_stats:
            ingredient_stats[name] = {
                "ingredient": name,
                "count": 0,
                "total_conf": 0.0,
                "total_area": 0.0,
            }
        ingredient_stats[name]["count"]      += 1
        ingredient_stats[name]["total_conf"] += det["confidence"]
        ingredient_stats[name]["total_area"] += det["area_px"]

    summary = []
    for stats in ingredient_stats.values():
        summary.append({
            "ingredient":    stats["ingredient"],
            "count":         stats["count"],
            "avg_confidence": round(stats["total_conf"] / stats["count"], 3),
            "total_area":    round(stats["total_area"], 1),
            # relevance score: more instances + bigger area = more important
            "relevance":     round(stats["count"] * stats["total_area"], 1),
        })

    # Sort by relevance (highest first) so recipes focus on what's most present
    summary.sort(key=lambda s: s["relevance"], reverse=True)

    # Final ingredient list, ordered by relevance
    ingredients = [s["ingredient"] for s in summary]

    return {
        "ingredients": ingredients,
        "detections":  all_detections,
        "summary":     summary,
    }


# ---------------------------------------------------------------------------
# Save annotated image
# ---------------------------------------------------------------------------

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
    print(f"Model weights:        {args.weights}")
    print(f"Confidence threshold: {args.conf}\n")

    output = detect_ingredients(
        args.source, args.weights, args.conf, debug=args.debug
    )

    # ---- Pretty print summary -----------------------------------------------
    print("\n--- Detection Summary ---")
    for s in output["summary"]:
        print(
            f"  {s['ingredient']:20s}  "
            f"count={s['count']}  "
            f"avg_conf={s['avg_confidence']:.3f}  "
            f"total_area={s['total_area']:>10.1f}px²  "
            f"relevance={s['relevance']:>12.1f}"
        )

    print(f"\nIngredients (by relevance): {output['ingredients']}")
    print(json.dumps({"ingredients": output["ingredients"]}, indent=2))

    if args.save:
        save_annotated_image(args.source, args.weights, args.output, args.conf)