"""
train.py — FridgeAI
Fine-tune YOLOv8 on a food ingredient dataset.

Usage:
    python src/train.py --data data/processed/dataset.yaml --epochs 30 --batch 16
    python src/train.py --epochs 5 --batch 8   # quick test on Colab free GPU
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for food detection")

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/dataset.yaml",
        help="Path to the YOLO dataset config file (.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",          # 'n' = nano, fastest on Colab free tier
        help="YOLOv8 variant: yolov8n.pt | yolov8s.pt | yolov8m.pt",
    )
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--img",     type=int,   default=640,  help="Input image size")
    parser.add_argument("--lr",      type=float, default=0.01, help="Initial learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' for GPU, 'cpu' for CPU, '0' for first GPU",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """
    Fine-tune YOLOv8 on the food dataset.

    The training loop (forward pass → loss → backward → optimizer step)
    is handled internally by the Ultralytics YOLO trainer.
    Here we configure all the key hyperparameters and paths.
    """

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained YOLOv8 weights (transfer learning)
    # Starting from COCO weights so the model already knows basic shapes
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # -----------------------------------------------------------------------
    # Fine-tuning
    # The YOLO trainer runs:
    #   1. Forward pass  → predictions (bounding boxes + class scores)
    #   2. Loss          → combined: box loss + classification loss + objectness loss
    #   3. Backward pass → backpropagation (chain rule, as in the lecture notes)
    #   4. Optimizer step → SGD / Adam update
    # Validation mAP is computed at the end of each epoch.
    # -----------------------------------------------------------------------

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Dataset config: {args.data}")
    print(f"Device: {args.device}\n")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        lr0=args.lr,
        device=args.device,
        patience=args.patience,         # early stopping
        save=True,                      # save best.pt and last.pt
        project=str(output_dir),
        name="fridgeai_run",
        exist_ok=True,
        pretrained=True,                # use COCO pretrained weights
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Results summary
    # -----------------------------------------------------------------------

    print("\n--- Training complete ---")
    print(f"Best weights saved to: {output_dir}/fridgeai_run/weights/best.pt")
    print(f"Last weights saved to: {output_dir}/fridgeai_run/weights/last.pt")

    # Print key metrics (for the report)
    metrics = results.results_dict
    print(f"\nFinal validation metrics:")
    print(f"  mAP@0.5       : {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP@0.5:0.95  : {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision     : {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall        : {metrics.get('metrics/recall(B)', 'N/A'):.4f}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
