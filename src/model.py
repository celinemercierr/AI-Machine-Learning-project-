"""
model.py — FridgeAI
Wrapper around YOLOv8 for food ingredient detection.

Responsibilities:
    - Load a pretrained or fine-tuned YOLOv8 model
    - Expose a clean predict() interface used by infer.py and app.py
    - Provide model info / class names utilities
"""

from pathlib import Path
from typing import List, Optional, Union

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = "yolov8n.pt"                              # COCO pretrained (fallback)
FRIDGE_WEIGHTS  = "outputs/checkpoints/fridgeai_run/weights/best.pt"


# ---------------------------------------------------------------------------
# FridgeModel
# ---------------------------------------------------------------------------

class FridgeModel:
    """
    Thin wrapper around Ultralytics YOLO.

    Usage:
        model = FridgeModel()                        # loads best.pt if it exists
        model = FridgeModel(weights="yolov8n.pt")    # load specific weights
        detections = model.predict("image.jpg")
    """

    def __init__(self, weights: Optional[str] = None):
        """
        Load YOLOv8 weights.

        Priority:
            1. Explicit `weights` argument
            2. Fine-tuned best.pt  (outputs/checkpoints/fridgeai_run/weights/best.pt)
            3. COCO pretrained     (yolov8n.pt)
        """
        if weights is not None:
            weights_path = weights
        elif Path(FRIDGE_WEIGHTS).exists():
            weights_path = FRIDGE_WEIGHTS
            print(f"[FridgeModel] Loading fine-tuned weights: {weights_path}")
        else:
            weights_path = DEFAULT_WEIGHTS
            print(f"[FridgeModel] Fine-tuned weights not found. Loading pretrained: {weights_path}")

        self.model = YOLO(weights_path)
        self.weights_path = weights_path

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict(
        self,
        source: Union[str, list],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """
        Run object detection on one or more images.

        Args:
            source  : path to image, list of paths, or directory
            conf    : minimum confidence threshold (0–1)
            iou     : NMS IoU threshold (0–1)
            imgsz   : inference image size
            device  : 'cpu' or 'cuda'
            verbose : print per-image results

        Returns:
            list of ultralytics Results objects
            Each result has:
                .boxes.xyxy       — bounding boxes (x1,y1,x2,y2)
                .boxes.conf       — confidence scores
                .boxes.cls        — class indices
                .names            — dict {idx: class_name}
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=verbose,
        )
        return results

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def get_class_names(self) -> dict:
        """Return {class_idx: class_name} dict from the loaded model."""
        return self.model.names

    def get_ingredient_list(self, results) -> List[str]:
        """
        Extract unique detected ingredient names from predict() results.

        Args:
            results: list of Results from predict()

        Returns:
            Sorted list of unique ingredient names detected across all images.
            Example: ['apple', 'carrot', 'egg']
        """
        detected = set()
        names = self.get_class_names()

        for result in results:
            for cls_idx in result.boxes.cls.tolist():
                detected.add(names[int(cls_idx)])

        return sorted(detected)

    def info(self):
        """Print model summary (layers, parameters, etc.)."""
        self.model.info()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sanity check — loads model and prints class names
    model = FridgeModel()
    print("\nAvailable classes:")
    for idx, name in model.get_class_names().items():
        print(f"  {idx:3d}: {name}")