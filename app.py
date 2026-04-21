"""
app.py — FridgeAI
End-to-end demo: upload a fridge photo → detect ingredients → generate recipes.

Run locally:
    python app.py

Run on Colab / Codespaces:
    python app.py
    # Then open the public URL that Gradio prints
"""

import gradio as gr
from PIL import Image
import tempfile
import os

from ultralytics import YOLO
from src.infer import detect_ingredients
from src.recipe_engine import RecipeEngine


# ---------------------------------------------------------------------------
# Load model + recipe engine once at startup
# ---------------------------------------------------------------------------

WEIGHTS_PATH = "outputs/checkpoints/fridgeai_run/weights/best.pt"

# Verify weights exist, fallback to base yolov8n if not found
if not os.path.exists(WEIGHTS_PATH):
    print(f"[WARN] Trained weights not found at {WEIGHTS_PATH}")
    print("[WARN] Falling back to yolov8n.pt (pretrained, not fine-tuned)")
    WEIGHTS_PATH = "yolov8n.pt"

print(f"Loading model: {WEIGHTS_PATH}")
# Pre-load model once so we don't reload on every request
_model = YOLO(WEIGHTS_PATH)

recipe_engine = RecipeEngine()
print("FridgeAI ready.\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image: Image.Image, confidence: float):
    """
    Full pipeline:
        1. Save PIL image to temp file (YOLO needs a path)
        2. Run detection → multiple ingredients with counts & relevance
        3. Generate annotated image with bounding boxes
        4. Generate recipes + nutrition via Gemini (or fallback)
        5. Return annotated image + ingredient list + recipes
    """
    if image is None:
        return None, "Please upload an image.", ""

    # Save to temp file (YOLO needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image.save(tmp_path)

    try:
        # ---- 1. Detect ingredients (new infer.py) --------------------------
        output = detect_ingredients(
            image_path=tmp_path,
            weights=WEIGHTS_PATH,
            conf=confidence,
            debug=False,
        )

        ingredients = output["ingredients"]      # sorted by relevance
        summary     = output["summary"]           # count, area, relevance per ingredient
        detections  = output["detections"]        # every individual box

        # ---- 2. Annotated image with bounding boxes -------------------------
        results = _model.predict(
            source=tmp_path,
            conf=confidence,
            verbose=False,
        )
        annotated = results[0].plot()                        # numpy BGR
        annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR → RGB

        # ---- 3. Format ingredient list for display --------------------------
        if not ingredients:
            return (
                annotated_pil,
                "No ingredients detected.\nTry lowering the confidence slider.",
                "",
            )

        ing_lines = []
        for s in summary:
            name  = s["ingredient"].replace("-", " ").title()
            count = s["count"]
            conf  = s["avg_confidence"]
            ing_lines.append(f"• {name}  ×{count}  (conf {conf:.0%})")

        ingredient_text = "\n".join(ing_lines)

        # ---- 4. Generate recipes + nutrition --------------------------------
        recipes_text = recipe_engine.generate(
            ingredients=ingredients,
            summary=summary,
        )

    finally:
        os.unlink(tmp_path)

    return annotated_pil, ingredient_text, recipes_text


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="FridgeAI", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # FridgeAI
    ### Snap your fridge → discover what you can cook
    Upload a photo of your fridge or ingredients and FridgeAI will detect what's inside
    and suggest recipes you can make right now — with nutritional info.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload your fridge photo",
                height=350,
            )
            confidence_slider = gr.Slider(
                minimum=0.05,
                maximum=0.90,
                value=0.20,
                step=0.05,
                label="Detection confidence threshold",
                info="Lower = detect more (may include false positives). Higher = stricter.",
            )
            detect_btn = gr.Button(
                "Detect & Generate Recipes",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            image_output = gr.Image(label="Detected Ingredients", height=350)

    with gr.Row():
        with gr.Column(scale=1):
            ingredients_output = gr.Textbox(
                label="Detected Ingredients",
                lines=10,
                interactive=False,
            )
        with gr.Column(scale=2):
            recipes_output = gr.Textbox(
                label="Recipe Suggestions & Nutrition",
                lines=18,
                interactive=False,
            )

    detect_btn.click(
        fn=run_pipeline,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, ingredients_output, recipes_output],
    )

    gr.Markdown("""
    ---
    **FridgeAI** · IE Business School · AI & Machine Learning · April 2026
    Model: YOLOv8 fine-tuned on Food Recognition 2022 (50 classes) ·
    Recipes: Gemini API / RecipeNLG fallback
    """)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        share=True,      # public URL — essential for Colab/Codespaces
        debug=False,
    )