"""
app.py — FridgeAI
End-to-end demo: upload a fridge photo → detect ingredients → generate recipes.

Run locally:
    python app.py

Run on Colab:
    !python app.py
    # Then open the public URL that Gradio prints
"""

import gradio as gr
from PIL import Image
import tempfile
import os

from src.model import FridgeModel
from src.recipe_engine import RecipeEngine   # adjust if your import differs


# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------

print("Loading FridgeAI model...")
model = FridgeModel()   # loads best.pt if available, else yolov8n.pt
recipe_engine = RecipeEngine()
print("Ready.\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image: Image.Image, confidence: float):
    """
    Full pipeline:
        1. Save PIL image to temp file (YOLO needs a path)
        2. Run detection with FridgeModel
        3. Extract ingredient list
        4. Generate recipes
        5. Return annotated image + ingredient list + recipes
    """
    if image is None:
        return None, "Please upload an image.", ""

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image.save(tmp_path)

    try:
        # 1. Detect ingredients
        results = model.predict(
            source=tmp_path,
            conf=confidence,
            device="cpu",   # Gradio usually runs on CPU for demo
            verbose=False,
        )

        # 2. Get annotated image (YOLO renders bounding boxes)
        annotated = results[0].plot()   # numpy array (BGR)
        annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR → RGB

        # 3. Extract ingredient names
        ingredients = model.get_ingredient_list(results)

        if not ingredients:
            return annotated_pil, "No ingredients detected. Try lowering the confidence threshold.", ""

        ingredient_text = "\n".join(f"• {ing.replace('-', ' ').title()}" for ing in ingredients)

        # 4. Generate recipes
        recipes = recipe_engine.generate(ingredients)

    finally:
        os.unlink(tmp_path)

    return annotated_pil, ingredient_text, recipes


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="FridgeAI", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧊 FridgeAI
    ### Snap your fridge → discover what you can cook
    Upload a photo of your fridge or ingredients and FridgeAI will detect what's inside
    and suggest recipes you can make right now.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📷 Upload your fridge photo",
                height=350,
            )
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Detection confidence threshold",
                info="Lower = detect more (may include false positives). Higher = stricter."
            )
            detect_btn = gr.Button("🔍 Detect & Generate Recipes", variant="primary", size="lg")

        with gr.Column(scale=1):
            image_output = gr.Image(label="🎯 Detected Ingredients", height=350)

    with gr.Row():
        with gr.Column(scale=1):
            ingredients_output = gr.Textbox(
                label="🥦 Detected Ingredients",
                lines=8,
                interactive=False,
            )
        with gr.Column(scale=2):
            recipes_output = gr.Textbox(
                label="👨‍🍳 Recipe Suggestions",
                lines=8,
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
    Model: YOLOv8n fine-tuned on Food Recognition 2022 (50 classes)
    """)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        share=True,      # generates a public URL — essential for Colab
        debug=False,
    )