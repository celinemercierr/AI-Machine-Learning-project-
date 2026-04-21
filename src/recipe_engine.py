"""
recipe_engine.py — FridgeAI
Given a list of detected ingredients (with optional counts/relevance),
generate recipes with nutritional info.

Primary:   Google Gemini API (free tier) — generates recipes + nutrition as JSON
Fallback:  RecipeNLG dataset search + Open Food Facts nutrition lookup

Usage:
    # With Gemini (set env var first):
    export GEMINI_API_KEY="your-key-here"
    python src/recipe_engine.py --ingredients "chicken,tomato,egg"

    # Without API key → falls back to RecipeNLG + Open Food Facts
    python src/recipe_engine.py --ingredients "chicken,tomato,egg" --fallback
"""

import argparse
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate recipes from detected ingredients")
    parser.add_argument(
        "--ingredients",
        type=str,
        required=True,
        help="Comma-separated list of detected ingredients e.g. 'chicken,tomato,egg'",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional: JSON string with ingredient summary from infer.py "
             "(includes count, relevance). If not provided, uses flat list.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of recipes to return",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Force use of RecipeNLG fallback instead of Gemini",
    )
    parser.add_argument(
        "--recipes_path",
        type=str,
        default="/home/codespace/.cache/kagglehub/datasets/saldenisov/recipenlg/versions/1",
        help="Path to RecipeNLG dataset (used in fallback mode)",
    )
    parser.add_argument(
        "--nutrition_path",
        type=str,
        default="/home/codespace/.cache/kagglehub/datasets/openfoodfacts/world-food-facts/versions/5",
        help="Path to Open Food Facts dataset (used in fallback mode)",
    )
    return parser.parse_args()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PRIMARY MODE — Gemini API (free tier)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

GEMINI_PROMPT_TEMPLATE = """You are a professional chef and nutritionist.
A user took a photo of their fridge and our AI detected these ingredients:

{ingredient_info}

Based on what is available, suggest exactly {top_k} recipes that:
- Prioritize the ingredients with the highest count/relevance (the user has more of those)
- Are practical, everyday meals (not overly complex)
- Minimize extra ingredients the user would need to buy
- Include estimated nutritional info per serving

Respond ONLY with valid JSON (no markdown, no backticks, no extra text).
Use this exact structure:

{{
  "recipes": [
    {{
      "title": "Recipe Name",
      "match_score": 3,
      "ingredients_used": ["chicken", "tomato"],
      "extra_ingredients_needed": ["olive oil", "salt"],
      "servings": 4,
      "prep_time_min": 15,
      "cook_time_min": 30,
      "difficulty": "easy",
      "steps": [
        "Step 1 description",
        "Step 2 description"
      ],
      "nutrition_per_serving": {{
        "calories_kcal": 350,
        "protein_g": 28.0,
        "carbs_g": 15.0,
        "fat_g": 12.0,
        "fiber_g": 3.0
      }}
    }}
  ],
  "nutrition_summary": [
    {{
      "ingredient": "chicken",
      "calories_per_100g": 239,
      "protein_per_100g": 27.0,
      "carbs_per_100g": 0.0,
      "fat_per_100g": 14.0
    }}
  ]
}}
"""


def _build_ingredient_info(ingredients: list[str], summary: list[dict] | None) -> str:
    """Format ingredient info for the prompt, using summary if available."""
    if summary:
        lines = []
        for s in summary:
            lines.append(
                f"- {s['ingredient']}: detected {s['count']} time(s), "
                f"avg confidence {s.get('avg_confidence', 'N/A')}, "
                f"relevance score {s.get('relevance', 'N/A')}"
            )
        return "\n".join(lines)
    else:
        return "\n".join(f"- {ing}" for ing in ingredients)


def generate_recipes_gemini(
    ingredients: list[str],
    summary: list[dict] | None = None,
    top_k: int = 3,
    api_key: str | None = None,
) -> dict | None:
    """
    Call Gemini API to generate recipes + nutrition from ingredients.

    Args:
        ingredients: list of ingredient names
        summary:     optional list of dicts from infer.py with count/relevance
        top_k:       number of recipes to generate
        api_key:     Gemini API key (reads from GEMINI_API_KEY env var if None)

    Returns:
        dict with 'recipes' and 'nutrition_summary', or None on failure
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[INFO] No GEMINI_API_KEY found — will use fallback mode.")
        return None

    try:
        import requests
    except ImportError:
        print("[WARN] 'requests' library not installed. pip install requests")
        return None

    ingredient_info = _build_ingredient_info(ingredients, summary)
    prompt = GEMINI_PROMPT_TEMPLATE.format(
        ingredient_info=ingredient_info,
        top_k=top_k,
    )

    # Gemini API endpoint (free tier: gemini-2.0-flash)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
        },
    }

    print("[INFO] Calling Gemini API for recipe generation...")
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Extract text from Gemini response
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Clean potential markdown fences
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        result = json.loads(text)
        print("[INFO] Gemini returned recipes successfully.")
        return result

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gemini API request failed: {e}")
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[ERROR] Could not parse Gemini response: {e}")
        return None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FALLBACK MODE — RecipeNLG + Open Food Facts                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_recipes(recipes_path: str):
    """Load RecipeNLG dataset."""
    import pandas as pd

    csv_path = os.path.join(recipes_path, "full_dataset.csv")
    if not os.path.exists(csv_path):
        for f in Path(recipes_path).rglob("*.csv"):
            csv_path = str(f)
            break

    print(f"[FALLBACK] Loading recipes from: {csv_path}")
    df = pd.read_csv(csv_path, nrows=50000)
    return df


def find_recipes_fallback(ingredients: list[str], recipes_df, top_k: int = 3):
    """Score recipes by how many detected ingredients they contain."""
    ingredients_lower = [ing.lower().replace("-", " ") for ing in ingredients]

    def score_recipe(row):
        try:
            recipe_text = str(row["ingredients"]).lower()
            return sum(1 for ing in ingredients_lower if ing in recipe_text)
        except Exception:
            return 0

    print("[FALLBACK] Searching RecipeNLG...")
    recipes_df["match_score"] = recipes_df.apply(score_recipe, axis=1)

    top = (
        recipes_df[recipes_df["match_score"] > 0]
        .sort_values("match_score", ascending=False)
        .head(top_k)
    )

    results = []
    for _, row in top.iterrows():
        recipe = {
            "title":       row.get("title", "Unknown"),
            "match_score": int(row["match_score"]),
            "ingredients": row.get("ingredients", ""),
            "steps":       [],
        }
        try:
            dirs = json.loads(row.get("directions", "[]"))
            if isinstance(dirs, list):
                recipe["steps"] = dirs
        except Exception:
            pass
        results.append(recipe)

    return results


def load_nutrition(nutrition_path: str):
    """Load Open Food Facts dataset."""
    import pandas as pd

    tsv_path = None
    for f in Path(nutrition_path).rglob("*.tsv"):
        tsv_path = str(f)
        break
    for f in Path(nutrition_path).rglob("*.csv"):
        tsv_path = str(f)
        break

    if not tsv_path:
        print("[FALLBACK] Open Food Facts file not found, skipping nutrition.")
        return None

    print(f"[FALLBACK] Loading nutrition from: {tsv_path}")
    cols = [
        "product_name", "energy_100g", "proteins_100g",
        "carbohydrates_100g", "fat_100g",
    ]
    try:
        df = pd.read_csv(tsv_path, sep="\t", usecols=cols, nrows=100000, low_memory=False)
        df = df.dropna(subset=["product_name"])
        return df
    except Exception as e:
        print(f"[FALLBACK] Could not load nutrition data: {e}")
        return None


def get_nutrition_fallback(ingredient: str, nutrition_df):
    """Look up nutritional info from Open Food Facts for one ingredient."""
    if nutrition_df is None:
        return {"ingredient": ingredient, "note": "nutrition database unavailable"}

    matches = nutrition_df[
        nutrition_df["product_name"]
        .str.lower()
        .str.contains(ingredient.lower().replace("-", " "), na=False)
    ].head(1)

    if matches.empty:
        return {"ingredient": ingredient, "note": "not found in database"}

    row = matches.iloc[0]
    return {
        "ingredient":       ingredient,
        "calories_per_100g": round(float(row.get("energy_100g", 0) or 0) / 4.184, 1),
        "protein_per_100g":  round(float(row.get("proteins_100g", 0) or 0), 1),
        "carbs_per_100g":    round(float(row.get("carbohydrates_100g", 0) or 0), 1),
        "fat_per_100g":      round(float(row.get("fat_100g", 0) or 0), 1),
    }


def generate_recipes_fallback(
    ingredients: list[str],
    recipes_path: str,
    nutrition_path: str,
    top_k: int = 3,
) -> dict:
    """Fallback pipeline using RecipeNLG + Open Food Facts."""
    recipes_df = load_recipes(recipes_path)
    recipes    = find_recipes_fallback(ingredients, recipes_df, top_k)

    nutrition_df = load_nutrition(nutrition_path)
    nutrition    = [get_nutrition_fallback(ing, nutrition_df) for ing in ingredients]

    return {
        "recipes":           recipes,
        "nutrition_summary": nutrition,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Unified interface                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run(
    ingredients: list[str],
    summary: list[dict] | None = None,
    top_k: int = 3,
    force_fallback: bool = False,
    recipes_path: str = "",
    nutrition_path: str = "",
) -> dict:
    """
    Full pipeline: ingredients → recipes + nutrition.

    Tries Gemini first; if unavailable or fails, falls back to RecipeNLG.

    Args:
        ingredients:     list of detected ingredient names (sorted by relevance)
        summary:         optional per-ingredient stats from infer.py
        top_k:           number of recipes to return
        force_fallback:  skip Gemini and go straight to RecipeNLG
        recipes_path:    path for RecipeNLG (fallback)
        nutrition_path:  path for Open Food Facts (fallback)

    Returns:
        dict with 'recipes', 'nutrition_summary', and metadata
    """
    print(f"\n{'='*50}")
    print(f"  FridgeAI Recipe Engine")
    print(f"  Ingredients: {ingredients}")
    print(f"{'='*50}\n")

    result = None

    # --- Try Gemini first ---
    if not force_fallback:
        result = generate_recipes_gemini(ingredients, summary, top_k)

    # --- Fallback to RecipeNLG ---
    if result is None:
        print("[INFO] Using fallback mode (RecipeNLG + Open Food Facts).")
        result = generate_recipes_fallback(
            ingredients, recipes_path, nutrition_path, top_k
        )
        result["source"] = "fallback_recipenlg"
    else:
        result["source"] = "gemini_api"

    result["ingredients_detected"] = ingredients
    return result


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_output(output: dict):
    """Pretty print the results to terminal."""
    source = output.get("source", "unknown")
    print(f"\n{'='*60}")
    print(f"  SOURCE: {source}")
    print(f"  DETECTED: {output['ingredients_detected']}")
    print(f"{'='*60}")

    print("\nRECIPES:")
    for i, recipe in enumerate(output.get("recipes", []), 1):
        title = recipe.get("title", "Untitled")
        score = recipe.get("match_score", "N/A")
        print(f"\n  {i}. {title}  (match score: {score})")

        # Show ingredients used (Gemini mode)
        if "ingredients_used" in recipe:
            print(f"     Uses from fridge: {', '.join(recipe['ingredients_used'])}")
        if "extra_ingredients_needed" in recipe:
            print(f"     Need to buy:      {', '.join(recipe['extra_ingredients_needed'])}")

        # Show nutrition per serving (Gemini mode)
        nutr = recipe.get("nutrition_per_serving", {})
        if nutr:
            print(
                f"     Per serving: {nutr.get('calories_kcal', '?')} kcal | "
                f"protein {nutr.get('protein_g', '?')}g | "
                f"carbs {nutr.get('carbs_g', '?')}g | "
                f"fat {nutr.get('fat_g', '?')}g"
            )

        # Show steps
        steps = recipe.get("steps", [])
        if steps:
            print(f"     Steps ({len(steps)}):")
            for j, step in enumerate(steps[:3], 1):
                print(f"       {j}. {step[:100]}...")
            if len(steps) > 3:
                print(f"       ... +{len(steps) - 3} more steps")

    # Nutrition summary per ingredient
    print("\nNUTRITION SUMMARY (per 100g):")
    for info in output.get("nutrition_summary", []):
        if "note" in info:
            print(f"  - {info['ingredient']}: {info['note']}")
        else:
            print(
                f"  - {info['ingredient']}: "
                f"{info.get('calories_per_100g', '?')} kcal | "
                f"protein {info.get('protein_per_100g', '?')}g | "
                f"carbs {info.get('carbs_per_100g', '?')}g | "
                f"fat {info.get('fat_per_100g', '?')}g"
            )

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# Class wrapper (used by app.py and Gradio)
# ---------------------------------------------------------------------------

class RecipeEngine:
    """
    Drop-in replacement for app.py.
    Tries Gemini first, falls back to RecipeNLG.
    """

    def __init__(
        self,
        recipes_path: str = "/home/codespace/.cache/kagglehub/datasets/saldenisov/recipenlg/versions/1",
        nutrition_path: str = "/home/codespace/.cache/kagglehub/datasets/openfoodfacts/world-food-facts/versions/5",
        top_k: int = 3,
    ):
        self.recipes_path   = recipes_path
        self.nutrition_path = nutrition_path
        self.top_k          = top_k

    def generate(
        self,
        ingredients: list[str],
        summary: list[dict] | None = None,
    ) -> str:
        """
        Generate recipes and return formatted text for Gradio UI.

        Args:
            ingredients: list of ingredient names (sorted by relevance)
            summary:     optional infer.py summary with count/relevance
        """
        output = run(
            ingredients=ingredients,
            summary=summary,
            top_k=self.top_k,
            recipes_path=self.recipes_path,
            nutrition_path=self.nutrition_path,
        )

        lines = []
        source = output.get("source", "unknown")
        lines.append(f"Source: {source}\n")

        for i, recipe in enumerate(output.get("recipes", []), 1):
            title = recipe.get("title", "Untitled")
            lines.append(f"{'─'*40}")
            lines.append(f"🍽️  Recipe {i}: {title}")

            if "ingredients_used" in recipe:
                lines.append(f"   ✅ Uses: {', '.join(recipe['ingredients_used'])}")
            if "extra_ingredients_needed" in recipe:
                lines.append(f"   🛒 Need: {', '.join(recipe['extra_ingredients_needed'])}")

            # Nutrition per serving
            nutr = recipe.get("nutrition_per_serving", {})
            if nutr:
                lines.append(
                    f"   📊 Per serving: "
                    f"{nutr.get('calories_kcal', '?')} kcal | "
                    f"P {nutr.get('protein_g', '?')}g | "
                    f"C {nutr.get('carbs_g', '?')}g | "
                    f"F {nutr.get('fat_g', '?')}g"
                )

            # Difficulty & time
            if "difficulty" in recipe:
                time_str = ""
                if "prep_time_min" in recipe:
                    time_str += f"prep {recipe['prep_time_min']}min"
                if "cook_time_min" in recipe:
                    time_str += f" + cook {recipe['cook_time_min']}min"
                lines.append(f"   ⏱️  {recipe['difficulty'].capitalize()} — {time_str}")

            # Steps
            steps = recipe.get("steps", [])
            if steps:
                lines.append(f"   Steps:")
                for j, step in enumerate(steps, 1):
                    lines.append(f"     {j}. {step}")
            lines.append("")

        # Nutrition summary per ingredient
        nutr_summary = output.get("nutrition_summary", [])
        if nutr_summary:
            lines.append(f"{'─'*40}")
            lines.append("📋 Ingredient Nutrition (per 100g):")
            for info in nutr_summary:
                if "note" in info:
                    lines.append(f"   {info['ingredient']}: {info['note']}")
                else:
                    lines.append(
                        f"   {info['ingredient']}: "
                        f"{info.get('calories_per_100g', '?')} kcal | "
                        f"P {info.get('protein_per_100g', '?')}g | "
                        f"C {info.get('carbs_per_100g', '?')}g | "
                        f"F {info.get('fat_per_100g', '?')}g"
                    )

        return "\n".join(lines) if lines else "No recipes found for these ingredients."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    ingredients = [ing.strip() for ing in args.ingredients.split(",")]

    # Parse optional summary JSON
    summary = None
    if args.summary_json:
        try:
            summary = json.loads(args.summary_json)
        except json.JSONDecodeError:
            print("[WARN] Could not parse --summary_json, ignoring.")

    output = run(
        ingredients=ingredients,
        summary=summary,
        top_k=args.top_k,
        force_fallback=args.fallback,
        recipes_path=args.recipes_path,
        nutrition_path=args.nutrition_path,
    )

    print_output(output)

    # Save output as JSON
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/recipe_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput saved to outputs/recipe_output.json")