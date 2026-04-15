"""
recipe_engine.py — FridgeAI
Given a list of detected ingredients, finds matching recipes and nutritional info.

Uses:
- RecipeNLG dataset: to find recipes by ingredients
- Open Food Facts dataset: to get nutritional info (calories, protein, carbs, fat)

Usage:
    python src/recipe_engine.py --ingredients "chicken,tomato,egg"
"""

import argparse
import pandas as pd
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Find recipes from detected ingredients")
    parser.add_argument(
        "--ingredients",
        type=str,
        required=True,
        help="Comma-separated list of detected ingredients e.g. 'chicken,tomato,egg'",
    )
    parser.add_argument(
        "--recipes_path",
        type=str,
        default="/home/codespace/.cache/kagglehub/datasets/saldenisov/recipenlg/versions/1",
        help="Path to RecipeNLG dataset",
    )
    parser.add_argument(
        "--nutrition_path",
        type=str,
        default="/home/codespace/.cache/kagglehub/datasets/openfoodfacts/world-food-facts/versions/5",
        help="Path to Open Food Facts dataset",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of recipes to return",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Recipe search — RecipeNLG
# ---------------------------------------------------------------------------

def load_recipes(recipes_path):
    """
    Load RecipeNLG dataset.
    The dataset is a CSV with columns: title, ingredients, directions, etc.
    """
    csv_path = os.path.join(recipes_path, 'full_dataset.csv')
    if not os.path.exists(csv_path):
        # Try to find it
        for f in Path(recipes_path).rglob('*.csv'):
            csv_path = str(f)
            break

    print(f"Loading recipes from: {csv_path}")
    df = pd.read_csv(csv_path, nrows=50000)  # Load first 50k for speed
    return df


def find_recipes(ingredients: list[str], recipes_df, top_k: int = 3):
    """
    Find recipes that use the most detected ingredients.

    Args:
        ingredients: list of detected ingredient names e.g. ["chicken", "tomato"]
        recipes_df:  RecipeNLG dataframe
        top_k:       how many recipes to return

    Returns:
        list of dicts with recipe info
    """
    ingredients_lower = [ing.lower().replace('-', ' ') for ing in ingredients]

    # Score each recipe by how many detected ingredients it contains
    def score_recipe(row):
        try:
            recipe_ingredients = str(row['ingredients']).lower()
            matches = sum(1 for ing in ingredients_lower if ing in recipe_ingredients)
            return matches
        except:
            return 0

    print("Searching recipes...")
    recipes_df['match_score'] = recipes_df.apply(score_recipe, axis=1)

    # Get top matches
    top_recipes = recipes_df[recipes_df['match_score'] > 0] \
        .sort_values('match_score', ascending=False) \
        .head(top_k)

    results = []
    for _, row in top_recipes.iterrows():
        results.append({
            'title':       row.get('title', 'Unknown'),
            'ingredients': row.get('ingredients', ''),
            'directions':  row.get('directions', ''),
            'match_score': int(row['match_score']),
        })

    return results


# ---------------------------------------------------------------------------
# Nutrition lookup — Open Food Facts
# ---------------------------------------------------------------------------

def load_nutrition(nutrition_path):
    """
    Load Open Food Facts dataset.
    Contains nutritional info per 100g: calories, protein, carbs, fat.
    """
    # Find the TSV or CSV file
    tsv_path = None
    for f in Path(nutrition_path).rglob('*.tsv'):
        tsv_path = str(f)
        break
    for f in Path(nutrition_path).rglob('*.csv'):
        tsv_path = str(f)
        break

    if not tsv_path:
        print("Open Food Facts file not found, skipping nutrition lookup.")
        return None

    print(f"Loading nutrition data from: {tsv_path}")

    # Load only relevant columns for speed
    cols = ['product_name', 'energy_100g', 'proteins_100g', 'carbohydrates_100g', 'fat_100g']
    try:
        df = pd.read_csv(tsv_path, sep='\t', usecols=cols, nrows=100000, low_memory=False)
        df = df.dropna(subset=['product_name'])
        return df
    except Exception as e:
        print(f"Could not load nutrition data: {e}")
        return None


def get_nutrition(ingredient: str, nutrition_df):
    """
    Look up nutritional info for a single ingredient.

    Args:
        ingredient:    ingredient name e.g. "chicken"
        nutrition_df:  Open Food Facts dataframe

    Returns:
        dict with nutritional info per 100g, or None if not found
    """
    if nutrition_df is None:
        return None

    ingredient_lower = ingredient.lower().replace('-', ' ')

    # Search by product name
    matches = nutrition_df[
        nutrition_df['product_name'].str.lower().str.contains(ingredient_lower, na=False)
    ].head(1)

    if matches.empty:
        return None

    row = matches.iloc[0]
    return {
        'ingredient':     ingredient,
        'calories_100g':  round(float(row.get('energy_100g', 0) or 0) / 4.184, 1),  # kJ to kcal
        'protein_100g':   round(float(row.get('proteins_100g', 0) or 0), 1),
        'carbs_100g':     round(float(row.get('carbohydrates_100g', 0) or 0), 1),
        'fat_100g':       round(float(row.get('fat_100g', 0) or 0), 1),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(ingredients: list[str], recipes_path: str, nutrition_path: str, top_k: int = 3):
    """
    Full pipeline: ingredients → recipes + nutrition.

    Args:
        ingredients:    list of detected ingredient names
        recipes_path:   path to RecipeNLG dataset
        nutrition_path: path to Open Food Facts dataset
        top_k:          number of recipes to return

    Returns:
        dict with recipes and nutritional info
    """
    print(f"\n=== FridgeAI Recipe Engine ===")
    print(f"Detected ingredients: {ingredients}\n")

    # 1. Find recipes
    recipes_df = load_recipes(recipes_path)
    recipes    = find_recipes(ingredients, recipes_df, top_k)

    # 2. Get nutrition for each ingredient
    nutrition_df = load_nutrition(nutrition_path)
    nutrition    = []
    for ing in ingredients:
        info = get_nutrition(ing, nutrition_df)
        if info:
            nutrition.append(info)
        else:
            nutrition.append({'ingredient': ing, 'note': 'not found in database'})

    # 3. Build output
    output = {
        'ingredients_detected': ingredients,
        'recipes':   recipes,
        'nutrition': nutrition,
    }

    return output


def print_output(output):
    """Pretty print the results."""
    print("\n" + "="*50)
    print("DETECTED INGREDIENTS:")
    for ing in output['ingredients_detected']:
        print(f"  - {ing}")

    print("\nRECIPES:")
    for i, recipe in enumerate(output['recipes'], 1):
        print(f"\n  {i}. {recipe['title']} (matches: {recipe['match_score']} ingredients)")
        try:
            directions = json.loads(recipe['directions'])
            if isinstance(directions, list):
                print(f"     Steps: {len(directions)} steps")
                print(f"     First step: {directions[0][:100]}...")
        except:
            pass

    print("\nNUTRITION (per 100g):")
    for info in output['nutrition']:
        if 'note' in info:
            print(f"  - {info['ingredient']}: {info['note']}")
        else:
            print(f"  - {info['ingredient']}: "
                  f"{info['calories_100g']} kcal | "
                  f"protein: {info['protein_100g']}g | "
                  f"carbs: {info['carbs_100g']}g | "
                  f"fat: {info['fat_100g']}g")
    print("="*50)

# ---------------------------------------------------------------------------
# Class wrapper (used by app.py)
# ---------------------------------------------------------------------------

class RecipeEngine:
    def __init__(
        self,
        recipes_path: str = "/home/codespace/.cache/kagglehub/datasets/saldenisov/recipenlg/versions/1",
        nutrition_path: str = "/home/codespace/.cache/kagglehub/datasets/openfoodfacts/world-food-facts/versions/5",
        top_k: int = 3,
    ):
        self.recipes_path   = recipes_path
        self.nutrition_path = nutrition_path
        self.top_k          = top_k

    def generate(self, ingredients: list[str]) -> str:
        output = run(
            ingredients=ingredients,
            recipes_path=self.recipes_path,
            nutrition_path=self.nutrition_path,
            top_k=self.top_k,
        )
        # Format as readable text for the Gradio UI
        lines = []
        for i, recipe in enumerate(output['recipes'], 1):
            lines.append(f"{i}. {recipe['title']}  (matches {recipe['match_score']} ingredients)")
            try:
                directions = json.loads(recipe['directions'])
                if isinstance(directions, list) and directions:
                    lines.append(f"   → {directions[0][:120]}...")
            except:
                pass
            lines.append("")
        return "\n".join(lines) if lines else "No recipes found for these ingredients."



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    ingredients = [ing.strip() for ing in args.ingredients.split(',')]

    output = run(
        ingredients=ingredients,
        recipes_path=args.recipes_path,
        nutrition_path=args.nutrition_path,
        top_k=args.top_k,
    )

    print_output(output)

    # Save output as JSON
    with open('outputs/recipe_output.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput saved to outputs/recipe_output.json")