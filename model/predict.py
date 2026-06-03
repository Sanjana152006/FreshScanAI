"""
model/predict.py
Standalone prediction script for FreshScan AI.
Use this to test a single image from the command line without running Django.

Usage:
    python model/predict.py --image path/to/your/image.jpg
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path so we can import settings-independent logic
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Class labels (must match training order) ──
CLASS_LABELS = [
    'fresh_fruits',
    'fresh_vegetables',
    'rotten_fruits',
    'rotten_vegetables',
]

# ── Shelf life map ──
SHELF_LIFE_MAP = {
    'Fresh':      {'days': 5,  'hours': 12},
    'Semi-Fresh': {'days': 1,  'hours': 8},
    'Rotten':     {'days': 0,  'hours': 0},
}

# ── Storage suggestions ──
STORAGE_SUGGESTIONS = {
    'Fresh':      'Store in a cool, dry place or refrigerate to extend freshness.',
    'Semi-Fresh': 'Consume within 24–48 hours. Refrigerate immediately.',
    'Rotten':     '❌ UNSAFE. Discard immediately. Do not consume.',
}


def load_model(model_path: str):
    """Load the trained Keras model from disk."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"✅  Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌  Failed to load model: {e}")
        sys.exit(1)


def preprocess_image(image_path: str, target_size=(224, 224)) -> np.ndarray:
    """
    Read an image with OpenCV, resize, normalise and add batch dimension.
    Returns array of shape (1, 224, 224, 3).
    """
    img = cv2.imread(image_path)

    if img is None:
        # Fallback: try PIL
        from PIL import Image
        pil = Image.open(image_path).convert('RGB')
        img = np.array(pil)[:, :, ::-1]   # RGB → BGR for OpenCV convention

    img_resized   = cv2.resize(img, target_size)
    img_rgb       = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm      = img_rgb.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0)


def predict(model, image_path: str) -> dict:
    """
    Run inference and return a structured result dictionary.
    """
    img_batch = preprocess_image(image_path)
    raw_preds = model.predict(img_batch, verbose=0)[0]

    best_idx    = int(np.argmax(raw_preds))
    best_label  = CLASS_LABELS[best_idx]
    confidence  = float(raw_preds[best_idx]) * 100

    # Map label → freshness state
    if 'fresh' in best_label:
        freshness   = 'Fresh'
        f_score     = min(confidence, 98.0)
    else:
        freshness   = 'Rotten'
        f_score     = max(100.0 - confidence, 5.0)

    # Semi-fresh band (60–74 %)
    if 60.0 <= f_score < 75.0:
        freshness = 'Semi-Fresh'

    # Category
    category = 'fruit' if 'fruit' in best_label else 'vegetable'

    # Shelf life
    sl          = SHELF_LIFE_MAP[freshness]
    days        = sl['days']
    hours       = sl['hours']

    # Fine-tune days
    if freshness == 'Fresh':
        if f_score > 90: days, hours = 7,  0
        elif f_score > 80: days, hours = 5, 12
        else: days, hours = 3, 6
    elif freshness == 'Semi-Fresh':
        days, hours = (2, 0) if f_score > 60 else (1, 0)

    expiry_date = (datetime.now() + timedelta(days=days, hours=hours)).date()
    suggestion  = STORAGE_SUGGESTIONS[freshness]

    # All class probabilities
    all_probs = {
        CLASS_LABELS[i]: round(float(raw_preds[i]) * 100, 2)
        for i in range(len(CLASS_LABELS))
    }

    return {
        'label':          best_label,
        'freshness':      freshness,
        'freshness_score': round(f_score, 1),
        'confidence':     round(confidence, 1),
        'category':       category,
        'shelf_life_days': days,
        'shelf_life_hours': hours,
        'expiry_date':    str(expiry_date),
        'suggestion':     suggestion,
        'all_probabilities': all_probs,
    }


def print_result(result: dict, image_path: str):
    """Pretty-print prediction result to console."""
    bar_len = 50
    score   = result['freshness_score']
    bar     = '█' * int(bar_len * score / 100) + '░' * (bar_len - int(bar_len * score / 100))

    emoji_map = {'Fresh': '✅', 'Semi-Fresh': '⚠️', 'Rotten': '❌'}
    emoji = emoji_map.get(result['freshness'], '❓')

    print("\n" + "=" * 60)
    print("  🌿  FRESHS CAN AI  –  PREDICTION RESULT")
    print("=" * 60)
    print(f"  Image        : {image_path}")
    print(f"  Category     : {result['category'].capitalize()}")
    print(f"  Freshness    : {emoji}  {result['freshness']}")
    print(f"  Score        : {score}%")
    print(f"  [{bar}]")
    print(f"  Confidence   : {result['confidence']}%")
    print(f"  Shelf Life   : {result['shelf_life_days']}d  {result['shelf_life_hours']}h")
    print(f"  Expiry Date  : {result['expiry_date']}")
    print(f"  Storage Tip  : {result['suggestion']}")
    print("\n  All Class Probabilities:")
    for label, prob in result['all_probabilities'].items():
        bar2 = '█' * int(30 * prob / 100) + '░' * (30 - int(30 * prob / 100))
        print(f"    {label:<25} {prob:5.1f}%  [{bar2}]")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='FreshScan AI – Predict food freshness from an image.'
    )
    parser.add_argument(
        '--image', '-i',
        required=True,
        help='Path to the food image (JPG / PNG / JPEG)'
    )
    parser.add_argument(
        '--model', '-m',
        default=str(BASE_DIR / 'freshness_model.h5'),
        help='Path to the trained Keras model (default: freshness_model.h5)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌  Image not found: {args.image}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"❌  Model not found: {args.model}")
        print("    Train the model first:  python model/train_model.py")
        sys.exit(1)

    model  = load_model(args.model)
    result = predict(model, args.image)
    print_result(result, args.image)


if __name__ == '__main__':
    main()