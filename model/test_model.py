"""
model/test_model.py
Comprehensive evaluation script for the trained FreshScan AI model.
Generates accuracy report, confusion matrix, and per-class metrics.

Usage:
    python model/test_model.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe on servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
TEST_DIR   = BASE_DIR / 'dataset' / 'test'
MODEL_PATH = BASE_DIR / 'freshness_model.h5'
OUT_DIR    = BASE_DIR / 'static' / 'images'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['fresh_fruits', 'fresh_vegetables', 'rotten_fruits', 'rotten_vegetables']
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32


def load_model():
    import tensorflow as tf
    if not MODEL_PATH.exists():
        print(f"❌  Model not found at {MODEL_PATH}")
        print("    Run  python model/train_model.py  first.")
        sys.exit(1)
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"✅  Model loaded  ({MODEL_PATH})")
    return model


def build_test_generator():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    if not TEST_DIR.exists():
        print(f"❌  Test dataset not found at {TEST_DIR}")
        sys.exit(1)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )
    print(f"📂  Test samples  : {gen.samples}")
    print(f"📂  Class indices : {gen.class_indices}")
    return gen


def evaluate(model, gen):
    """Run model.evaluate and return loss + accuracy."""
    results = model.evaluate(gen, verbose=1)
    # results = [loss, accuracy, ...]
    loss = results[0]
    acc  = results[1]
    print(f"\n📊  Test Loss     : {loss:.4f}")
    print(f"📊  Test Accuracy : {acc * 100:.2f}%")
    return loss, acc


def generate_predictions(model, gen):
    """Return true labels and predicted labels."""
    gen.reset()
    y_pred_probs = model.predict(gen, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = gen.classes
    return y_true, y_pred, y_pred_probs


def plot_confusion_matrix(y_true, y_pred):
    """Save a styled confusion matrix PNG."""
    from sklearn.metrics import confusion_matrix
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FreshScan AI – Confusion Matrix', fontsize=15, fontweight='bold')

    for ax, data, title, fmt in zip(
        axes,
        [cm, norm],
        ['Raw Counts', 'Normalised (%)'],
        ['d', '.2f']
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor='white', ax=ax,
            annot_kws={'size': 10}
        )
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    save_path = OUT_DIR / 'confusion_matrix.png'
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"💾  Confusion matrix saved → {save_path}")
    plt.close()


def plot_class_accuracy(y_true, y_pred):
    """Bar chart of per-class accuracy."""
    from sklearn.metrics import confusion_matrix
    cm        = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1) * 100

    colours = ['#00C853', '#64DD17', '#D32F2F', '#B71C1C']
    fig, ax  = plt.subplots(figsize=(8, 5))
    bars     = ax.bar(CLASS_NAMES, per_class, color=colours, edgecolor='white', linewidth=1.5)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha='right')

    for bar, val in zip(bars, per_class):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold'
        )

    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_path = OUT_DIR / 'class_accuracy.png'
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"💾  Class accuracy chart saved → {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print("\n📋  Classification Report:")
    print("─" * 60)
    print(report)
    # Save report
    rpt_path = OUT_DIR / 'classification_report.txt'
    rpt_path.write_text(report)
    print(f"💾  Report saved → {rpt_path}")


def sample_predictions(model, gen, n=8):
    """Show a grid of sample predictions vs ground truth."""
    import tensorflow as tf

    gen.reset()
    images, labels = next(gen)
    preds          = model.predict(images[:n], verbose=0)
    pred_classes   = np.argmax(preds, axis=1)
    true_classes   = np.argmax(labels[:n], axis=1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Sample Predictions', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis('off')
            continue
        ax.imshow(images[i])
        pred_lbl = CLASS_NAMES[pred_classes[i]]
        true_lbl = CLASS_NAMES[true_classes[i]]
        correct  = pred_classes[i] == true_classes[i]
        color    = '#00C853' if correct else '#D32F2F'
        ax.set_title(
            f"Pred: {pred_lbl}\nTrue: {true_lbl}\nConf: {preds[i][pred_classes[i]]*100:.1f}%",
            fontsize=8, color=color, fontweight='bold'
        )
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.tight_layout()
    save_path = OUT_DIR / 'sample_predictions.png'
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"💾  Sample predictions saved → {save_path}")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("  🌿  FRESHS CAN AI  –  MODEL EVALUATION")
    print("=" * 60 + "\n")

    model = load_model()
    gen   = build_test_generator()

    # Evaluate
    loss, acc = evaluate(model, gen)

    # Predictions
    y_true, y_pred, y_probs = generate_predictions(model, gen)

    # Reports & charts
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_class_accuracy(y_true, y_pred)
    sample_predictions(model, gen)

    print("\n✅  Evaluation complete!")
    print(f"    Overall Accuracy : {acc * 100:.2f}%")
    print(f"    Charts saved to  : {OUT_DIR}\n")


if __name__ == '__main__':
    main()