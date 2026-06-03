"""
model/train_model.py
CNN model training script for FreshScan AI.
Trains a MobileNetV2-based transfer learning model for food freshness detection.

Usage:
    cd FreshScanAI/
    python model/train_model.py

Dataset structure expected:
    dataset/
    ├── train/
    │   ├── fresh_fruits/
    │   ├── rotten_fruits/
    │   ├── fresh_vegetables/
    │   └── rotten_vegetables/
    └── test/
        ├── fresh_fruits/
        ├── rotten_fruits/
        ├── fresh_vegetables/
        └── rotten_vegetables/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train'
TEST_DIR = DATASET_DIR / 'test'
MODEL_SAVE_PATH = BASE_DIR / 'freshness_model.h5'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
NUM_CLASSES = 4

CLASS_NAMES = ['fresh_fruits', 'fresh_vegetables', 'rotten_fruits', 'rotten_vegetables']


def create_data_generators():
    """Create training and validation data generators with augmentation."""

    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2,
    )

    # Test data (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    print("📂 Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
    )

    print("📂 Loading validation data...")
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
    )

    print("📂 Loading test data...")
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


def build_model():
    """Build CNN model using MobileNetV2 transfer learning."""

    print("🏗️ Building model with MobileNetV2...")

    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print(f"✅ Model built. Total parameters: {model.count_params():,}")
    model.summary()

    return model, base_model


def train_model():
    """Full training pipeline."""

    # Check dataset exists
    if not TRAIN_DIR.exists():
        print(f"❌ Dataset not found at {TRAIN_DIR}")
        print("Download dataset from: https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables")
        print("Place in dataset/train/ and dataset/test/ folders")
        sys.exit(1)

    # Create generators
    train_gen, val_gen, test_gen = create_data_generators()

    print(f"📊 Training samples: {train_gen.samples}")
    print(f"📊 Validation samples: {val_gen.samples}")
    print(f"📊 Test samples: {test_gen.samples}")
    print(f"📊 Classes: {train_gen.class_indices}")

    # Build model
    model, base_model = build_model()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(MODEL_SAVE_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(log_dir='logs/'),
    ]

    # ── Phase 1: Train with frozen base ──
    print("\n🚀 Phase 1: Training classification head...")
    history1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # ── Phase 2: Fine-tune top layers of base model ──
    print("\n🚀 Phase 2: Fine-tuning top layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=10,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluate on test set ──
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)[:2]
    print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
    print(f"✅ Test Loss: {test_loss:.4f}")

    # Classification report
    test_gen.reset()
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Save final model
    model.save(str(MODEL_SAVE_PATH))
    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")

    # Plot training history
    plot_training_history(history1, history2)


def plot_training_history(history1, history2):
    """Plot and save training accuracy/loss graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FreshScan AI – Training History', fontsize=16, fontweight='bold')

    # Combine histories
    acc1 = history1.history.get('accuracy', [])
    val_acc1 = history1.history.get('val_accuracy', [])
    acc2 = history2.history.get('accuracy', [])
    val_acc2 = history2.history.get('val_accuracy', [])
    acc = acc1 + acc2
    val_acc = val_acc1 + val_acc2

    loss1 = history1.history.get('loss', [])
    val_loss1 = history1.history.get('val_loss', [])
    loss2 = history2.history.get('loss', [])
    val_loss2 = history2.history.get('val_loss', [])
    loss = loss1 + loss2
    val_loss = val_loss1 + val_loss2

    epochs_range = range(len(acc))

    # Accuracy
    axes[0].plot(epochs_range, acc, '#00C853', linewidth=2, label='Train Accuracy')
    axes[0].plot(epochs_range, val_acc, '#FF6D00', linewidth=2, linestyle='--', label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=len(acc1), color='blue', linestyle=':', alpha=0.7, label='Fine-tune start')

    # Loss
    axes[1].plot(epochs_range, loss, '#D32F2F', linewidth=2, label='Train Loss')
    axes[1].plot(epochs_range, val_loss, '#7B1FA2', linewidth=2, linestyle='--', label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = BASE_DIR / 'static' / 'images' / 'training_history.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"📊 Training plot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    train_model()