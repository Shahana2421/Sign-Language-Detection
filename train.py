
"""
2_train_model.py
CNN Model Training for Sign Language Detection

Gestures:
Hello, No, Please, Sorry, Thank You, Yes, I Love You

Run:
python 2_train_model.py
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =====================================================
# 1. CONFIGURATION
# =====================================================

DATASET_DIR = "dataset"

MODEL_PATH = "sign_model.h5"
LABEL_PATH = "class_labels.npy"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

SEED = 42

# Reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================
# 2. BUILD CNN MODEL
# =====================================================

def build_cnn(num_classes):

    model = models.Sequential([

        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Block 1
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(2, 2),

        # Classifier
        layers.Flatten(),

        layers.Dropout(0.5),

        layers.Dense(
            256,
            activation="relu"
        ),

        # Output layer
        layers.Dense(
            num_classes,
            activation="softmax"
        )

    ])

    return model


# =====================================================
# 3. TRAINING FUNCTION
# =====================================================

def train():

    print("\n===================================")
    print(" SIGN LANGUAGE CNN MODEL TRAINING ")
    print("===================================\n")

    # Check dataset directory
    if not os.path.exists(DATASET_DIR):

        raise FileNotFoundError(
            f"Dataset folder '{DATASET_DIR}' not found!"
        )

    # =============================================
    # 4. DATA AUGMENTATION
    # =============================================

    print("Preparing training and validation data...")

    train_datagen = ImageDataGenerator(

        rescale=1.0 / 255,

        validation_split=0.2,

        rotation_range=15,

        width_shift_range=0.1,

        height_shift_range=0.1,

        zoom_range=0.1

    )

    # Validation data: rescaling only
    val_datagen = ImageDataGenerator(

        rescale=1.0 / 255,

        validation_split=0.2

    )

    # =============================================
    # 5. LOAD TRAINING DATA
    # =============================================

    train_gen = train_datagen.flow_from_directory(

        DATASET_DIR,

        target_size=(IMG_SIZE, IMG_SIZE),

        batch_size=BATCH_SIZE,

        subset="training",

        class_mode="categorical",

        color_mode="rgb",

        shuffle=True,

        seed=SEED

    )

    # =============================================
    # 6. LOAD VALIDATION DATA
    # =============================================

    val_gen = val_datagen.flow_from_directory(

        DATASET_DIR,

        target_size=(IMG_SIZE, IMG_SIZE),

        batch_size=BATCH_SIZE,

        subset="validation",

        class_mode="categorical",

        color_mode="rgb",

        shuffle=False,

        seed=SEED

    )

    # =============================================
    # 7. CHECK DATASET
    # =============================================

    num_classes = len(train_gen.class_indices)

    print("\nClasses found:")

    for name, index in train_gen.class_indices.items():

        print(f"{index} -> {name}")

    print("\nTotal training images:", train_gen.samples)

    print("Total validation images:", val_gen.samples)

    print("Number of classes:", num_classes)

    if num_classes < 2:

        raise ValueError(
            "At least two gesture classes are required."
        )

    # =============================================
    # 8. BUILD CNN MODEL
    # =============================================

    print("\nBuilding CNN model...")

    model = build_cnn(num_classes)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    model.summary()

    # =============================================
    # 9. CALLBACKS
    # =============================================

    callbacks = [

        # Stop training if validation loss does not improve
        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True

        ),

        # Save the best model
        tf.keras.callbacks.ModelCheckpoint(

            MODEL_PATH,

            monitor="val_loss",

            save_best_only=True,

            verbose=1

        ),

        # Reduce learning rate when progress stops
        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=3,

            min_lr=0.00001,

            verbose=1

        )

    ]

    # =============================================
    # 10. TRAIN MODEL
    # =============================================

    print("\nStarting CNN training...")

    history = model.fit(

        train_gen,

        validation_data=val_gen,

        epochs=EPOCHS,

        callbacks=callbacks

    )

    # =============================================
    # 11. SAVE CLASS LABELS
    # =============================================

    labels = {

        index: name

        for name, index in train_gen.class_indices.items()

    }

    np.save(

        LABEL_PATH,

        labels

    )

    print("\nClass labels saved successfully!")

    # =============================================
    # 12. EVALUATE MODEL
    # =============================================

    print("\nEvaluating model...")

    val_loss, val_accuracy = model.evaluate(

        val_gen,

        verbose=1

    )

    print("\n===================================")

    print(" TRAINING COMPLETED ")

    print("===================================")

    print(f"Validation Loss: {val_loss:.4f}")

    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    print(f"\nModel saved: {MODEL_PATH}")

    print(f"Labels saved: {LABEL_PATH}")

    # =============================================
    # 13. TRAINING GRAPHS
    # =============================================

    plt.figure(figsize=(12, 5))

    # Accuracy graph
    plt.subplot(1, 2, 1)

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("CNN Model Accuracy")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend()

    # Loss graph
    plt.subplot(1, 2, 2)

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("CNN Model Loss")

    plt.xlabel("Epochs")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig("training_results.png")

    plt.show()

    print("\nTraining graphs saved to training_results.png")

    print("\nNext step: Run your real-time detection program.")


# =====================================================
# 14. MAIN
# =====================================================

if __name__ == "__main__":

    train()
