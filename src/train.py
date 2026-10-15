from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

MANIFEST = Path("data/processed/dataset_split.csv")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 12
SEED = 42

if not MANIFEST.exists():
    raise SystemExit(
        "Dataset manifest not found. Run: python src/prepare_dataset.py"
    )

df = pd.read_csv(MANIFEST)

required = {"image_path", "gesture", "subject", "split"}
if not required.issubset(df.columns):
    raise SystemExit(f"CSV must contain these columns: {required}")

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

if train_df.empty or test_df.empty:
    raise SystemExit("Training or testing split is empty.")

# Encode labels consistently for both datasets.
encoder = LabelEncoder()
encoder.fit(df["gesture"])
class_names = encoder.classes_.tolist()

def load_images(frame):
    images = []
    labels = []

    for row in frame.itertuples(index=False):
        path = Path(row.image_path)
        try:
            with Image.open(path) as image:
                image = image.convert("L").resize(IMAGE_SIZE)
                images.append(np.asarray(image, dtype=np.uint8))
                labels.append(encoder.transform([row.gesture])[0])
        except Exception as error:
            print(f"Skipping unreadable image: {path} ({error})")

    if not images:
        raise SystemExit("No images could be loaded.")

    x = np.stack(images)[..., np.newaxis].astype("float32") / 255.0
    y = np.asarray(labels, dtype="int64")
    return x, y

print("Loading training images...")
X_train, y_train = load_images(train_df)

print("Loading testing images...")
X_test, y_test = load_images(test_df)

print(f"Training images: {len(X_train)}")
print(f"Testing images: {len(X_test)}")
print(f"Gesture classes: {class_names}")

tf.keras.utils.set_random_seed(SEED)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
    verbose=1,
)

# Evaluate on subjects not used for training.
probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
predictions = probabilities.argmax(axis=1)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0,
)

model.save(MODEL_DIR / "hand_gesture_cnn.keras")

with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file, indent=2)

with open(RESULTS_DIR / "classification_report.txt", "w") as file:
    file.write(f"Test accuracy: {accuracy:.4f}\n\n")
    file.write(report)

matrix = confusion_matrix(
    y_test, predictions, labels=np.arange(len(class_names))
)
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix, display_labels=class_names
)
fig, ax = plt.subplots(figsize=(11, 9))
display.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.title("Hand Gesture Recognition — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training History")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_history.png", dpi=160)
plt.close()

print(f"\nTest accuracy: {accuracy:.4f}")
print("\nClassification report:\n", report)
print("\nSaved model: models/hand_gesture_cnn.keras")
print("Saved class labels: models/class_names.json")
print("Saved metrics: results/classification_report.txt")
print("Saved confusion matrix: results/confusion_matrix.png")
print("Saved training plot: results/training_history.png")
