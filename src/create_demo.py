from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MODEL_PATH = Path("models/hand_gesture_cnn.keras")
LABEL_PATH = Path("models/class_names.json")
MANIFEST_PATH = Path("data/processed/dataset_split.csv")
OUTPUT_PATH = Path("results/demo/prediction_gallery.png")
IMAGE_SIZE = (64, 64)

if not MODEL_PATH.exists() or not LABEL_PATH.exists():
    raise SystemExit("Trained model or class labels not found. Finish Step 5.")

if not MANIFEST_PATH.exists():
    raise SystemExit("Dataset manifest not found. Finish Step 4.")

import pandas as pd

df = pd.read_csv(MANIFEST_PATH)
test_df = df[df["split"] == "test"]

if test_df.empty:
    raise SystemExit("No test images found in the manifest.")

# Pick up to 8 test images, covering different gesture classes.
samples = (
    test_df.groupby("gesture", group_keys=False)
    .head(1)
    .head(8)
)

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    class_names = json.load(f)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for ax in axes:
    ax.axis("off")

for ax, row in zip(axes, samples.itertuples(index=False)):
    path = Path(row.image_path)

    with Image.open(path) as im:
        display_image = im.convert("L")
        display_image = display_image.resize(IMAGE_SIZE)
        arr = np.asarray(display_image, dtype=np.float32) / 255.0

    probabilities = model.predict(
        arr[np.newaxis, ..., np.newaxis], verbose=0
    )[0]

    predicted = class_names[int(np.argmax(probabilities))]
    confidence = float(np.max(probabilities))

    ax.imshow(arr.squeeze(), cmap="gray")
    ax.set_title(
        f"Actual: {row.gesture}\nPredicted: {predicted}\n"
        f"Confidence: {confidence:.1%}",
        fontsize=9
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
plt.close()

print(f"Demo gallery saved to: {OUTPUT_PATH}")
print(f"Images displayed: {len(samples)}")
