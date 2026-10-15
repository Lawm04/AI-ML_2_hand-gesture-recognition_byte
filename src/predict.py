from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/hand_gesture_cnn.keras")
LABEL_PATH = Path("models/class_names.json")
IMAGE_SIZE = (64, 64)

if len(sys.argv) != 2:
    print("Usage: python src/predict.py path/to/image.png")
    raise SystemExit(1)

image_path = Path(sys.argv[1])

if not MODEL_PATH.exists() or not LABEL_PATH.exists():
    raise SystemExit("Model or class labels missing. Run Step 5 first.")

if not image_path.is_file():
    raise SystemExit(f"Image not found: {image_path}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as file:
    class_names = json.load(file)

with Image.open(image_path) as image:
    image = image.convert("L").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0

image_array = image_array[np.newaxis, ..., np.newaxis]

probabilities = model.predict(image_array, verbose=0)[0]
best_index = int(np.argmax(probabilities))

print(f"\nImage: {image_path}")
print(f"Predicted gesture: {class_names[best_index]}")
print(f"Model confidence: {probabilities[best_index]:.2%}")

print("\nTop predictions:")
for index in np.argsort(probabilities)[::-1][:3]:
    print(f"  {class_names[index]}: {probabilities[index]:.2%}")
