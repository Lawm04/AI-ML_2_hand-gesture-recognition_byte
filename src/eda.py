
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# 1. Set paths
# --------------------------------------------------

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# --------------------------------------------------
# 2. Discover labeled images
# --------------------------------------------------

records = []

for path in DATA_DIR.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    # Example class folder: 01_palm
    class_name = path.parent.name

    # Only accept folders matching the dataset's class naming scheme
    if "_" not in class_name:
        continue

    subject_name = path.parent.parent.name

    try:
        with Image.open(path) as img:
            width, height = img.size
            img_mode = img.mode
            img.verify()

        records.append({
            "path": str(path),
            "class": class_name,
            "subject": subject_name,
            "width": width,
            "height": height,
            "mode": img_mode
        })

    except Exception as error:
        print(f"Could not read {path}: {error}")

# --------------------------------------------------
# 3. Check whether we found any labeled images
# --------------------------------------------------

df = pd.DataFrame(records)

if df.empty:
    raise ValueError(
        "No labeled images found. Check the data/raw folder "
        "and the gesture class folder names."
    )

print("\n========== DATASET SUMMARY ==========")
print(f"Total labeled images: {len(df)}")
print(f"Number of gesture classes: {df['class'].nunique()}")
print(f"Number of subject folders: {df['subject'].nunique()}")

# --------------------------------------------------
# 4. Count images in each class
# --------------------------------------------------

class_counts = df["class"].value_counts().sort_index()

print("\n========== IMAGES PER GESTURE ==========")
print(class_counts.to_string())

# Save the class distribution table
class_counts.rename_axis("gesture").reset_index(
    name="image_count"
).to_csv(RESULTS_DIR / "class_counts.csv", index=False)

# --------------------------------------------------
# 5. Plot the class distribution
# --------------------------------------------------

plt.figure(figsize=(12, 6))

class_counts.plot(kind="bar")

plt.title("Number of Images per Hand Gesture")
plt.xlabel("Gesture Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_distribution.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 6. Examine image dimensions
# --------------------------------------------------

print("\n========== IMAGE DIMENSIONS ==========")
print(
    df.groupby(["width", "height"])
    .size()
    .sort_values(ascending=False)
    .head(10)
    .to_string()
)

print("\n========== IMAGE MODES ==========")
print(df["mode"].value_counts().to_string())

# --------------------------------------------------
# 7. Plot a gallery of sample images
# --------------------------------------------------

classes = sorted(df["class"].unique())

num_classes = len(classes)

fig, axes = plt.subplots(
    num_classes,
    3,
    figsize=(10, max(4, num_classes * 2.2)),
    squeeze=False
)

for row, class_name in enumerate(classes):
    samples = (
        df[df["class"] == class_name]
        .sample(n=min(3, len(df[df["class"] == class_name])),
                random_state=42)
    )

    for col in range(3):
        ax = axes[row, col]

        if col < len(samples):
            image_path = samples.iloc[col]["path"]

            with Image.open(image_path) as img:
                ax.imshow(np.array(img), cmap="gray")
            
            ax.set_title(class_name)
        else:
            ax.axis("off")

        ax.axis("off")

plt.suptitle("Sample Images by Gesture Class", y=1.002)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "sample_gallery.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------------------
# 8. Create a concise text report
# --------------------------------------------------

report_lines = [
    "HAND GESTURE DATASET EDA REPORT",
    "=" * 35,
    f"Total labeled images: {len(df)}",
    f"Number of classes: {df['class'].nunique()}",
    f"Number of subject folders: {df['subject'].nunique()}",
    "",
    "Images per class:",
    class_counts.to_string(),
    "",
    "Image dimensions:",
    df.groupby(["width", "height"])
      .size()
      .sort_values(ascending=False)
      .head(10)
      .to_string(),
    "",
    "Image modes:",
    df["mode"].value_counts().to_string()
]

(RESULTS_DIR / "eda_report.txt").write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nEDA completed!")
print("Check the results/ folder for charts and reports.")
