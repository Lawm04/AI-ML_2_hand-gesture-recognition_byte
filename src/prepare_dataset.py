from pathlib import Path
import re
import pandas as pd

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
class_pattern = re.compile(r"^\d+_.+")

records = []

for path in DATA_DIR.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    class_dir = path.parent

    # Expected structure: subject_folder / gesture_folder / image
    if not class_pattern.match(class_dir.name):
        continue

    subject = class_dir.parent.name
    gesture = class_dir.name

    records.append({
        "image_path": str(path),
        "gesture": gesture,
        "subject": subject,
    })

if not records:
    raise SystemExit(
        "No labeled images found. Check the extracted dataset folders "
        "inside data/raw."
    )

df = pd.DataFrame(records)
subjects = sorted(df["subject"].unique())

if len(subjects) < 2:
    raise SystemExit(
        "Need at least two subject folders for a subject-independent split."
    )

# Keep each subject entirely in either train or test.
# With 10 subjects, this uses 8 for training and 2 for testing.
n_train = max(1, min(len(subjects) - 1, round(len(subjects) * 0.8)))
train_subjects = set(subjects[:n_train])

df["split"] = df["subject"].apply(
    lambda subject: "train" if subject in train_subjects else "test"
)

output_file = OUTPUT_DIR / "dataset_split.csv"
df.to_csv(output_file, index=False)

print(f"Images found: {len(df)}")
print(f"Gesture classes: {df['gesture'].nunique()}")
print(f"Subjects found: {len(subjects)}")
print(f"Training subjects: {sorted(train_subjects)}")
print(f"Testing subjects: {sorted(set(subjects) - train_subjects)}")
print("\nImages per split:")
print(df["split"].value_counts().to_string())
print("\nImages per gesture and split:")
print(pd.crosstab(df["gesture"], df["split"]).to_string())
print(f"\nSaved dataset manifest to: {output_file}")
