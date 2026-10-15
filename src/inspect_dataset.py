
from pathlib import Path
from collections import Counter
from PIL import Image
import re

# 1. Find the raw dataset directory
DATA_DIR = Path("data/raw")

# 2. Image formats we want to inspect
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# 3. Find all image files
image_files = [
    path
    for path in DATA_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Total images found: {len(image_files)}")

# 4. Keep images inside gesture-class folders
class_pattern = re.compile(r"^\d+_.+")

class_counts = Counter()
subject_counts = Counter()
image_sizes = Counter()
invalid_images = []

for path in image_files:
    # The immediate parent folder should be the gesture class
    class_name = path.parent.name

    if not class_pattern.match(class_name):
        continue

    class_counts[class_name] += 1
    subject_counts[path.parent.parent.name] += 1

    # 5. Check that the image can be opened
    try:
        with Image.open(path) as img:
            image_sizes[img.size] += 1
            img.verify()
    except Exception:
        invalid_images.append(str(path))

# 6. Print the dataset summary
print("\nImages per gesture:")
for label, count in sorted(class_counts.items()):
    print(f"{label}: {count}")

print("\nImages per subject folder:")
for subject, count in sorted(subject_counts.items()):
    print(f"{subject}: {count}")

print("\nImage dimensions:")
for size, count in image_sizes.most_common():
    print(f"{size}: {count} images")

print(f"\nInvalid/unreadable images: {len(invalid_images)}")

if invalid_images:
    print("Examples of invalid images:")
    for path in invalid_images[:5]:
        print(path)