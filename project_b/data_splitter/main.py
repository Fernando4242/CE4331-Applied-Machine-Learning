import edgeimpulse as ei
import os
from datetime import datetime
import random
import shutil
from collections import defaultdict

base_dir = "data/raw-img/"
target_dir = "."
train_ratio = 0.8
total_target_images = 8000

translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider",
}

# Rename folders from Italian to English
for item in os.listdir(base_dir):
    full_path = os.path.join(base_dir, item)
    if os.path.isdir(full_path) and item in translate:
        new_name = translate[item]
        new_path = os.path.join(base_dir, new_name)
        if not os.path.exists(new_path):
            os.rename(full_path, new_path)
            print(f"Renamed '{item}' to '{new_name}'")
        else:
            print(f"Skipping '{item}' -> '{new_name}': target already exists")
    else:
        print(f"Skipping '{item}': not in Italian list or not a directory")

# Gather images per class
class_to_images = defaultdict(list)
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    files = [
        os.path.join(class_path, f)
        for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
    ]
    class_to_images[class_name].extend(files)

# Determine how many per class
num_classes = len(class_to_images)
images_per_class = total_target_images // num_classes

# Balance: Sample images_per_class from each class
balanced_images = []
for class_name, files in class_to_images.items():
    if len(files) < images_per_class:
        raise ValueError(f"Not enough images in class '{class_name}' (needed {images_per_class}, found {len(files)})")
    sampled = random.sample(files, images_per_class)
    balanced_images.extend((class_name, path) for path in sampled)

# Shuffle
random.shuffle(balanced_images)

# Split per class
train_images = []
test_images = []
class_split_counts = defaultdict(lambda: {"train": 0, "test": 0})

class_temp = defaultdict(list)
for cls, path in balanced_images:
    class_temp[cls].append(path)

for cls, paths in class_temp.items():
    test_count = int(images_per_class * (1 - train_ratio))
    test_paths = paths[:test_count]
    train_paths = paths[test_count:]

    test_images.extend((cls, p) for p in test_paths)
    train_images.extend((cls, p) for p in train_paths)

    class_split_counts[cls]["train"] = len(train_paths)
    class_split_counts[cls]["test"] = len(test_paths)

def copy_to_split(split_name, data):
    class_counters = defaultdict(int)
    for cls, path in data:
        ext = os.path.splitext(path)[1]
        count = class_counters[cls]
        new_filename = f"{cls}.{count}{ext}"
        class_counters[cls] += 1

        dest_folder = os.path.join(target_dir, split_name, cls)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, new_filename)

        shutil.copy2(path, dest_path)

# Clean and copy to output folders
train_path = os.path.join(target_dir, "train")
test_path = os.path.join(target_dir, "test")

if os.path.isdir(train_path):
    shutil.rmtree(train_path)
if os.path.isdir(test_path):
    shutil.rmtree(test_path)

copy_to_split("train", train_images)
copy_to_split("test", test_images)

total_train = len(train_images)
total_test = len(test_images)
total = total_train + total_test
train_pct = (total_train / total) * 100
test_pct = (total_test / total) * 100

print(f"\n📊 Overall split: {total_train} train ({train_pct:.1f}%), {total_test} test ({test_pct:.1f}%) — total {total}\n")
print("📂 Per-class breakdown:")
for cls, counts in class_split_counts.items():
    c_train, c_test = counts["train"], counts["test"]
    c_total = c_train + c_test
    t_pct = (c_train / c_total) * 100 if c_total else 0
    v_pct = (c_test / c_total) * 100 if c_total else 0
    print(f"{cls}: {c_train} train ({t_pct:.1f}%), {c_test} test ({v_pct:.1f}%) — total {c_total}")
