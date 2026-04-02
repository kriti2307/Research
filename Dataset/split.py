import os
import shutil
import random

# path where your spectrograms currently are
SOURCE_DIR = "../spectrograms"

# new split folder
DEST_DIR = "../data_split"

# split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# create main folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

# process each class
for cls in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # create class folders inside each split
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

    # copy files
    def copy_files(file_list, split):
        for file in file_list:
            src = os.path.join(class_path, file)
            dst = os.path.join(DEST_DIR, split, cls, file)
            shutil.copy(src, dst)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(f"✅ Done class: {cls}")

print("\n🔥 DATA SPLIT COMPLETE")