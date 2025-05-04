import os
import shutil
import sys
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import sys 

def organize_images(base_dir):
    raw_dir = os.path.join(base_dir, "dataset_raw")
    jpg_dir = os.path.join(base_dir, "dataset_jpg")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    for filename in os.listdir(base_dir):
        if filename.startswith("IMG_"):
            src = os.path.join(base_dir, filename)
            if filename.lower().endswith(".cr2"):
                shutil.move(src, os.path.join(raw_dir, filename))
            elif filename.lower().endswith(".jpg"):
                shutil.move(src, os.path.join(jpg_dir, filename))

    print("✅ Images organized into dataset_raw and dataset_jpg.")

def get_exif_datetime(filepath):
    """Uses macOS file creation time as the timestamp."""
    try:
        stat = os.stat(filepath)
        return datetime.fromtimestamp(stat.st_birthtime)
    except Exception as e:
        print(f"Error reading creation time for {filepath}: {e}")
        return None


def split_into_folders(dataset_dir, is_raw=True):
    sub_dir_name = "dataset_raw" if is_raw else "dataset_jpg"
    sub_dir_path = os.path.join(dataset_dir, sub_dir_name)

    # Temp folders (outside sub_dir for now)
    temp_out_dirs = {
        0: os.path.join(dataset_dir, "filter_long_exp"),
        1: os.path.join(dataset_dir, "long_exp"),
        2: os.path.join(dataset_dir, "short_exp")
    }

    # Create temp output folders
    for d in temp_out_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"📂 Processing files in: {sub_dir_path}")

    # List only image files (not directories)
    files = [
        f for f in os.listdir(sub_dir_path)
        if os.path.isfile(os.path.join(sub_dir_path, f)) and f.lower().endswith((".jpg", ".jpeg", ".cr2"))
    ]

    print(f"🖼️ Found {len(files)} files.")

    if len(files) % 3 != 0:
        print(f"❌ Dataset needs manual fix: number of files is not divisible by 3.")
        return

    # Sort files by EXIF datetime
    sorted_files = sorted(
        files,
        key=lambda f: get_exif_datetime(os.path.join(sub_dir_path, f)) or datetime.min
    )

    # Distribute files into temp folders
    third = len(sorted_files) // 3
    for idx, filename in enumerate(sorted_files):
        group = min(idx // third, 2)
        src = os.path.join(sub_dir_path, filename)
        dst = os.path.join(temp_out_dirs[group], filename)
        shutil.move(src, dst)  # Now move instead of copy

    # Move the filled folders into sub_dir
    for group_name, folder_path in zip(["filter_long_exp", "long_exp", "short_exp"], temp_out_dirs.values()):
        final_path = os.path.join(sub_dir_path, group_name)
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        shutil.move(folder_path, final_path)

    print(f"✅ Files sorted and original files removed. Folders now inside {sub_dir_name}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python organize_images.py /path/to/your/dataset")
        sys.exit(1)

    dataset_path = os.path.abspath(sys.argv[1])

    if not os.path.isdir(dataset_path):
        print(f"❌ The path '{dataset_path}' is not a valid directory.")
        sys.exit(1)

    organize_images(dataset_path)
    split_into_folders(dataset_path, is_raw=True)
    split_into_folders(dataset_path, is_raw=False)

    # Check lengths of output folders
    folders = ["filter_long_exp", "long_exp", "short_exp"]
    lengths = {}

    print("\n📦 Folder distribution summary:")
    
    # go into sub_dir_path
    sub_dir_path = "dataset/dataset_raw"
    os.chdir(sub_dir_path)
    dataset_path_raw = os.getcwd()

    ### TODO: Check if this is the correct path
    dataset_path_jpg = os.path.join(dataset_path, "dataset/dataset_jpg")

    for folder in folders:
        full_path = os.path.join(dataset_path_raw, folder)
        
        if not os.path.exists(full_path):
            print(f"  ❌ Folder not found: {folder}")
            lengths[folder] = 0
            continue

        # Count only .cr2 files (adjust if needed)
        count = len([
            f for f in os.listdir(full_path)
            if os.path.isfile(os.path.join(full_path, f)) and f.lower().endswith((".cr2", ".jpg", ".jpeg"))
        ])
        
        lengths[folder] = count
        print(f"  - {folder}: {count} files")

    # Only compare counts for existing folders
    existing_counts = [count for count in lengths.values() if count > 0]

    if len(existing_counts) == 3 and len(set(existing_counts)) == 1:
        print("\n✅ All folders contain the same number of files.")
    else:
        print("\n⚠️ Warning: Folders have unequal number of files or some folders are missing.")
