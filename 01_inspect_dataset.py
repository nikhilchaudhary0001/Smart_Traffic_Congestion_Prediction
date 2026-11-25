from pathlib import Path
import zipfile
from collections import Counter


# ============================================================
# STEP 1: Locate the dataset ZIP file
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_DIR / "archive (1).zip"


print("=" * 70)
print("SMART TRAFFIC CONGESTION PREDICTION")
print("DATASET INITIAL INSPECTION")
print("=" * 70)


# ============================================================
# STEP 2: Check whether the ZIP file exists
# ============================================================

print("\n[1] Checking dataset file...")

if not ZIP_PATH.exists():
    print("ERROR: Dataset ZIP file was not found.")
    print(f"Expected location: {ZIP_PATH}")
    raise FileNotFoundError(ZIP_PATH)

print("Dataset found:")
print(ZIP_PATH)
print(f"File size: {ZIP_PATH.stat().st_size / (1024 * 1024):.2f} MB")


# ============================================================
# STEP 3: Open and verify the ZIP file
# ============================================================

print("\n[2] Checking ZIP integrity...")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

    bad_file = zip_file.testzip()

    if bad_file is None:
        print("ZIP integrity check: PASSED")
    else:
        print(f"WARNING: Corrupted file found: {bad_file}")
        raise RuntimeError(f"Corrupted file inside ZIP: {bad_file}")


    # ========================================================
    # STEP 4: List every file inside the ZIP
    # ========================================================

    all_files = [
        name for name in zip_file.namelist()
        if not name.endswith("/")
    ]

    print(f"\n[3] Total files inside ZIP: {len(all_files)}")

    print("\nFiles inside the dataset:")
    print("-" * 70)

    for i, file_name in enumerate(all_files, start=1):
        print(f"{i:02d}. {file_name}")


    # ========================================================
    # STEP 5: Count file types
    # ========================================================

    extensions = []

    for file_name in all_files:
        suffix = Path(file_name).suffix.lower()

        if suffix:
            extensions.append(suffix)
        else:
            extensions.append("[no extension]")

    extension_counts = Counter(extensions)

    print("\n[4] File type summary:")
    print("-" * 70)

    for extension, count in sorted(extension_counts.items()):
        print(f"{extension:20s} : {count}")


    # ========================================================
    # STEP 6: Show top-level folders
    # ========================================================

    print("\n[5] Top-level folders:")
    print("-" * 70)

    top_level_items = set()

    for file_name in all_files:
        parts = Path(file_name).parts

        if len(parts) > 1:
            top_level_items.add(parts[0])
        else:
            top_level_items.add("[ROOT FILE]")

    for item in sorted(top_level_items):
        print(item)


    # ========================================================
    # STEP 7: Count files by top-level folder
    # ========================================================

    folder_counts = Counter()

    for file_name in all_files:
        parts = Path(file_name).parts

        if len(parts) > 1:
            folder_counts[parts[0]] += 1
        else:
            folder_counts["[ROOT FILE]"] += 1

    print("\n[6] Files by top-level folder:")
    print("-" * 70)

    for folder, count in sorted(folder_counts.items()):
        print(f"{folder:30s} : {count}")


print("\n" + "=" * 70)
print("INITIAL DATASET INSPECTION COMPLETED")
print("=" * 70)