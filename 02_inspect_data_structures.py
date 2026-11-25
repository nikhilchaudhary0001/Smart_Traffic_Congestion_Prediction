from pathlib import Path
import zipfile
import json
import pandas as pd


# ============================================================
# STEP 1: Locate project and ZIP file
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_DIR / "archive (1).zip"


print("=" * 80)
print("SMART TRAFFIC CONGESTION PREDICTION")
print("DATA STRUCTURE INSPECTION")
print("=" * 80)


# ============================================================
# STEP 2: Open ZIP
# ============================================================

with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

    all_files = [
        name for name in zip_file.namelist()
        if not name.endswith("/")
    ]

    # --------------------------------------------------------
    # Identify important file groups
    # --------------------------------------------------------

    probe_files = [
        f for f in all_files
        if "probe_counts/geojson/" in f
        and f.endswith(".geojson")
    ]

    csv_files = [
        f for f in all_files
        if f.endswith(".csv")
    ]

    json_files = [
        f for f in all_files
        if f.endswith(".json")
    ]

    print("\n[1] DATA FILE GROUPS")
    print("-" * 80)

    print(f"Probe-count GeoJSON files : {len(probe_files)}")
    print(f"CSV files                 : {len(csv_files)}")
    print(f"JSON files                : {len(json_files)}")


    # ========================================================
    # STEP 3: Inspect one probe-count GeoJSON
    # ========================================================

    probe_files = sorted(probe_files)

    first_probe_file = probe_files[0]

    print("\n[2] FIRST PROBE-COUNT GEOJSON")
    print("-" * 80)

    print("File:")
    print(first_probe_file)

    with zip_file.open(first_probe_file) as f:
        probe_data = json.load(f)

    print("\nTop-level Python type:")
    print(type(probe_data).__name__)

    if isinstance(probe_data, dict):

        print("\nTop-level keys:")
        for key in probe_data.keys():
            print(f"  - {key}")

        features = probe_data.get("features", [])

    else:
        features = []


    print(f"\nNumber of features: {len(features)}")


    # ========================================================
    # STEP 4: Inspect first few features
    # ========================================================

    print("\n[3] FEATURE STRUCTURE")
    print("-" * 80)

    for index, feature in enumerate(features[:3]):

        print(f"\n--- Feature {index + 1} ---")

        print("Feature type:")
        print(feature.get("type"))

        print("Feature keys:")
        print(list(feature.keys()))

        properties = feature.get("properties", {})

        print("\nProperty keys:")
        print(list(properties.keys()))

        print("\nProperty values/types:")

        for key, value in properties.items():

            if isinstance(value, list):
                print(
                    f"  {key}: list "
                    f"(length={len(value)})"
                )

            elif isinstance(value, dict):
                print(
                    f"  {key}: dictionary "
                    f"(keys={list(value.keys())[:10]})"
                )

            else:
                print(
                    f"  {key}: "
                    f"{type(value).__name__} "
                    f"= {value}"
                )

        geometry = feature.get("geometry")

        print("\nGeometry:")
        if isinstance(geometry, dict):
            print("  Type:", geometry.get("type"))
            print("  Coordinate data present:",
                  "coordinates" in geometry)
        else:
            print("  No geometry")


    # ========================================================
    # STEP 5: Inspect nested hourly probe-count structure
    # ========================================================

    print("\n[4] NESTED HOURLY PROBE-COUNT STRUCTURE")
    print("-" * 80)

    hourly_records_found = False

    for feature in features:

        properties = feature.get("properties", {})

        if "segmentProbeCounts" in properties:

            hourly_records = properties["segmentProbeCounts"]

            print("Found property: segmentProbeCounts")
            print("Python type:",
                  type(hourly_records).__name__)

            if isinstance(hourly_records, list):

                print("Number of hourly records:",
                      len(hourly_records))

                if len(hourly_records) > 0:

                    print("\nFirst hourly record:")

                    print(
                        json.dumps(
                            hourly_records[0],
                            indent=4
                        )
                    )

                hourly_records_found = True

            break

    if not hourly_records_found:
        print("No segmentProbeCounts structure found "
              "in the inspected features.")


    # ========================================================
    # STEP 6: Inspect all probe-count files
    # ========================================================

    print("\n[5] PROBE-COUNT FILE SUMMARY")
    print("-" * 80)

    for file_name in probe_files:

        with zip_file.open(file_name) as f:
            data = json.load(f)

        features = data.get("features", [])

        print(
            f"{Path(file_name).name:65s} "
            f"features={len(features)}"
        )


    # ========================================================
    # STEP 7: Inspect CSV structures
    # ========================================================

    print("\n[6] CSV STRUCTURES")
    print("-" * 80)

    for file_name in sorted(csv_files):

        print(f"\nFile: {Path(file_name).name}")

        with zip_file.open(file_name) as f:

            df = pd.read_csv(f)

        print("Shape:", df.shape)

        print("Columns:")
        for column in df.columns:
            print(f"  - {column}")

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))


    # ========================================================
    # STEP 8: Inspect JSON structures
    # ========================================================

    print("\n[7] JSON STRUCTURES")
    print("-" * 80)

    for file_name in sorted(json_files):

        print(f"\nFile: {Path(file_name).name}")

        with zip_file.open(file_name) as f:
            data = json.load(f)

        print("Top-level type:",
              type(data).__name__)

        if isinstance(data, dict):

            print("Top-level keys:")

            for key in data.keys():
                print(f"  - {key}")

        elif isinstance(data, list):

            print("Number of items:",
                  len(data))

            if len(data) > 0:

                print("First item type:",
                      type(data[0]).__name__)

                if isinstance(data[0], dict):
                    print("First item keys:")
                    for key in data[0].keys():
                        print(f"  - {key}")


print("\n" + "=" * 80)
print("DATA STRUCTURE INSPECTION COMPLETED")
print("=" * 80)