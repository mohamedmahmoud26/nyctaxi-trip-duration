import os
import json

# Folder containing ZIP files
data_folder = os.getcwd()

# List of ZIP files
zip_files = ["train.zip", "val.zip", "test.zip"]

# Generate metadata.json
metadata = {
    "version": 0,
    "description": "NYC Taxi Trip Duration dataset - split into train, validation, and test sets",
    "files": {},
    "created_at": "2025-08-30"
}

for file_name in zip_files:
    file_path = os.path.join(data_folder, file_name)
    if os.path.exists(file_path):
        size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
        metadata["files"][file_name] = {
            "rows": "unknown",
            "size_MB": size_mb
        }
    else:
        metadata["files"][file_name] = "File not found"

# Save metadata.json next to ZIPs
metadata_path = os.path.join(data_folder, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata saved to {metadata_path}")
