import os
import re

folder_path = "dataset/ai/"

pattern = re.compile(r"^(\d{4})\.jpg$", re.IGNORECASE)

max_index = -1

for f in os.listdir(folder_path):
    match = pattern.match(f)
    if match:
        num = int(match.group(1))
        if num > max_index:
            max_index = num

start_index = max_index + 1
print(f"Starting index: {start_index:04d}")

jfif_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jfif")]
jfif_files.sort()

for i, filename in enumerate(jfif_files):
    old_path = os.path.join(folder_path, filename)
    new_index = start_index + i
    new_name = f"{new_index:04d}.jpg"
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed: {filename} -> {new_name}")

print("Done renaming files")
